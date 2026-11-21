#!/usr/bin/env python3
"""Rasterise a lanelet2 map into per-frame BEV class masks in the ego frame.

Grid convention mirrors the training pipeline (LoadBEVSegmentation): 200x200
cells at 0.5 m over [-50, 50] m, patch centred on the ego and rotated to its
heading. Rows/cols orientation is the one thing that cannot be read off the code
(a square grid hides a transpose), so --validate picks it empirically against the
LiDAR ground plane.

Classes produced (the three the ring-road map can actually support):
  drivable_area  filled lanelet polygons (lane_type=driving)
  ped_crossing   lanelet polygons with subtype=crosswalk
  divider        ways shared as a boundary by two or more lanelets
  valid          cells with map coverage (dilate); supervise only these, so
                 unmapped ground is not taught as "not drivable"

  lanelet_rasterize.py --map roads_around_campus.osm --frames ringroad_frames \
                       --out ringroad_masks [--flip-y]
  lanelet_rasterize.py --map ... --frames ... --validate --out png/
"""
import argparse
import collections
import math
import pathlib
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw

RES, EXTENT = 0.5, 50.0
SIZE = int(2 * EXTENT / RES)          # 200
CLASSES = ["drivable_area", "ped_crossing", "divider", "stop_line"]


# ----------------------------------------------------------------- WGS84 -> UTM
def utm(lat, lon, zone=17):
    a, f = 6378137.0, 1 / 298.257223563
    k0 = 0.9996
    e2 = 2 * f - f * f
    ep2 = e2 / (1 - e2)
    phi, lam = math.radians(lat), math.radians(lon)
    lam0 = math.radians((zone - 1) * 6 - 180 + 3)
    N = a / math.sqrt(1 - e2 * math.sin(phi) ** 2)
    T = math.tan(phi) ** 2
    C = ep2 * math.cos(phi) ** 2
    A = math.cos(phi) * (lam - lam0)
    M = a * ((1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256) * phi
             - (3 * e2 / 8 + 3 * e2 ** 2 / 32 + 45 * e2 ** 3 / 1024) * math.sin(2 * phi)
             + (15 * e2 ** 2 / 256 + 45 * e2 ** 3 / 1024) * math.sin(4 * phi)
             - (35 * e2 ** 3 / 3072) * math.sin(6 * phi))
    east = k0 * N * (A + (1 - T + C) * A ** 3 / 6
                     + (5 - 18 * T + T ** 2 + 72 * C - 58 * ep2) * A ** 5 / 120) + 500000.0
    north = k0 * (M + N * math.tan(phi) * (A ** 2 / 2 + (5 - T + 9 * C + 4 * C ** 2) * A ** 4 / 24
                                            + (61 - 58 * T + T ** 2 + 600 * C - 330 * ep2) * A ** 6 / 720))
    return east, north


# ------------------------------------------------------------------ map loader
def load_map(path):
    """-> dict(class -> list of (N,2) UTM polyline/polygon), plus bboxes."""
    root = ET.parse(path).getroot()
    nodes = {}
    for n in root.findall("node"):
        lat, lon = n.get("lat"), n.get("lon")
        if not lat or not lon:
            continue
        nodes[n.get("id")] = utm(float(lat), float(lon))
    ways, way_tags = {}, {}
    for w in root.findall("way"):
        pts = [nodes[nd.get("ref")] for nd in w.findall("nd") if nd.get("ref") in nodes]
        if len(pts) >= 2:
            ways[w.get("id")] = np.array(pts, dtype=np.float64)
            way_tags[w.get("id")] = {t.get("k"): t.get("v") for t in w.findall("tag")}

    def tags(e):
        return {t.get("k"): t.get("v") for t in e.findall("tag")}

    lanelets = []           # (polygon, tags, (left_id, right_id))
    for r in root.findall("relation"):
        t = tags(r)
        if t.get("type") != "lanelet":
            continue
        left = right = None
        for m in r.findall("member"):
            if m.get("role") == "left":
                left = m.get("ref")
            elif m.get("role") == "right":
                right = m.get("ref")
        if left in ways and right in ways:
            poly = np.vstack([ways[left], ways[right][::-1]])
            lanelets.append((poly, t, (left, right)))

    boundary_use = collections.Counter()
    for _poly, _t, (l, r) in lanelets:
        boundary_use[l] += 1
        boundary_use[r] += 1
    shared = [wid for wid, c in boundary_use.items() if c > 1 and wid in ways]

    stop_lines = [ways[w] for w, t in way_tags.items() if t.get("type") == "stop_line"]
    # newer Lanelet2 maps (WISE/AVIN, JOSM exports) tag crosswalks as ways
    ped_marks = [ways[w] for w, t in way_tags.items()
                 if t.get("type") in ("pedestrian_marking", "zebra_marking")]
    markings = [ways[w] for w, t in way_tags.items()
                if t.get("type") in ("line_thin", "line_thick")]
    out = {
        "drivable_area": [p for p, t, _ in lanelets
                          if t.get("subtype") not in ("crosswalk",)],
        "ped_crossing": ([p for p, t, _ in lanelets if t.get("subtype") == "crosswalk"]
                         + ped_marks),
        # shared lanelet boundaries plus explicitly tagged lane markings
        "divider": [ways[w] for w in shared] + markings,
        "stop_line": stop_lines,
    }
    for name in CLASSES:                     # bbox per class (lat/lon-free, UTM)
        polys = out[name]
        bbox = (np.array([[p[:, 0].min(), p[:, 1].min(),
                           p[:, 0].max(), p[:, 1].max()] for p in polys])
                if polys else np.zeros((0, 4)))
        out[name] = {"polys": polys, "bbox": bbox}
    out["_all"] = out["drivable_area"]["polys"] + out["ped_crossing"]["polys"]
    return out


def bbox_dist(bb, p):
    """Distance from point p to each axis-aligned bbox (0 if inside)."""
    if len(bb) == 0:
        return np.zeros(0)
    dx = np.maximum(np.maximum(bb[:, 0] - p[0], p[0] - bb[:, 2]), 0.0)
    dy = np.maximum(np.maximum(bb[:, 1] - p[1], p[1] - bb[:, 3]), 0.0)
    return np.hypot(dx, dy)


# ------------------------------------------------------------------ rasterising
def to_pixels(xy, ego, yaw, flip_y=False):
    """UTM -> image pixel coords (col, row), ego frame rotated by -yaw."""
    d = xy - ego
    c, s = math.cos(-yaw), math.sin(-yaw)
    xe = d[:, 0] * c - d[:, 1] * s
    ye = d[:, 0] * s + d[:, 1] * c
    col = (xe + EXTENT) / RES
    row = (EXTENT - ye) / RES if flip_y else (ye + EXTENT) / RES
    return np.stack([col, row], axis=1)


def rasterize(m, ego, yaw, flip_y=False, radius=120.0, valid_dilate_m=6.0):
    masks = {k: np.zeros((SIZE, SIZE), np.uint8) for k in CLASSES}
    # filter on distance to the bbox itself: lanelets are long, so centre
    # distance drops geometry that is right next to the ego
    kee = {name: bbox_dist(m[name]["bbox"], ego) < radius for name in CLASSES}
    for ci, name in enumerate(CLASSES):
        sel = [p for p, k in zip(m[name]["polys"], kee[name]) if k]
        if not sel:
            continue
        img = Image.new("L", (SIZE, SIZE), 0)
        dr = ImageDraw.Draw(img)
        for poly in sel:
            px = to_pixels(poly, ego, yaw, flip_y)
            if name in ("divider", "stop_line"):
                dr.line([tuple(p) for p in px], fill=1, width=2)
            else:
                dr.polygon([tuple(p) for p in px], fill=1)
        masks[name] = np.asarray(img, dtype=np.uint8)

    # valid = lanelet coverage, dilated
    img = Image.new("L", (SIZE, SIZE), 0)
    dr = ImageDraw.Draw(img)
    keep_all = bbox_dist(
        np.array([[p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()]
                  for p in m["_all"]]), ego) < radius
    for poly, k in zip(m["_all"], keep_all):
        if k:
            dr.polygon([tuple(p) for p in to_pixels(poly, ego, yaw, flip_y)], fill=1)
    valid = np.asarray(img, dtype=np.uint8)
    if valid_dilate_m > 0:
        r = max(1, int(round(valid_dilate_m / RES)))
        try:
            from scipy.ndimage import binary_dilation
            valid = binary_dilation(valid.astype(bool),
                                    np.ones((2 * r + 1, 2 * r + 1), bool)).astype(np.uint8)
        except Exception:      # no scipy: fall back to a cheaper box blur
            import numpy.lib.stride_tricks as st
            k = np.ones((2 * r + 1, 2 * r + 1), np.uint8)
            pad = np.pad(valid, r)
            win = st.sliding_window_view(pad, k.shape)
            valid = (win.max(axis=(2, 3)) > 0).astype(np.uint8)
    return masks, valid


# ---------------------------------------------------------------------- checks
def ground_mask_from_points(points, z_thresh=-1.5):
    """LiDAR ground occupancy in the same ego grid (independent of the map)."""
    p = points[np.isfinite(points).all(axis=1)]
    g = p[p[:, 2] < z_thresh]
    img = Image.new("L", (SIZE, SIZE), 0)
    dr = ImageDraw.Draw(img)
    for x, y in g[:, :2]:
        if abs(x) <= EXTENT and abs(y) <= EXTENT:
            dr.point((((x + EXTENT) / RES), ((EXTENT - y) / RES)), fill=1)
    return np.asarray(img, dtype=np.uint8)


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--flip-y", action="store_true")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    m = load_map(args.map)
    print(f"map: drivable={len(m['drivable_area']['polys'])} "
          f"crosswalk={len(m['ped_crossing']['polys'])} "
          f"divider={len(m['divider']['polys'])}")

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(pathlib.Path(args.frames).glob("frame_*.npz"))
    if args.limit:
        files = files[: args.limit]

    stats = []
    for fp in files:
        f = np.load(fp, allow_pickle=True)
        if "utm" not in f:
            raise SystemExit(f"{fp.name} has no utm pose; re-extract with the patched extractor")
        ex, ey, yaw = [float(v) for v in f["utm"]]
        if not np.isfinite(ex):
            print(f"{fp.name}: no pose, skipping")
            continue
        masks, valid = rasterize(m, np.array([ex, ey]), yaw, args.flip_y)
        np.savez_compressed(out / fp.name.replace(".npz", "_mask.npz"),
                            **masks, valid=valid)
        g = ground_mask_from_points(f["points"])
        row = {"frame": fp.name,
               **{k: int(masks[k].sum()) for k in CLASSES},
               "valid": int(valid.sum()),
               "iou_drivable_vs_ground": iou(masks["drivable_area"], g)}
        stats.append(row)
        print(f"{fp.name}: drivable={row['drivable_area']:5d} ped={row['ped_crossing']:3d} "
              f"divider={row['divider']:4d} valid={row['valid']:5d}  "
              f"IoU(drivable, lidar ground)={row['iou_drivable_vs_ground']:.3f}")

        if args.validate:
            # side-by-side: lanelet drivable | lidar ground | overlay
            rgb = np.zeros((SIZE, SIZE, 3), np.uint8)
            rgb[masks["drivable_area"] > 0] = (0, 180, 0)
            rgb[g > 0] = (200, 200, 200)
            both = (masks["drivable_area"] > 0) & (g > 0)
            rgb[both] = (255, 220, 0)
            Image.fromarray(rgb).resize((400, 400), Image.NEAREST).save(
                out / f"{fp.stem}_overlay.png")

    if stats:
        io = np.nanmean([s["iou_drivable_vs_ground"] for s in stats])
        print(f"\nmean IoU(lanelet drivable, lidar ground) = {io:.3f}  "
              f"(flip_y={args.flip_y})")


if __name__ == "__main__":
    main()
