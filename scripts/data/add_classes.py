#!/usr/bin/env python3
"""Add walkway / ped_crossing / carpark_area channels to existing lanelet masks.

The lanelet2 maps only carry drivable, ped_crossing, divider and stop_line, so
`walkway` and `carpark_area` were always empty and silently dropped from mIoU —
which is why our numbers were over 3 or 4 classes instead of nuScenes' 6.

Sources (Region/City of Waterloo open data, downloaded as EPSG:4326 GeoJSON and
converted here to WGS84 UTM 17N so they share the IMU/odom datum):

  Sidewalks.geojson            polylines  -> walkway        (buffered to --sidewalk-width)
  Trails.geojson               polylines  -> walkway        (paved-ish trails)
  Parking_Public_Lots.geojson  polygons   -> carpark_area

Deliberately *not* used: Crosswalks (the layer is empty/unavailable), so
ped_crossing keeps coming from lanelet.

Grid convention is the training pipeline's: 200x200 cells at 0.5 m over
[-50, 50] m, rows +y, cols +x, rotated into the ego heading.

  add_classes.py --manifest MANIFEST.json --mask-dir DIR --data DIR [--dry-run]
"""
import argparse
import json
import math
import pathlib

import numpy as np
from PIL import Image, ImageDraw

RES, EXTENT = 0.5, 50.0
SIZE = int(2 * EXTENT / RES)


# WGS84 ellipsoid -> UTM 17N, matching the IMU (which reports WGS84 UTM)
def utm_17n(lat, lon):
    a, f = 6378137.0, 1 / 298.257223563
    k0, zone = 0.9996, 17
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


def load_lines(path, extent=2000.0, centre=None):
    """GeoJSON -> list of (N,2) UTM polylines near `centre` (metres)."""
    d = json.loads(pathlib.Path(path).read_text())
    out = []
    for feat in d.get("features", []):
        g = feat.get("geometry") or {}
        t = g.get("type")
        chains = []
        if t == "LineString":
            chains = [g["coordinates"]]
        elif t == "MultiLineString":
            chains = g["coordinates"]
        elif t == "Polygon":
            chains = g["coordinates"]
        elif t == "MultiPolygon":
            chains = [ring for poly in g["coordinates"] for ring in poly]
        for ch in chains:
            pts = np.array([utm_17n(c[1], c[0]) for c in ch], dtype=np.float64)
            if len(pts) < 2:
                continue
            if centre is not None:
                if np.min(np.hypot(pts[:, 0] - centre[0], pts[:, 1] - centre[1])) > extent:
                    continue
            out.append(pts)
    return out


def to_pixels(xy, ego, yaw):
    d = xy - ego
    c, s = math.cos(-yaw), math.sin(-yaw)
    return np.stack([d[:, 0] * c - d[:, 1] * s, d[:, 0] * s + d[:, 1] * c], 1)


def rasterise(polys, ego, yaw, mode="line", width_m=1.8, radius=70.0):
    img = Image.new("L", (SIZE, SIZE), 0)
    dr = ImageDraw.Draw(img)
    wpx = max(1, int(round(width_m / RES)))
    drew = 0
    for p in polys:
        if np.min(np.hypot(p[:, 0] - ego[0], p[:, 1] - ego[1])) > radius:
            continue
        px = to_pixels(p, ego, yaw)
        col = (px[:, 0] + EXTENT) / RES
        row = (EXTENT - px[:, 1]) / RES
        pts = np.stack([col, row], 1)
        if mode == "line":
            dr.line([tuple(q) for q in pts], fill=1, width=wpx)
        else:
            dr.polygon([tuple(q) for q in pts], fill=1)
        drew += 1
    return np.asarray(img, dtype=np.uint8), drew


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--mask-dir", required=True)
    ap.add_argument("--data", required=True, help="dir with the *.geojson files")
    ap.add_argument("--sidewalk-width", type=float, default=1.8)
    ap.add_argument("--osm", default="", help="OSM extract (osm_paths.geojson) for campus/paths")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    data = pathlib.Path(args.data)
    man = json.loads(pathlib.Path(args.manifest).read_text())
    print(f"{len(man)} frames")

    # cache the layers; the whole region is ~25 MB so just load it once
    sw = load_lines(data / "Sidewalks.geojson") if (data / "Sidewalks.geojson").exists() else []
    tr = load_lines(data / "Trails.geojson") if (data / "Trails.geojson").exists() else []
    pk = load_lines(data / "Parking_Public_Lots.geojson") if (data / "Parking_Public_Lots.geojson").exists() else []
    osw, opk = [], []
    if args.osm:
        import json as _j
        _d = _j.loads(pathlib.Path(args.osm).read_text())
        for feat in _d.get("features", []):
            kind = (feat.get("properties") or {}).get("kind")
            one = {"type": "FeatureCollection", "features": [feat]}
            tmp = pathlib.Path("/tmp/_one.geojson"); tmp.write_text(_j.dumps(one))
            if kind in ("footway", "path", "pedestrian", "steps", "cycleway", "living_street"):
                osw += load_lines(tmp)
            elif kind == "parking":
                opk += load_lines(tmp)
        print(f"osm: paths={len(osw)} parking={len(opk)}")
    print(f"layers: sidewalks={len(sw)} trails={len(tr)} parking_lots={len(pk)}")

    mdir = pathlib.Path(args.mask_dir)
    stats = {"walkway": [], "carpark_area": []}
    for i, r in enumerate(man):
        mp = mdir / f"{r['id']}.npz"
        if not mp.exists():
            continue
        ego = np.array(r["utm"][:2], dtype=np.float64)
        yaw = float(r["yaw"])
        w1, n1 = rasterise(sw + tr + osw, ego, yaw, "line", args.sidewalk_width)
        w2, n2 = rasterise(pk + opk, ego, yaw, "poly", radius=90.0)
        stats["walkway"].append(int(w1.sum()))
        stats["carpark_area"].append(int(w2.sum()))
        if args.dry_run:
            continue
        m = dict(np.load(mp, allow_pickle=True))
        # walkway: keep whatever the lanelet rasteriser had, add the municipal layer
        m["walkway"] = np.maximum(m.get("walkway", np.zeros((SIZE, SIZE), np.uint8)), w1)
        m["carpark_area"] = np.maximum(m.get("carpark_area", np.zeros((SIZE, SIZE), np.uint8)), w2)
        np.savez_compressed(mp, **m)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(man)}", flush=True)

    for k, v in stats.items():
        if v:
            arr = np.array(v)
            print(f"{k:14s} cells: mean {arr.mean():7.0f}  non-empty {int((arr > 0).sum())}/{len(arr)}")
    print("dry run, nothing written" if args.dry_run else "masks updated")


if __name__ == "__main__":
    main()
