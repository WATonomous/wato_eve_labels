#!/usr/bin/env python3
"""Stage 2: turn the BEV ground truth into per-camera image labels.

No new human labelling. The rasterised BEV mask already says which 0.5 m cells
are drivable / walkway / crossing; this projects each of those cells onto its
camera using the **LiDAR-measured ground surface**, so the label lands on the
right pixel even where the road is not flat.

Two things it is good for:

  1. supervision for the 2D branch (sparse but correct),
  2. a projection check — if the labels do not land on road in the image, the
     calibration is wrong, and you find that out in one picture.

Output per frame: one uint8 PNG per camera, same size as the image, values in
`wato2d.classes.CLASSES`, 255 = unlabelled.

  stage2_labels.py --frames DIR --masks DIR --manifest JSON --out DIR \
                   [--cams 8] [--downscale 4] [--overlay 3]
"""
import argparse
import json
import pathlib
import sys

import math

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from wato2d.classes import BEV_CLASSES, CLASSES          # noqa: E402

RES, EXTENT = 0.5, 50.0
SIZE = int(2 * EXTENT / RES)

# BEV channel -> 2D class (carpark counts as drivable, stop_line/divider are too
# thin to be visible labels in an image and are left unlabelled)
BEV_TO_2D = {
    "drivable_area": "drivable",
    "carpark_area": "drivable",
    "walkway": "walkway",
    "ped_crossing": "crossing",
}


def ground_surface(pts_ego, cell=2.0, extent=60.0):
    n = int(2 * extent / cell)
    c = np.clip(((pts_ego[:, 0] + extent) / cell).astype(int), 0, n - 1)
    r = np.clip(((extent - pts_ego[:, 1]) / cell).astype(int), 0, n - 1)
    flat = r * n + c
    order = np.argsort(flat, kind="stable")
    fs, zs = flat[order], pts_ego[order, 2]
    starts = np.searchsorted(fs, np.arange(n * n), side="left")
    ends = np.searchsorted(fs, np.arange(n * n), side="right")
    g = np.full(n * n, np.nan, np.float32)
    for i in np.flatnonzero(ends > starts):
        g[i] = np.percentile(zs[starts[i]:ends[i]], 10)
    g = g.reshape(n, n)
    for _ in range(8):
        bad = ~np.isfinite(g)
        if not bad.any():
            break
        pad = np.pad(g, 1, constant_values=np.nan)
        st = np.stack([pad[1 + dr:n + 1 + dr, 1 + dc:n + 1 + dc]
                       for dr in (-1, 0, 1) for dc in (-1, 0, 1)])
        with np.errstate(all="ignore"):
            fill = np.nanmedian(st, axis=0)
        g = np.where(bad, fill, g)
    if not np.isfinite(g).all():
        g = np.where(np.isfinite(g), g, np.nanmedian(g))
    pad = np.pad(g, 2, mode="edge")
    g = np.median(np.lib.stride_tricks.sliding_window_view(pad, (5, 5)), axis=(2, 3))
    return g, cell, extent


def sample(g, cell, extent, x, y):
    n = g.shape[0]
    c = np.clip(((x + extent) / cell).astype(int), 0, n - 1)
    r = np.clip(((extent - y) / cell).astype(int), 0, n - 1)
    return g[r, c]


def occupancy_grid(pts_ego, ground, cell=0.5, extent=EXTENT, min_residual=0.30):
    """Per BEV cell, the tallest return above the ground surface.

    Used to *not* label cells that are occupied (the car in front, a parked van,
    a sign). Without this the projected labels paint over vehicles and teach the
    segmenter that cars are drivable.
    """
    g, gcell, gext = ground
    m = np.abs(pts_ego).max(1) < extent
    p = pts_ego[m]
    if len(p) == 0:
        return np.zeros((SIZE, SIZE), np.float32)
    z0 = sample(g, gcell, gext, p[:, 0], p[:, 1])
    res = p[:, 2] - z0
    n = SIZE
    c = np.clip(((p[:, 0] + extent) / cell).astype(int), 0, n - 1)
    r = np.clip(((extent - p[:, 1]) / cell).astype(int), 0, n - 1)
    occ = np.zeros(n * n, np.float32)
    np.maximum.at(occ, r * n + c, res.astype(np.float32))
    return occ.reshape(n, n)


def label_camera(mask, ground, c2e, K, W, H, downscale=1, max_range=45.0,
                 max_quad_px=4000, occ=None, occ_max=0.30):
    """mask: (C,200,200) -> label image. Rasterises each BEV cell as a quad.

    A single projected point per 0.5 m cell leaves a dotted label map (and the
    dots are what the network would train on). Projecting the cell's four
    corners and filling the polygon gives solid regions. Cells whose ground
    height was interpolated rather than measured, or whose quad explodes near
    the horizon, are dropped.
    """
    g, cell, extent = ground
    lab = Image.new("L", (W // downscale, H // downscale), 255)
    dr = ImageDraw.Draw(lab)
    inv_c2e = np.linalg.inv(c2e)

    rows, cols = np.nonzero(mask.any(0))
    n = 0
    for r, c in zip(rows, cols):
        x = c * RES - EXTENT
        y = EXTENT - r * RES
        if math.hypot(x, y) > max_range:
            continue
        # class: last non-empty BEV channel wins (they do not overlap in practice)
        if occupied:
            # an object stands here: this is the only place the fine-tune gets
            # negative (non-road) supervision, so label it instead of skipping
            cls = CLASSES.index("other")
        else:
            cls = -1
            for ci, name in enumerate(BEV_CLASSES):
                if name in BEV_TO_2D and mask[ci, r, c] > 0:
                    cls = CLASSES.index(BEV_TO_2D[name])
            if cls < 0:
                continue
        occupied = occ is not None and occ[r, c] > occ_max
        z = float(sample(g, cell, extent, np.array([x]), np.array([y]))[0])
        if not np.isfinite(z):
            continue
        quad = []
        ok = True
        for dx, dy in ((-0.25, -0.25), (0.25, -0.25), (0.25, 0.25), (-0.25, 0.25)):
            P = np.array([x + dx, y + dy, z, 1.0])
            pc = inv_c2e @ P
            if pc[2] < 1.5:
                ok = False
                break
            uvw = K @ (pc[:3] / pc[2])
            quad.append((uvw[0] / downscale, uvw[1] / downscale))
        if not ok:
            continue
        us = [q[0] for q in quad]
        vs = [q[1] for q in quad]
        area = (max(us) - min(us)) * (max(vs) - min(vs))
        if area > max_quad_px:          # grazing angle near the horizon
            continue
        if max(us) < 0 or min(us) > lab.size[0] or max(vs) < 0 or min(vs) > lab.size[1]:
            continue
        dr.polygon(quad, fill=int(cls))
        n += 1
    return np.asarray(lab, np.uint8), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--downscale", type=int, default=4)
    ap.add_argument("--overlay", type=int, default=0, help="write N overlay PNGs per camera")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    man = json.loads(pathlib.Path(args.manifest).read_text())
    pal = np.array([[0, 0, 0], [0, 170, 90], [0, 120, 255], [255, 160, 0]], np.uint8)

    n_done = 0
    tot_px = 0
    for rec in man:
        if args.limit and n_done >= args.limit:
            break
        fpath = pathlib.Path(rec["frame_path"])
        if not fpath.exists():
            continue
        mp = pathlib.Path(args.masks) / f"{rec['id']}.npz"
        if not mp.exists():
            continue
        d = np.load(fpath, allow_pickle=True)
        cal = np.load(fpath.parent / "calib.npz", allow_pickle=True)
        mask = np.load(mp, allow_pickle=True)
        # masks from the lanelet rasteriser only carry the four classes the map
        # supports; walkway/carpark are optional additions
        mask_arr = np.stack([mask[c] if c in mask else np.zeros((SIZE, SIZE), np.uint8)
                             for c in BEV_CLASSES])

        pts = d["points"].astype(np.float64)
        l2e = cal["lidar2ego"].astype(np.float64)
        ph = np.concatenate([pts[:, :3], np.ones((len(pts), 1))], 1)
        pts_ego = (l2e @ ph.T).T[:, :3]
        ground = ground_surface(pts_ego)
        occ = occupancy_grid(pts_ego, ground)

        sess_dir = out / pathlib.Path(fpath).parent.name
        sess_dir.mkdir(parents=True, exist_ok=True)
        stem = fpath.stem
        for ci in range(len(cal["names"])):
            im = Image.open(__import__("io").BytesIO(bytes(d["jpegs"][ci])))
            W, H = im.size
            lab, npix = label_camera(mask_arr, ground, cal["camera2ego"][ci], cal["K"][ci],
                                     W, H, args.downscale, occ=occ)
            np.save(sess_dir / f"{stem}_cam{ci}.npy", lab)
            tot_px += npix
            if args.overlay and n_done < args.overlay:
                rgb = np.asarray(im.convert("RGB")).copy()
                small = rgb[::args.downscale, ::args.downscale]
                m = lab < 255
                small[m] = (0.45 * small[m] + 0.55 * pal[lab[m]]).astype(np.uint8)
                Image.fromarray(small).save(sess_dir / f"{stem}_cam{ci}_overlay.png")
        n_done += 1
        if n_done % 10 == 0:
            print(f"  {n_done} frames, {tot_px} labelled pixels", flush=True)

    print(f"done: {n_done} frames -> {out}  ({tot_px} labelled pixels)")


if __name__ == "__main__":
    main()
