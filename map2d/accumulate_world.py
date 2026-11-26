#!/usr/bin/env python3
"""Accumulate our own drives into one world-frame map — the basis for GT.

Why: we currently label *frames*, but the ring road is a loop and the same
100 m of road appears in dozens of frames. Accumulating the clouds into a world
grid means the road can be annotated **once** and every frame that drives over
it inherits the label, at LiDAR accuracy rather than the 1-3 m error of the
hand-drawn lanelet boundaries.

Output per drive-set: a world grid with, per cell,
  zmin / zmax  ground and top height (metres, UTM)
  count        number of returns
  ground       boolean: within --ground-band of the local ground surface

and a rendered hill-shaded PNG at the same resolution, which is what a human
annotator draws polygons on.

  accumulate_world.py --manifest M.json --frames DIR [--frames DIR2] \
                      --out world/ring --res 0.10
"""
import argparse
import json
import pathlib
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from wato2d.bev_geom import ground_surface, sample_grid   # noqa: E402


def world_xy(pts_ego, utm, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    x = pts_ego[:, 0] * c - pts_ego[:, 1] * s + utm[0]
    y = pts_ego[:, 0] * s + pts_ego[:, 1] * c + utm[1]
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", action="append", required=True)
    ap.add_argument("--frames", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=float, default=0.10)
    ap.add_argument("--margin", type=float, default=45.0)
    ap.add_argument("--ground-band", type=float, default=0.20)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    recs = []
    for m in args.manifest:
        recs += json.loads(pathlib.Path(m).read_text())
    if args.limit:
        recs = recs[:args.limit]

    # bounds from the ego track
    us = np.array([[r["utm"][0], r["utm"][1]] for r in recs if "utm" in r])
    if len(us) == 0:
        raise SystemExit("manifest has no poses")
    x0, y0 = us.min(0) - args.margin
    x1, y1 = us.max(0) + args.margin
    W = int((x1 - x0) / args.res) + 2
    H = int((y1 - y0) / args.res) + 2
    print(f"world grid {W} x {H} at {args.res} m  "
          f"({(x1-x0)/1000:.2f} x {(y1-y0)/1000:.2f} km, {W*H/1e6:.1f} Mcells)")

    zmin = np.full(W * H, np.inf, np.float32)
    zmax = np.full(W * H, -np.inf, np.float32)
    cnt = np.zeros(W * H, np.uint16)
    gsum = np.zeros(W * H, np.float32)      # sum of |z - ground| for the band test
    gn = np.zeros(W * H, np.uint16)

    done = 0
    for r in recs:
        fp = pathlib.Path(r["frame_path"])
        if not fp.exists():
            continue
        d = np.load(fp, allow_pickle=True)
        cal_path = fp.parent / "calib.npz"
        if not cal_path.exists():
            continue
        cal = np.load(cal_path, allow_pickle=True)
        pts = d["points"].astype(np.float64)
        l2e = cal["lidar2ego"].astype(np.float64)
        pe = (l2e @ np.concatenate([pts[:, :3], np.ones((len(pts), 1))], 1).T).T[:, :3]
        yaw = float(r["yaw"])
        wx, wy = world_xy(pe, r["utm"], yaw)

        # local ground for the band test (per frame, ego frame)
        g, gc, ge = ground_surface(pe)
        dz = pe[:, 2] - sample_grid((g, gc, ge), pe[:, 0], pe[:, 1])

        ci = ((wx - x0) / args.res).astype(np.int64)
        ri = ((wy - y0) / args.res).astype(np.int64)
        ok = (ci >= 0) & (ci < W) & (ri >= 0) & (ri < H)
        ci, ri, zz, dd = ci[ok], ri[ok], pe[ok, 2], dz[ok]
        flat = ri * W + ci
        np.minimum.at(zmin, flat, zz.astype(np.float32))
        np.maximum.at(zmax, flat, zz.astype(np.float32))
        np.add.at(cnt, flat, 1)
        band = np.abs(dd) < args.ground_band
        np.add.at(gsum, flat[band], np.abs(dd[band]).astype(np.float32))
        np.add.at(gn, flat[band], 1)
        done += 1
        if done % 20 == 0:
            print(f"  {done} frames", flush=True)

    occ = cnt > 0
    zmin[~occ] = np.nan
    zmax[~occ] = np.nan
    ground = (gn > 2) & (gsum / np.maximum(gn, 1) < args.ground_band * 0.5)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out.with_suffix(".npz"),
        zmin=zmin.reshape(H, W), zmax=zmax.reshape(H, W),
        cnt=cnt.reshape(H, W), ground=ground.reshape(H, W),
        origin=np.array([x0, y0]), res=np.float64(args.res),
    )
    print(f"occupied cells: {occ.sum()/1e6:.2f} M of {occ.size/1e6:.1f} M  "
          f"({100*occ.mean():.1f}%)")

    # render: height of the *ground* where known, else the top surface, hillshaded
    zg = zmin.reshape(H, W)
    zt = zmax.reshape(H, W)
    h = np.where(np.isfinite(zg), zg, np.nan)
    valid = np.isfinite(h)
    if valid.any():
        lo, hi = np.nanpercentile(h, [2, 98])
        n = np.clip((h - lo) / max(hi - lo, 1e-6), 0, 1)
        # hillshade from the gradient
        gy, gx = np.gradient(np.nan_to_num(h, nan=np.nanmedian(h)))
        shade = np.clip(0.6 - 2.5 * (gx + gy), 0.2, 1.4)
        rgb = np.zeros((H, W, 3), np.uint8)
        rgb[..., 0] = np.clip(255 * n * shade, 0, 255)
        rgb[..., 1] = np.clip(255 * (0.35 + 0.65 * n) * shade, 0, 255)
        rgb[..., 2] = np.clip(255 * (0.8 - 0.6 * n) * shade, 0, 255)
        rgb[~valid] = 0
        rgb[ground.reshape(H, W)] = (rgb[ground.reshape(H, W)] * 0.5
                                     + np.array([40, 90, 160]) * 0.5).astype(np.uint8)
        im = Image.fromarray(rgb[::-1])          # flip so +y is up
        scale = max(1, int(round(im.width / 2400)))
        im.resize((im.width // scale, im.height // scale), Image.LANCZOS).save(
            out.with_suffix(".png"))
        print(f"canvas -> {out.with_suffix('.png')}  ({im.width}x{im.height}, "
              f"downsampled /{scale})")


if __name__ == "__main__":
    main()
