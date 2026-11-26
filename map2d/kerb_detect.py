#!/usr/bin/env python3
"""Kerb detection from an accumulated world height map.

Why this matters for GT: the lanelet maps give us no walkway class, and their
road boundaries are 1-3 m off the paint. A kerb, on the other hand, is a
physical 10-20 cm step that the LiDAR measures directly — and the raised side of
a kerb *is* the walkway. So this is the cheapest path to walkway labels that
matches reality.

Method: on the world ground surface, look for cells where the height differs
across a 4-cell window by a small-but-real step (kerbs are 3-30 cm; walls and
buildings are metres, so they are excluded by the upper bound). Threshold,
thin, and vectorise by tracing.

  kerb_detect.py world/ringday.npz --out world/kerbs [--window 4]
"""
import argparse
import json
import pathlib

import numpy as np
from PIL import Image


def fill_small_holes(a, passes=3):
    a = a.astype(np.float32)
    for _ in range(passes):
        bad = ~np.isfinite(a)
        if not bad.any():
            break
        pad = np.pad(a, 1, constant_values=np.nan)
        st = np.stack([pad[1 + dr:a.shape[0] + 1 + dr, 1 + dc:a.shape[1] + 1 + dc]
                       for dr in (-1, 0, 1) for dc in (-1, 0, 1)])
        with np.errstate(all="ignore"):
            f = np.nanmedian(st, axis=0)
        a = np.where(bad, f, a)
    return a


def step_height(z, w):
    """Max |dz| across a window along x and y, and the direction of the step."""
    def shift(a, dy, dx, fill):
        out = np.full_like(a, fill)
        ys0, ys1 = max(0, dy), a.shape[0] + min(0, dy)
        xs0, xs1 = max(0, dx), a.shape[1] + min(0, dx)
        out[ys0:ys1, xs0:xs1] = a[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
        return out

    best = np.zeros_like(z)
    ang = np.zeros_like(z)
    big = np.nanmax(z) + 10.0
    for dy, dx, a in ((0, w, 0.0), (0, -w, 0.0), (w, 0, np.pi / 2), (-w, 0, np.pi / 2),
                      (w, w, np.pi / 4), (-w, -w, np.pi / 4), (w, -w, -np.pi / 4), (-w, w, -np.pi / 4)):
        d = np.abs(shift(z, dy, dx, big) - z)
        m = np.isfinite(d) & (d > best)
        best = np.where(m, d, best)
        ang = np.where(m, a, ang)
    return best, ang


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("world")
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--min-step", type=float, default=0.04)
    ap.add_argument("--max-step", type=float, default=0.30)
    ap.add_argument("--min-returns", type=int, default=3)
    args = ap.parse_args()

    d = np.load(args.world, allow_pickle=True)
    z = d["zmin"].astype(np.float32)
    cnt = d["cnt"]
    res = float(d["res"])
    origin = d["origin"]
    print(f"grid {z.shape} at {res} m, {np.isfinite(z).mean()*100:.1f}% observed")

    z = fill_small_holes(z, passes=4)
    step, ang = step_height(z, args.window)

    # a kerb is a small step between two well-observed ground cells
    enough = (cnt >= args.min_returns)
    kerb = (step >= args.min_step) & (step <= args.max_step) & enough & np.isfinite(z)
    print(f"kerb candidates: {kerb.sum()} cells "
          f"({kerb.sum() * res * res / 1000:.2f} thousand m of edge at one cell wide)")

    # a kerb separates two *flat* surfaces. Confetti comes from noisy cells on
    # rough ground, so require each side of the step to be locally flat.
    def local_flat(a, r=2, thr=0.05):
        a2 = np.nan_to_num(a, nan=np.nanmedian(a))
        pad = np.pad(a2, r, mode="edge")
        st = np.stack([pad[dr:dr + a.shape[0], dc:dc + a.shape[1]]
                       for dr in range(2 * r + 1) for dc in range(2 * r + 1)])
        return np.nanstd(st, axis=0) < thr
    flat = local_flat(z)
    kerb &= flat

    # thin: non-maximum suppression on the *magnitude*, not on the mask
    mag = np.where(kerb, step, 0.0)
    pad = np.pad(mag, 1)
    nms = np.ones_like(mag, bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nms &= mag >= pad[1 + dr:mag.shape[0] + 1 + dr, 1 + dc:mag.shape[1] + 1 + dc]
    kerb_thin = kerb & nms
    print(f"after thinning: {kerb_thin.sum()} cells")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out.with_suffix(".npz"),
                        kerb=kerb_thin, step=step.astype(np.float16),
                        angle=ang.astype(np.float16),
                        origin=origin, res=res)
    json.dump(dict(grid=list(z.shape), res=res,
                   candidates=int(kerb.sum()), thinned=int(kerb_thin.sum()),
                   window=args.window, min_step=args.min_step, max_step=args.max_step),
              open(out.with_suffix(".json"), "w"), indent=2)

    # render: grey ground, red kerbs
    lo, hi = np.nanpercentile(z, [2, 98])
    n = np.clip((z - lo) / max(hi - lo, 1e-6), 0, 1)
    rgb = np.zeros((*z.shape, 3), np.uint8)
    rgb[..., 0] = (60 + 100 * n).astype(np.uint8)
    rgb[..., 1] = (60 + 100 * n).astype(np.uint8)
    rgb[..., 2] = (60 + 100 * (1 - n)).astype(np.uint8)
    rgb[~np.isfinite(z)] = 0
    rgb[kerb_thin] = (255, 40, 40)
    im = Image.fromarray(rgb[::-1])
    s = max(1, int(round(im.width / 2400)))
    im.resize((im.width // s, im.height // s), Image.LANCZOS).save(out.with_suffix(".png"))
    print(f"-> {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
