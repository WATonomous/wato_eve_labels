#!/usr/bin/env python3
"""Stage 4: rebuild the BEV from the 2D predictions and score it.

This is the experiment that decides whether the LiDAR ground surface earns its
keep. The same 2D probabilities are warped twice:

  flat : every BEV cell is assumed to sit at z = const
  dem  : each cell sits at the height the LiDAR actually measured

and both are scored against the rasterised ground truth, day and night
separately. If `dem` does not beat `flat` on slopes and at range, the LiDAR is
not buying us anything and we should stop paying for it.

  stage4_bev.py --frames DIR --masks DIR --manifest JSON --ckpt CKPT \
                --out DIR [--modes flat,dem] [--sessions day,night]
"""
import argparse
import io
import json
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from wato2d.classes import BEV_CLASSES, CLASSES, NUM_CLASSES   # noqa: E402
from wato2d.models.segformer import build_model                # noqa: E402

RES, EXTENT = 0.5, 50.0
SIZE = int(2 * EXTENT / RES)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
THRS = np.arange(0.05, 0.96, 0.05)

# 2D class -> which BEV channels it supports
TO_BEV = {
    "drivable": ["drivable_area", "carpark_area"],
    "walkway": ["walkway"],
    "crossing": ["ped_crossing"],
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


def sample_grid(g, cell, extent, x, y):
    n = g.shape[0]
    c = np.clip(((x + extent) / cell).astype(int), 0, n - 1)
    r = np.clip(((extent - y) / cell).astype(int), 0, n - 1)
    return g[r, c]


@torch.no_grad()
def segment(model, arr, size=(256, 320)):
    x = (arr.astype(np.float32) / 255.0 - MEAN) / STD
    t = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).cuda()
    if t.shape[-2:] != size:
        t = F.interpolate(t, size=size, mode="bilinear", align_corners=False)
    with torch.autocast("cuda", dtype=torch.float16):
        logits = model(t)
    p = logits.float().softmax(1)[0]                 # (C,h,w)
    return F.interpolate(p.unsqueeze(0), size=(arr.shape[0], arr.shape[1]),
                         mode="bilinear", align_corners=False)[0]


def warp_bev(probs, c2e, K, xs, ys, zs_list, max_range=45.0):
    """Backward-project BEV cells into every camera and average the 2D probs."""
    ncell = len(xs)
    acc = np.zeros((NUM_CLASSES, ncell), np.float32)
    cnt = np.zeros(ncell, np.float32)
    for prob, T, Kc, zs in zip(probs, c2e, K, zs_list):
        H, W = prob.shape[-2:]          # image height/width, not the BEV size
        # for each candidate height, project and sample; a cell uses the height
        # supplied by the caller (flat or DEM), so z is already resolved here
        P = np.stack([xs, ys, zs], 1)
        Ph = np.concatenate([P, np.ones((ncell, 1))], 1)
        pc = (np.linalg.inv(T) @ Ph.T).T
        ok = pc[:, 2] > 1.5
        if not ok.any():
            continue
        u = (Kc[0, 0] * pc[:, 0] + Kc[0, 1] * pc[:, 1] + Kc[0, 2] * pc[:, 2]) / pc[:, 2]
        v = (Kc[1, 0] * pc[:, 0] + Kc[1, 1] * pc[:, 1] + Kc[1, 2] * pc[:, 2]) / pc[:, 2]
        inb = ok & (u > 0) & (u < W - 1) & (v > 0) & (v < H - 1)
        if not inb.any():
            continue
        gx = (u[inb] / W) * 2 - 1
        gy = (v[inb] / H) * 2 - 1
        grid = torch.from_numpy(np.stack([gx, gy], 1)[None, None].astype(np.float32)).cuda()
        pr = prob.unsqueeze(0).to(grid.device)       # (1,C,H,W)
        s = F.grid_sample(pr, grid, mode="bilinear", align_corners=False,
                          padding_mode="border")[0, :, 0, :].cpu().numpy()
        acc[:, inb] += s
        cnt[inb] += 1
    out = np.zeros((NUM_CLASSES, ncell), np.float32)
    m = cnt > 0
    out[:, m] = acc[:, m] / cnt[m]
    return out, m


def to_bev_channels(prob2d, valid):
    """2D class probs -> the 6 BEV channels the GT uses."""
    p = {c: prob2d[CLASSES.index(c)] for c in CLASSES}
    out = np.zeros((len(BEV_CLASSES), prob2d.shape[1]), np.float32)
    for c, chans in TO_BEV.items():
        for ch in chans:
            out[BEV_CLASSES.index(ch)] = p[c]
    return out, valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--modes", default="flat,dem")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    model = build_model().cuda().eval()
    sd = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(sd["model"] if "model" in sd else sd)
    print(f"checkpoint {args.ckpt}", flush=True)

    recs = json.loads(pathlib.Path(args.manifest).read_text())
    if args.limit:
        recs = recs[:args.limit]
    modes = args.modes.split(",")
    stats = {(m, "day" if "night" not in r["session"] else "night"): None
             for m in modes for r in recs}
    tp = {k: np.zeros((len(BEV_CLASSES), len(THRS))) for k in stats}
    fp = {k: np.zeros((len(BEV_CLASSES), len(THRS))) for k in stats}
    fn = {k: np.zeros((len(BEV_CLASSES), len(THRS))) for k in stats}
    present = {k: np.zeros(len(BEV_CLASSES)) for k in stats}

    zspreads = []
    rows, cols = np.mgrid[0:SIZE, 0:SIZE]
    xs = (cols * RES - EXTENT).ravel()
    ys = (EXTENT - rows * RES).ravel()
    rng = np.hypot(xs, ys)

    for fi, rec in enumerate(recs):
        fp_path = pathlib.Path(rec["frame_path"])
        mp = None
        for d in args.masks.split(","):
            cand = pathlib.Path(d) / f"{rec['id']}.npz"
            if cand.exists():
                mp = cand
                break
        if not fp_path.exists() or mp is None:
            continue
        d = np.load(fp_path, allow_pickle=True)
        cal = np.load(fp_path.parent / "calib.npz", allow_pickle=True)
        mask = np.load(mp, allow_pickle=True)
        gt = np.stack([mask[c] if c in mask else np.zeros((SIZE, SIZE), np.uint8)
                       for c in BEV_CLASSES]) > 0
        valid = mask["valid"].astype(bool) if "valid" in mask else np.ones((SIZE, SIZE), bool)

        pts = d["points"].astype(np.float64)
        l2e = cal["lidar2ego"].astype(np.float64)
        ph = np.concatenate([pts[:, :3], np.ones((len(pts), 1))], 1)
        pts_ego = (l2e @ ph.T).T[:, :3]
        g, gc, ge = ground_surface(pts_ego)

        probs = []
        for cam in range(len(cal["names"])):
            im = Image.open(io.BytesIO(bytes(d["jpegs"][cam]))).convert("RGB")
            probs.append(segment(model, np.asarray(im)).cpu())

        split = "night" if "night" in rec["session"] else "day"
        # the ground surface does not depend on which warp mode we are testing
        zdem = sample_grid(g, gc, ge, xs, ys)
        zdem = np.where(np.isfinite(zdem), zdem, np.nanmedian(zdem)).astype(np.float64)
        zspreads.append(float(np.percentile(zdem, 95) - np.percentile(zdem, 5)))
        for mode in modes:
            if mode == "dem":
                z = zdem
            elif mode == "flat":
                # same plane, but with the true height: isolates terrain *shape*
                z = np.full(len(xs), float(np.median(zdem)))
            else:                       # "zero": the naive z=0 control
                z = np.zeros(len(xs), np.float64)

            zs_list = [z] * len(probs)
            p2, vis = warp_bev(probs, cal["camera2ego"], cal["K"], xs, ys, zs_list,
                               max_range=45.0)
            pbev, vis = to_bev_channels(p2, vis)
            key = (mode, split)
            if key not in tp:
                continue
            flat = vis
            gt_f = gt.reshape(gt.shape[0], -1)[:, flat]
            val_f = valid.ravel()[flat]
            pb = pbev[:, flat]
            for c in range(len(BEV_CLASSES)):
                g_c = gt_f[c] & val_f
                if g_c.sum() == 0:
                    continue
                present[key][c] += 1
                for ti, t in enumerate(THRS):
                    pred = (pb[c] > t) & val_f
                    tp[key][c, ti] += int((pred & g_c).sum())
                    fp[key][c, ti] += int((pred & ~g_c).sum())
                    fn[key][c][ti] += int((~pred & g_c).sum())
        if (fi + 1) % 10 == 0:
            print(f"  {fi+1}/{len(recs)}", flush=True)

    print(f"\n{'mode':6s} {'split':6s} " + " ".join(f"{c[:9]:>10s}" for c in BEV_CLASSES) + f" {'mIoU':>7s}")
    summary = {}
    for key in sorted(tp):
        tot = tp[key] + fp[key] + fn[key]
        iou = np.where(tot > 0, tp[key] / np.maximum(tot, 1), np.nan)
        best = np.nanmax(iou, axis=1)
        vals = [best[i] for i in range(len(BEV_CLASSES)) if present[key][i] > 0]
        mi = float(np.mean(vals)) if vals else float("nan")
        summary[f"{key[0]}_{key[1]}"] = dict(miou=mi, **{
            BEV_CLASSES[i]: (None if np.isnan(best[i]) else float(best[i]))
            for i in range(len(BEV_CLASSES))})
        cells = " ".join("--" if np.isnan(best[i]) else f"{best[i]:10.3f}"
                         for i in range(len(BEV_CLASSES)))
        print(f"{key[0]:6s} {key[1]:6s} {cells} {mi:7.3f}")
    if zspreads:
        print(f"\nground-height spread (p95-p5) over the BEV window: "
              f"median {np.median(zspreads):.2f} m, max {max(zspreads):.2f} m")
    summary["_z_spread_m"] = dict(median=float(np.median(zspreads)) if zspreads else 0.0,
                                  max=float(max(zspreads)) if zspreads else 0.0)
    (out / "stage4.json").write_text(json.dumps(summary, indent=2))
    print(f"\n-> {out/'stage4.json'}")


if __name__ == "__main__":
    main()
