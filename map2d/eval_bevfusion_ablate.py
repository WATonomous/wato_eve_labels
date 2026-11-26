#!/usr/bin/env python3
"""Evaluation suite for the dual-head map head on Eve data, with input ablations.

NOTE: this one needs the BEVFusion environment (mmdet3d, envs/bf), not the
seg2d one. It is kept here so the numbers in docs/results-2026-11-26.md are
reproducible from a single repository.

Runs the SAME checkpoint over the SAME frames under a set of input
perturbations, so the deltas are attributable. Focus is the large classes
(drivable_area, walkway) plus ped_crossing/divider, reported per class with
mIoU@max over valid cells only.

Ablations answer "how good can we get, and what is the lever":

  base            as recorded
  yaw_p2/m5/p10   rotate the cloud about z (sensor yaw error sensitivity)
  z_p20/m50       shift z (height/ground-plane calibration)
  range_30/50     clip points beyond N m
  dens_25/50      random subsample (density sensitivity)
  no_intensity    zero the intensity channel
  no_lidar        drop the cloud entirely (camera-only floor)
  flip_y          mirror y (convention sanity check; must be much worse)

  eval_ablate.py --cfg CONFIG --ckpt CKPT --sessions a,b --variants base,yaw_m5
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import torch
import yaml

from mmcv.parallel import DataContainer as DC
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model

CLASSES = ["drivable_area", "ped_crossing", "walkway", "stop_line",
           "carpark_area", "divider"]
SUPERVISED = ["drivable_area", "ped_crossing", "divider", "stop_line"]
THRS = np.arange(0.02, 0.71, 0.02)


def rot_y(p, deg):
    """Pitch: rotate about the lateral axis (a mount-angle error)."""
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    q = p.copy()
    q[:, 0] = p[:, 0] * c + p[:, 2] * s
    q[:, 2] = -p[:, 0] * s + p[:, 2] * c
    return q


def rot_z(p, deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    q = p.copy()
    q[:, 0] = p[:, 0] * c - p[:, 1] * s
    q[:, 1] = p[:, 0] * s + p[:, 1] * c
    return q


VARIANTS = {
    "base": lambda p: p,
    "yaw_p2": lambda p: rot_z(p, 2.0),
    "yaw_m2": lambda p: rot_z(p, -2.0),
    "yaw_p5": lambda p: rot_z(p, 5.0),
    "yaw_m5": lambda p: rot_z(p, -5.0),
    "yaw_p10": lambda p: rot_z(p, 10.0),
    "yaw_m10": lambda p: rot_z(p, -10.0),
    "yaw_p20": lambda p: rot_z(p, 20.0),
    "yaw_m20": lambda p: rot_z(p, -20.0),
    "yaw_p30": lambda p: rot_z(p, 30.0),
    "yaw_m30": lambda p: rot_z(p, -30.0),
    "yaw_p45": lambda p: rot_z(p, 45.0),
    "yaw_m45": lambda p: rot_z(p, -45.0),
    "yaw_p50": lambda p: rot_z(p, 50.0),
    "yaw_m50": lambda p: rot_z(p, -50.0),
    "yaw_180": lambda p: rot_z(p, 180.0),
    "range_20": lambda p: p[np.hypot(p[:, 0], p[:, 1]) < 20],
    "range_10": lambda p: p[np.hypot(p[:, 0], p[:, 1]) < 10],
    "range_40": lambda p: p[np.hypot(p[:, 0], p[:, 1]) < 40],
    "yaw_p1": lambda p: rot_z(p, 1.0),
    "yaw_m1": lambda p: rot_z(p, -1.0),
    "yaw_p2b": lambda p: rot_z(p, 2.0),
    "yaw_m2b": lambda p: rot_z(p, -2.0),
    "x_p50": lambda p: np.concatenate([p[:, 0:1] + 0.5, p[:, 1:]], 1),
    "x_m50": lambda p: np.concatenate([p[:, 0:1] - 0.5, p[:, 1:]], 1),
    "x_p100": lambda p: np.concatenate([p[:, 0:1] + 1.0, p[:, 1:]], 1),
    "x_m100": lambda p: np.concatenate([p[:, 0:1] - 1.0, p[:, 1:]], 1),
    "y_p50": lambda p: np.concatenate([p[:, 0:1], p[:, 1:2] + 0.5, p[:, 2:]], 1),
    "y_m50": lambda p: np.concatenate([p[:, 0:1], p[:, 1:2] - 0.5, p[:, 2:]], 1),
    "scale_102": lambda p: np.concatenate([p[:, :3] * 1.02, p[:, 3:]], 1),
    "scale_98": lambda p: np.concatenate([p[:, :3] * 0.98, p[:, 3:]], 1),
    "pitch_p1": lambda p: rot_y(p, 1.0),
    "pitch_m1": lambda p: rot_y(p, -1.0),
    "pitch_p3": lambda p: rot_y(p, 3.0),
    "pitch_m3": lambda p: rot_y(p, -3.0),
    "lidar_shuffle": None,
    "z_p20": lambda p: np.concatenate([p[:, :2], p[:, 2:3] + 0.2, p[:, 3:]], 1),
    "z_m20": lambda p: np.concatenate([p[:, :2], p[:, 2:3] - 0.2, p[:, 3:]], 1),
    "z_p50": lambda p: np.concatenate([p[:, :2], p[:, 2:3] + 0.5, p[:, 3:]], 1),
    "z_m50": lambda p: np.concatenate([p[:, :2], p[:, 2:3] - 0.5, p[:, 3:]], 1),
    "range_30": lambda p: p[np.hypot(p[:, 0], p[:, 1]) < 30],
    "range_50": lambda p: p[np.hypot(p[:, 0], p[:, 1]) < 50],
    "dens_25": None,
    "dens_50": None,
    "no_intensity": lambda p: np.concatenate([p[:, :3], np.zeros_like(p[:, 3:4]), p[:, 4:]], 1),
    "no_lidar": lambda p: p[:1],
    "flip_y": lambda p: np.concatenate([p[:, 0:1], -p[:, 1:2], p[:, 2:]], 1),
}


def apply_variant(name, p):
    if name == "lidar_shuffle":
        q = p.copy()
        rs = np.random.RandomState(0)
        rs.shuffle(q[:, 0])
        rs.shuffle(q[:, 1])
        rs.shuffle(q[:, 2])
        return q
    if name == "dens_25":
        keep = np.random.rand(len(p)) < 0.25
        return p[keep]
    if name == "dens_50":
        keep = np.random.rand(len(p)) < 0.5
        return p[keep]
    return VARIANTS[name](p)


def run_variant(model, ds, name, gpu, max_frames, seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    tp = {c: np.zeros(len(THRS)) for c in SUPERVISED}
    fp = {c: np.zeros(len(THRS)) for c in SUPERVISED}
    fn = {c: np.zeros(len(THRS)) for c in SUPERVISED}
    present = {c: 0 for c in SUPERVISED}
    root = pathlib.Path(ds.root)
    n = len(ds) if not max_frames else min(len(ds), max_frames)
    for i in range(n):
        s = ds[i]
        pts = s["points"].data.numpy()
        pts = apply_variant(name, pts)
        if len(pts) < 1:
            pts = np.zeros((1, 5), np.float32)
        batch = {}
        for k, v in s.items():
            if torch.is_tensor(v):
                batch[k] = v.unsqueeze(0).to(gpu)
            elif isinstance(v, DC):
                if k == "points":
                    batch[k] = [torch.from_numpy(pts.astype(np.float32)).to(gpu)]
                else:
                    batch[k] = [v.data]
            else:
                batch[k] = [v]
        with torch.no_grad():
            out = model(return_loss=False, **batch)[0]
        prob = out["masks_bev"].float().cpu().numpy()
        rec = ds.records[i]
        mask = np.load(root / "masks" / f"{rec['id']}.npz", allow_pickle=True)
        valid = mask["valid"].astype(bool) if "valid" in mask else np.ones_like(prob[0], bool)
        for c in SUPERVISED:
            gt = (mask[c] > 0).astype(bool) & valid
            if gt.sum() == 0:
                continue
            present[c] += 1
            p = prob[CLASSES.index(c)]
            for ti, t in enumerate(THRS):
                pred = (p > t) & valid
                tp[c][ti] += int((pred & gt).sum())
                fp[c][ti] += int((pred & ~gt).sum())
                fn[c][ti] += int((~pred & gt).sum())
    res = {}
    for c in SUPERVISED:
        if present[c] == 0:
            continue
        iou = tp[c] / np.maximum(tp[c] + fp[c] + fn[c], 1)
        bi = int(np.argmax(iou))
        res[c] = dict(iou_max=float(iou[bi]), best_thr=float(THRS[bi]),
                      iou_05=float(iou[int(np.argmin(np.abs(THRS - 0.5)))]),
                      frames=int(present[c]))
    vals = [v["iou_max"] for v in res.values()]
    res["mIoU"] = float(np.mean(vals)) if vals else 0.0
    return res


def main():
    global SUPERVISED
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sessions", required=True)
    ap.add_argument("--variants", default=",".join(VARIANTS.keys()))
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--gpu", default="cuda:0")
    ap.add_argument("--out", default="")
    ap.add_argument("--classes", default=",".join(SUPERVISED),
                    help="classes to average over (default: all four supervised)")
    args = ap.parse_args()

    sys.path.insert(0, str(pathlib.Path.cwd()))
    from mmdet3d.datasets.eve_map_dataset import EveMapDataset  # noqa: E402

    cfg = yaml.safe_load(open(args.cfg))
    model = build_model(cfg["model"], train_cfg=None, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, args.ckpt, map_location="cpu", strict=False)
    model.to(args.gpu).eval()

    root = cfg["data"]["val"]["root"]
    ds = EveMapDataset(root=root, sessions=args.sessions.split(","),
                       is_train=False, test_mode=True)
    print(f"{len(ds)} frames, variants: {args.variants}\n", flush=True)

    SUPERVISED = [c for c in args.classes.split(",") if c]
    print(f"classes: {SUPERVISED}\n", flush=True)

    results = {}
    order = args.variants.split(",")
    for name in order:
        if name not in VARIANTS:
            print(f"  skip unknown variant {name}")
            continue
        r = run_variant(model, ds, name, args.gpu, args.max_frames)
        results[name] = r
        line = " ".join(f"{c[:4]}={r[c]['iou_max']:.3f}" for c in SUPERVISED if c in r)
        print(f"{name:14s} mIoU={r['mIoU']:.4f}  {line}", flush=True)

    print(f"\n{'variant':14s} " + " ".join(f"{c[:10]:>11s}" for c in SUPERVISED) + f" {'mIoU':>8s}")
    base = results.get("base", {}).get("mIoU")
    for name in order:
        if name not in results:
            continue
        r = results[name]
        cells = " ".join(f"{r[c]['iou_max']:11.3f}" if c in r else f"{'-':>11s}" for c in SUPERVISED)
        d = "" if base is None or name == "base" else f"  ({r['mIoU'] - base:+.3f})"
        print(f"{name:14s} {cells} {r['mIoU']:8.4f}{d}")

    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
