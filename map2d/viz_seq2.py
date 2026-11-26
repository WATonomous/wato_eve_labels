#!/usr/bin/env python3
"""Sequence renderer with cross-camera weighting, occlusion and occupancy.

Left: camera with the 2D segmentation. Right: the BEV map in five states.

  viz_seq2.py --frames DIR --masks DIR --manifest JSON --ckpt CKPT \
              --session clip_night --out DIR
"""
import argparse
import io
import json
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from wato2d.bev_geom import ground_surface, sample_grid, RES, EXTENT, SIZE   # noqa: E402
from wato2d.classes import CLASSES                                          # noqa: E402
from wato2d.fuse import (depth_buffer, fuse, occupancy, to_states, smooth_states,  # noqa: E402
                         STATE_DRIVABLE, STATE_WALKWAY, STATE_CROSSING,
                         STATE_OCCUPIED, STATE_UNKNOWN)
from wato2d.models.segformer import build_model                             # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
CLS_COL = {"other": (0, 0, 0), "drivable": (0, 190, 90), "walkway": (40, 130, 255),
           "crossing": (255, 170, 0)}
STATE_COL = np.array([
    (0, 190, 90),        # drivable
    (40, 130, 255),      # walkway
    (255, 170, 0),       # crossing
    (225, 60, 60),       # occupied
    (58, 58, 66),        # unknown
], np.uint8)
BEV_CH = {"drivable_area": "drivable", "carpark_area": "drivable",
          "walkway": "walkway", "ped_crossing": "crossing"}
BEV_CLASSES = ["drivable_area", "ped_crossing", "walkway", "stop_line",
               "carpark_area", "divider"]


@torch.no_grad()
def segment(model, arr, size=(256, 320)):
    x = (arr.astype(np.float32) / 255.0 - MEAN) / STD
    t = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).cuda()
    t = F.interpolate(t, size=size, mode="bilinear", align_corners=False)
    with torch.autocast("cuda", dtype=torch.float16):
        logits = model(t)
    p = logits.float().softmax(1)[0]
    return F.interpolate(p.unsqueeze(0), size=(arr.shape[0], arr.shape[1]),
                         mode="bilinear", align_corners=False)[0]


def render_states(st, w=430):
    rgb = STATE_COL[st[::-1]]
    return Image.fromarray(rgb).resize((w, w), Image.NEAREST)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--thr", default="0.35,0.35,0.45")
    ap.add_argument("--occ-max", type=float, default=0.30)
    ap.add_argument("--min-cams", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--smooth", action="store_true", default=True)
    args = ap.parse_args()
    thr = tuple(float(x) for x in args.thr.split(","))

    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    model = build_model().cuda().eval()
    sd = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(sd["model"] if "model" in sd else sd)

    recs = [r for r in json.loads(pathlib.Path(args.manifest).read_text())
            if pathlib.Path(r["frame_path"]).parent.name == args.session]
    recs.sort(key=lambda r: r["frame_path"])
    if args.limit:
        recs = recs[:args.limit]
    print(f"{len(recs)} frames for {args.session}", flush=True)

    rows_, cols_ = np.mgrid[0:SIZE, 0:SIZE]
    xs = (cols_ * RES - EXTENT).ravel()
    ys = (EXTENT - rows_ * RES).ravel()
    stats = []

    for i, rec in enumerate(recs):
        fp = pathlib.Path(rec["frame_path"])
        mp = None
        for md in args.masks.split(","):
            c = pathlib.Path(md) / f"{rec['id']}.npz"
            if c.exists():
                mp = c; break
        if not fp.exists() or mp is None:
            continue
        d = np.load(fp, allow_pickle=True)
        cal = np.load(fp.parent / "calib.npz", allow_pickle=True)
        mask = np.load(mp, allow_pickle=True)

        pts = d["points"].astype(np.float64)
        l2e = cal["lidar2ego"].astype(np.float64)
        pe = (l2e @ np.concatenate([pts[:, :3], np.ones((len(pts), 1))], 1).T).T[:, :3]
        ground = ground_surface(pe)
        z = sample_grid(ground, xs, ys)
        z = np.where(np.isfinite(z), z, np.nanmedian(z))

        probs, imgs = [], []
        for c in range(len(cal["names"])):
            im = Image.open(io.BytesIO(bytes(d["jpegs"][c]))).convert("RGB")
            arr = np.asarray(im)
            imgs.append(arr)
            probs.append(segment(model, arr).cpu())
        W, H = imgs[0].shape[1], imgs[0].shape[0]

        dbufs = []
        dscale = 4
        for c in range(len(cal["names"])):
            db, dscale = depth_buffer(pe, cal["camera2ego"][c], cal["K"][c], W, H, scale=4)
            dbufs.append(db)

        cells = np.stack([xs, ys, z], 1)
        p4, wsum, vis, seen = fuse(probs, cal["camera2ego"], cal["K"], W, H,
                                   cells, dbufs, dscale,
                                   min_weight=0.02, entropy_scale=1.0)
        occ = occupancy(pe, ground, occ_max=args.occ_max, min_points=3)
        st = to_states(p4, wsum, occ, thr=thr, occ_max=args.occ_max,
                       min_cams=args.min_cams, seen=seen)
        if args.smooth:
            st = smooth_states(st)
            for _ in range(1):
                st = smooth_states(st)

        frac = {n: float((st == k).mean()) for k, n in enumerate(
            ["drivable", "walkway", "crossing", "occupied", "unknown"])}
        stats.append(frac)

        # camera panel
        ov = imgs[args.cam].copy()
        front = probs[args.cam].argmax(0).numpy()
        for k, nm in enumerate(CLASSES):
            if nm == "other":
                continue
            m = front == k
            ov[m] = (0.42 * ov[m] + 0.58 * np.array(CLS_COL[nm])).astype(np.uint8)

        gt = np.full((SIZE, SIZE), STATE_UNKNOWN, np.int8)
        for ci, ch in enumerate(BEV_CLASSES):
            if ch in BEV_CH and ch in mask:
                gt[mask[ch] > 0] = [STATE_DRIVABLE, STATE_WALKWAY,
                                    STATE_CROSSING][
                    ["drivable", "walkway", "crossing"].index(BEV_CH[ch])]
        gt_rgb = STATE_COL[gt[::-1]]
        gt_img = Image.fromarray(gt_rgb).resize((430, 430), Image.NEAREST)

        panel = Image.new("RGB", (760 + 450 + 450 + 24, 500), (16, 16, 20))
        panel.paste(Image.fromarray(np.asarray(Image.fromarray(ov).resize((760, 428)))), (8, 40))
        panel.paste(render_states(st), (776, 40))
        panel.paste(gt_img, (1234, 40))
        dr = ImageDraw.Draw(panel)
        dr.text((10, 8), f"{args.session}  frame {i:04d}", fill=(225, 225, 225))
        dr.text((780, 8), "fused BEV   green drivable  blue walkway  orange crossing",
                fill=(200, 200, 200))
        dr.text((780, 22), "            red occupied (LiDAR)   grey unknown", fill=(200, 200, 200))
        dr.text((1238, 22), "lanelet GT", fill=(200, 200, 200))
        dr.text((10, 476), "2D segmentation on CAM_FRONT", fill=(190, 190, 190))
        panel.save(out / f"panel_{i:04d}.png")
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(recs)}", flush=True)

    if stats:
        mean = {k: float(np.mean([s[k] for s in stats])) for k in stats[0]}
        print("mean state fractions:", {k: round(v, 3) for k, v in mean.items()})
    print("done ->", out)


if __name__ == "__main__":
    main()
