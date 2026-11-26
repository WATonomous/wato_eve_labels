#!/usr/bin/env python3
"""Render an ordered sequence of demo frames: camera | predicted BEV | GT BEV.

Feeds straight into ffmpeg:

  viz_seq.py --frames DIR --masks DIR --manifest JSON --ckpt CKPT \
             --session clip_night --out DIR
  ffmpeg -y -framerate 10 -i DIR/panel_%04d.png -c:v libx264 -pix_fmt yuv420p out.mp4
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
from wato2d.models.segformer import build_model                             # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
COL = {"other": (0, 0, 0), "drivable": (0, 205, 95), "walkway": (45, 135, 255),
       "crossing": (255, 175, 0)}
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


def warp(probs, c2e, K, xs, ys, z):
    n = len(xs)
    acc = np.zeros((4, n), np.float32); cnt = np.zeros(n, np.float32)
    P = np.stack([xs, ys, z, np.ones(n)], 1)
    for prob, T, Kc in zip(probs, c2e, K):
        H, W = prob.shape[-2:]
        pc = (np.linalg.inv(T) @ P.T).T
        ok = pc[:, 2] > 1.5
        if not ok.any():
            continue
        u = (Kc[0, 0] * pc[:, 0] + Kc[0, 1] * pc[:, 1] + Kc[0, 2] * pc[:, 2]) / pc[:, 2]
        v = (Kc[1, 0] * pc[:, 0] + Kc[1, 1] * pc[:, 1] + Kc[1, 2] * pc[:, 2]) / pc[:, 2]
        inb = ok & (u > 0) & (u < W - 1) & (v > 0) & (v < H - 1)
        if not inb.any():
            continue
        g = torch.from_numpy(np.stack([(u[inb] / W) * 2 - 1, (v[inb] / H) * 2 - 1], 1)
                             [None, None].astype(np.float32)).cuda()
        s = F.grid_sample(prob.unsqueeze(0).to(g.device), g, mode="bilinear",
                          align_corners=False, padding_mode="border")[0, :, 0, :].cpu().numpy()
        acc[:, inb] += s; cnt[inb] += 1
    out = np.zeros_like(acc); m = cnt > 0
    out[:, m] = acc[:, m] / cnt[m]
    return out, m


def colourise(cls, vis=None):
    rgb = np.zeros((SIZE * SIZE, 3), np.uint8)
    for k, nm in enumerate(CLASSES):
        sel = cls.ravel() == k
        if vis is not None:
            sel &= vis
        rgb[sel] = COL[nm]
    return rgb.reshape(SIZE, SIZE, 3)[::-1]


def bev_panel(rgb, title, w=430):
    im = Image.fromarray(rgb).resize((w, w), Image.NEAREST)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 138, 15], fill=(0, 0, 0))
    d.text((4, 3), title, fill=(235, 235, 235))
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

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
    xs = (cols_ * RES - EXTENT).ravel(); ys = (EXTENT - rows_ * RES).ravel()

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
        g, gc, ge = ground_surface(pe)
        z = sample_grid((g, gc, ge), xs, ys)
        z = np.where(np.isfinite(z), z, np.nanmedian(z))

        probs, imgs = [], []
        for c in range(len(cal["names"])):
            im = Image.open(io.BytesIO(bytes(d["jpegs"][c]))).convert("RGB")
            arr = np.asarray(im)
            imgs.append(arr)
            probs.append(segment(model, arr).cpu())

        p2, vis = warp(probs, cal["camera2ego"], cal["K"], xs, ys, z)
        pred_rgb = colourise(p2.argmax(0).reshape(SIZE, SIZE), vis)

        gt = np.zeros((SIZE, SIZE), np.int8)
        for ci, ch in enumerate(BEV_CLASSES):
            if ch in BEV_CH and ch in mask:
                gt[mask[ch] > 0] = CLASSES.index(BEV_CH[ch])
        gt_rgb = colourise(gt)

        ov = imgs[args.cam].copy()
        front = probs[args.cam].argmax(0).numpy()
        for k, nm in enumerate(CLASSES):
            if nm == "other":
                continue
            m = front == k
            ov[m] = (0.42 * ov[m] + 0.58 * np.array(COL[nm])).astype(np.uint8)
        ov = np.asarray(Image.fromarray(ov).resize((760, 428)))

        panel = Image.new("RGB", (760 + 450 + 450 + 24, 470), (16, 16, 20))
        panel.paste(Image.fromarray(ov), (8, 26))
        panel.paste(bev_panel(pred_rgb, "predicted BEV free space"), (776, 26))
        panel.paste(bev_panel(gt_rgb, "lanelet GT"), (1234, 26))
        dr = ImageDraw.Draw(panel)
        dr.text((10, 6), f"{args.session}  frame {i:04d}", fill=(220, 220, 220))
        dr.text((778, 8), "green drivable   blue walkway   orange crossing",
                fill=(190, 190, 190))
        panel.save(out / f"panel_{i:04d}.png")
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(recs)}", flush=True)
    print("done ->", out)


if __name__ == "__main__":
    main()
