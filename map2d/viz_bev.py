#!/usr/bin/env python3
"""Visual summary of the 2D free-space pipeline on our own frames.

Produces one figure per frame with four panels:

  1. CAM_FRONT with the predicted drivable / walkway / crossing overlay
  2. the predicted BEV free-space map (warped from all 8 cameras)
  3. the lanelet ground truth for the same frame
  4. predicted vs GT agreement (green = both, red = predicted only,
     blue = GT only)

and a contact sheet across frames. Deliberately honest: the GT panel is shown
next to the prediction so nobody has to take the numbers on faith.

  viz_bev.py --frames DIR --masks DIR --manifest JSON --ckpt CKPT --out DIR
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

# colours for the 4-class ontology
COL = {"other": (0, 0, 0), "drivable": (0, 200, 90), "walkway": (40, 130, 255),
       "crossing": (255, 170, 0)}
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


def warp(probs, c2e, K, xs, ys, z, rmax=45.0):
    n = len(xs)
    acc = np.zeros((4, n), np.float32)
    cnt = np.zeros(n, np.float32)
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
        acc[:, inb] += s
        cnt[inb] += 1
    out = np.zeros_like(acc)
    m = cnt > 0
    out[:, m] = acc[:, m] / cnt[m]
    return out, m


def panel_bev(rgb_pred, title):
    im = Image.fromarray(rgb_pred).resize((420, 420), Image.NEAREST)
    dr = ImageDraw.Draw(im)
    dr.text((6, 4), title, fill=(255, 255, 255))
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()

    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    model = build_model().cuda().eval()
    sd = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(sd["model"] if "model" in sd else sd)

    recs = json.loads(pathlib.Path(args.manifest).read_text())
    # pick a spread: alternate day/night, spread through the list
    night = [r for r in recs if "night" in r["session"]]
    day = [r for r in recs if "night" not in r["session"]]
    pick = []
    for i in range(args.n):
        src = night if i % 2 else day
        if src:
            pick.append(src[(i // 2) * max(1, len(src) // max(1, args.n // 2)) % len(src)])
    print(f"{len(recs)} records, {len(day)} day / {len(night)} night, picked {len(pick)}", flush=True)
    rows = []
    for rec in pick:
        fp = pathlib.Path(rec["frame_path"])
        mp = None
        for md in args.masks.split(","):
            cand = pathlib.Path(md) / f"{rec['id']}.npz"
            if cand.exists():
                mp = cand
                break
        if not fp.exists() or mp is None:
            print("  skip", rec["id"], flush=True)
            continue
        d = np.load(fp, allow_pickle=True)
        cal = np.load(fp.parent / "calib.npz", allow_pickle=True)
        mask = np.load(mp, allow_pickle=True)
        pts = d["points"].astype(np.float64)
        l2e = cal["lidar2ego"].astype(np.float64)
        pe = (l2e @ np.concatenate([pts[:, :3], np.ones((len(pts), 1))], 1).T).T[:, :3]
        g, gc, ge = ground_surface(pe)

        probs = []
        imgs = []
        for c in range(len(cal["names"])):
            im = Image.open(io.BytesIO(bytes(d["jpegs"][c]))).convert("RGB")
            arr = np.asarray(im)
            imgs.append(arr)
            probs.append(segment(model, arr).cpu())

        rows_, cols_ = np.mgrid[0:SIZE, 0:SIZE]
        xs = (cols_ * RES - EXTENT).ravel()
        ys = (EXTENT - rows_ * RES).ravel()
        z = sample_grid((g, gc, ge), xs, ys)
        z = np.where(np.isfinite(z), z, np.nanmedian(z))
        p2, vis = warp(probs, cal["camera2ego"], cal["K"], xs, ys, z)

        # BEV prediction (argmax over the 4 classes, masked by visibility)
        cls = p2.argmax(0)
        rgb = np.zeros((SIZE * SIZE, 3), np.uint8)
        for k, nm in enumerate(CLASSES):
            rgb[(cls == k) & vis] = COL[nm]
        rgb = rgb.reshape(SIZE, SIZE, 3)[::-1]

        gt = np.zeros((SIZE, SIZE), np.int8)
        for ci, ch in enumerate(BEV_CLASSES):
            if ch in BEV_CH and ch in mask:
                gt[mask[ch] > 0] = CLASSES.index(BEV_CH[ch])
        gt_rgb = np.zeros((SIZE * SIZE, 3), np.uint8)
        for k, nm in enumerate(CLASSES):
            gt_rgb[gt.ravel() == k] = COL[nm]
        gt_rgb = gt_rgb.reshape(SIZE, SIZE, 3)[::-1]

        # agreement for drivable only (the class that matters)
        pd = (rgb[::-1][..., 1] > 150)
        gd = (gt_rgb[::-1][..., 1] > 150)
        agr = np.zeros((SIZE, SIZE, 3), np.uint8)
        agr[pd & gd] = (0, 200, 90)
        agr[pd & ~gd] = (255, 60, 60)
        agr[~pd & gd] = (60, 120, 255)

        # camera overlay on the front camera
        front = probs[0].argmax(0).numpy()
        ov = imgs[0].copy()
        for k, nm in enumerate(CLASSES):
            if nm == "other":
                continue
            m = front == k
            ov[m] = (0.45 * ov[m] + 0.55 * np.array(COL[nm])).astype(np.uint8)

        row = Image.new("RGB", (1400, 460), (18, 18, 22))
        cam = Image.fromarray(ov).resize((560, 448))
        row.paste(cam, (10, 6))
        ImageDraw.Draw(row).text((14, 424), "CAM_FRONT: green drivable / blue walkway / orange crossing",
                                 fill=(200, 200, 200))
        row.paste(panel_bev(rgb, "predicted BEV free space"), (600, 20))
        row.paste(panel_bev(gt_rgb, "lanelet GT"), (1030 // 1 + 0, 20)) if False else None
        # three BEV panels side by side, smaller
        row = Image.new("RGB", (1400, 470), (18, 18, 22))
        row.paste(Image.fromarray(ov).resize((520, 416)), (8, 8))
        ImageDraw.Draw(row).text((12, 430), f"CAM_FRONT  {rec['id'][:40]}", fill=(210, 210, 210))
        row.paste(panel_bev(rgb, "predicted BEV"), (545, 20))
        row.paste(panel_bev(gt_rgb, "lanelet GT"), (975, 20))
        rows.append(row)
        nm = f"{rec['id']}.png".replace("/", "_")
        row.save(out / nm)
        print("wrote", out / nm, flush=True)

    if rows:
        sheet = Image.new("RGB", (1400, 470 * len(rows)), (0, 0, 0))
        for i, r in enumerate(rows):
            sheet.paste(r, (0, 470 * i))
        sheet.save(out / "contact_sheet.png")
        print("contact sheet:", out / "contact_sheet.png")


if __name__ == "__main__":
    main()
