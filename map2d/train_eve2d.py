#!/usr/bin/env python3
"""Stage 3: fine-tune the 2D branch on the car's own frames.

Consumes what `stage2_labels.py` produced: one uint8 label map per
frame x camera, in image space, derived from the BEV ground truth through the
LiDAR ground surface.

Splitting is **by session** (a session is a separate drive), never by frame —
two frames from one drive share geometry and weather, so a frame split leaks.

  train_eve2d.py --frames DIR --labels DIR --manifest JSON \
                 --init runs/seg2d_ade20k_acdc/best.pth --out runs/eve2d_v1
"""
import argparse
import io
import json
import math
import pathlib
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from wato2d.classes import CLASSES, IGNORE          # noqa: E402
from wato2d.metrics import Confusion               # noqa: E402
from wato2d.models.segformer import build_model, count_params   # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def colour_jitter(im, rs):
    """Brightness / gamma / noise. Night is not a hue shift, so no inversion."""
    a = im.astype(np.float32) / 255.0
    a *= rs.uniform(0.6, 1.45)                     # exposure
    g = math.exp(rs.uniform(-0.45, 0.45))          # gamma (shadow lift)
    a = np.clip(a, 1e-4, 1.0) ** g
    if rs.rand() < 0.5:                            # sensor noise, worse at night
        a += rs.normal(0, rs.uniform(0.005, 0.04), a.shape).astype(np.float32)
    return np.clip(a, 0, 1)


class EveFrames(Dataset):
    def __init__(self, frames, labels, manifest, sessions, train=True, size=(256, 320)):
        self.size = size
        recs = json.loads(pathlib.Path(manifest).read_text())
        want = set(sessions)
        self.items = []
        for r in recs:
            fp = pathlib.Path(r["frame_path"])
            sess = fp.parent.name
            if sess not in want:
                continue
            lab_dir = pathlib.Path(labels) / sess
            if not lab_dir.exists():
                continue
            cams = sorted(int(p.stem.split("_cam")[-1]) for p in lab_dir.glob(f"{fp.stem}_cam*.npy"))
            if cams:
                self.items.append((fp, lab_dir, fp.stem, cams))
        if not self.items:
            raise RuntimeError(f"no labelled frames for sessions {sessions}")
        self.train = train

    def __len__(self):
        return len(self.items) if not self.train else len(self.items) * 8

    def __getitem__(self, i):
        fp, lab_dir, stem, cams = self.items[i % len(self.items)]
        cam = random.choice(cams) if self.train else cams[i % len(cams)]
        d = np.load(fp, allow_pickle=True)
        im = Image.open(io.BytesIO(bytes(d["jpegs"][cam]))).convert("RGB")
        lab = np.load(lab_dir / f"{stem}_cam{cam}.npy")
        im = im.resize((self.size[1], self.size[0]), Image.BILINEAR)
        lab = np.array(Image.fromarray(lab).resize((self.size[1], self.size[0]), Image.NEAREST))
        arr = np.asarray(im, np.uint8)
        rs = np.random.RandomState(i * 7919 + cam)
        if self.train:
            if rs.rand() < 0.5:
                arr = arr[:, ::-1]
                lab = lab[:, ::-1]
            arr = (colour_jitter(arr, rs) * 255).astype(np.uint8)
        x = (np.asarray(arr, np.float32) / 255.0 - MEAN) / STD
        return torch.from_numpy(x.transpose(2, 0, 1)), torch.from_numpy(lab.astype(np.int64))


def class_weights(ds, n=len(CLASSES)):
    cnt = np.zeros(n, np.float64)
    for i in range(0, len(ds), max(1, len(ds) // 120)):
        y = ds[i][1].numpy()
        for c in range(n):
            cnt[c] += int((y == c).sum())
    present = cnt > 0
    w = np.zeros(n)
    if present.any():
        inv = 1.0 / np.sqrt(cnt[present] / cnt[present].sum())
        w[present] = np.clip(inv / inv.mean(), 0.1, 12.0)
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--val-sessions", default="")
    ap.add_argument("--init", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--all-sessions", action="store_true",
                    help="train on every session (no holdout; for the shipped model)")
    args = ap.parse_args()

    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    recs = json.loads(pathlib.Path(args.manifest).read_text())
    sessions = sorted({pathlib.Path(r["frame_path"]).parent.name for r in recs})
    val = [s for s in args.val_sessions.split(",") if s] or sessions[-1:]
    tr_s = sessions if args.all_sessions else [s for s in sessions if s not in val]
    print(f"sessions: train {tr_s}  val {val}", flush=True)

    tr = EveFrames(args.frames, args.labels, args.manifest, tr_s, True)
    va = EveFrames(args.frames, args.labels, args.manifest, val, False)
    print(f"frames: {len(tr)} train views / {len(va)} val views", flush=True)

    w = class_weights(tr)
    print("class weights:", {n: round(float(x), 2) for n, x in zip(CLASSES, w)}, flush=True)
    class_w = torch.tensor(w, dtype=torch.float32).cuda()

    dev = "cuda"
    model = build_model().to(dev)
    if args.init:
        sd = torch.load(args.init, map_location="cpu")
        model.load_state_dict(sd["model"] if "model" in sd else sd)
        print(f"initialised from {args.init}", flush=True)
    print(f"{count_params(model):.1f} M params", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.02)
    steps = max(1, len(tr) // args.batch) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.2)
    scaler = torch.amp.GradScaler("cuda")

    log, best = [], 0.0
    for ep in range(args.epochs):
        model.train()
        order = torch.randperm(len(tr))
        run, n = 0.0, 0
        for b in range(0, len(order), args.batch):
            xs, ys = [], []
            for i in order[b:b + args.batch].tolist():
                x, y = tr[i]; xs.append(x); ys.append(y)
            x = torch.stack(xs).to(dev, non_blocking=True)
            y = torch.stack(ys).to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = F.cross_entropy(model(x), y, weight=class_w, ignore_index=IGNORE)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            run += float(loss); n += 1
        model.eval()
        conf = Confusion()
        with torch.no_grad():
            for i in range(len(va)):
                x, y = va[i]
                with torch.autocast("cuda", dtype=torch.float16):
                    p = model(x.unsqueeze(0).to(dev))
                conf.update(p.float().argmax(1)[0].cpu().numpy(), y.numpy())
        m = conf.iou()
        print(f"epoch {ep}: loss {run/max(n,1):.3f}  "
              + " ".join(f"{k}={'--' if v != v else f'{v:.3f}'}" for k, v in m.items()),
              flush=True)
        log.append(dict(epoch=ep, loss=run / max(n, 1), **{k: (None if v != v else v) for k, v in m.items()}))
        (out / "log.json").write_text(json.dumps(log, indent=2))
        ck = dict(model=model.state_dict(), epoch=ep, best=best, classes=len(CLASSES))
        torch.save(ck, out / "last.pth")
        if m["miou"] > best:
            best = m["miou"]; ck["best"] = best
            torch.save(ck, out / "best.pth")
            print(f"  new best {best:.4f}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
