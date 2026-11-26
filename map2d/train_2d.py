#!/usr/bin/env python3
"""Stage 1: train the 2D segmentation branch on public data.

  train_2d.py --data /users/gnakamot/wato/data/ade20k/ADEChallengeData2016 \
              --out  /users/gnakamot/wato/runs/seg2d_ade20k --epochs 12

Checkpoint is what the EVE fine-tune and the BEV warp start from, so it is
written every epoch and the best validation mIoU is kept separately.
"""
import argparse
import json
import pathlib
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from wato2d.classes import IGNORE, NUM_CLASSES
from wato2d.datasets.ade20k import ADE20K
from wato2d.metrics import Confusion
from wato2d.models.segformer import build_model, count_params


def evaluate(model, loader, dev, limit=0):
    model.eval()
    conf = Confusion()
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if limit and i >= limit:
                break
            x = x.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(x)
            pred = logits.float().argmax(1).cpu().numpy()
            conf.update(pred, y.numpy())
    return conf.iou()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", default="mit_b0")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=6e-5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--val-limit", type=int, default=120)
    ap.add_argument("--resume", default="")
    ap.add_argument("--init", default="", help="start from a checkpoint (weights only)")
    ap.add_argument("--acdc", default="", help="ACDC root; mixes it with ADE20K")
    ap.add_argument("--acdc-ratio", type=float, default=0.5)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    torch.backends.cudnn.benchmark = True

    tr = ADE20K(args.data, "training", args.size, augment=True)
    va = ADE20K(args.data, "validation", args.size, augment=False)
    if args.acdc:
        from wato2d.datasets.acdc import ACDC, Concat
        ad_tr = ACDC(args.acdc, "train", args.size, augment=True)
        ad_va = ACDC(args.acdc, "val", args.size, augment=False)
        print(f"acdc: {len(ad_tr)} train / {len(ad_va)} val", flush=True)
        tr = Concat(tr, ad_tr, args.acdc_ratio)
        from torch.utils.data import ConcatDataset
        va = ConcatDataset([va, ad_va])
    print(f"ade20k: {len(tr)} train / {len(va)} val", flush=True)
    tl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                    pin_memory=True, drop_last=True, persistent_workers=True)
    vl = DataLoader(va, batch_size=args.batch, shuffle=False, num_workers=4,
                    pin_memory=True)

    # class weights: ADE20K is ~83% `other`, so unweighted CE would learn to
    # predict background and ignore the road entirely
    import numpy as _np
    from wato2d.classes import CLASSES as _C
    cnt = _np.zeros(len(_C) + 1, _np.float64)
    for _i in range(0, len(tr), max(1, len(tr) // 80)):
        _v = tr[_i][1].numpy()
        for _c in range(len(_C)):
            cnt[_c] += int((_v == _c).sum())
        cnt[-1] += int((_v == IGNORE).sum())
    present = cnt[:len(_C)] > 0
    freq = cnt[:len(_C)] / max(cnt[:len(_C)].sum(), 1)
    w = _np.zeros(len(_C))
    if present.any():
        inv = 1.0 / _np.sqrt(freq[present])
        # normalise over the classes that exist; absent classes get weight 0 so
        # they cannot skew the normalisation (ADE20K has no `crossing`)
        w[present] = _np.clip(inv / inv.mean(), 0.1, 10.0)
    class_w = torch.tensor(w, dtype=torch.float32, device=dev)
    print("class weights:", {n: round(float(x), 2) for n, x in zip(_C, w)}, flush=True)

    model = build_model(args.arch).to(dev)
    if args.init:
        sd = torch.load(args.init, map_location="cpu")
        model.load_state_dict(sd["model"] if "model" in sd else sd)
        print(f"initialised from {args.init}", flush=True)
    print(f"{args.arch}: {count_params(model):.1f} M params", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = len(tl) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps,
                                                pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda")

    start_epoch, best = 0, 0.0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_epoch = ck["epoch"] + 1
        best = ck.get("best", 0.0)
        print(f"resumed from {args.resume} at epoch {start_epoch}", flush=True)

    log = []
    for ep in range(start_epoch, args.epochs):
        model.train()
        t0, run = time.time(), 0.0
        for i, (x, y) in enumerate(tl):
            x = x.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = F.cross_entropy(model(x), y, weight=class_w, ignore_index=IGNORE)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += float(loss)
            if (i + 1) % 50 == 0:
                print(f"  ep{ep} {i+1}/{len(tl)} loss {run/(i+1):.3f} "
                      f"{(time.time()-t0)/(i+1):.2f}s/it", flush=True)

        m = evaluate(model, vl, dev, args.val_limit)
        print(f"epoch {ep}: train_loss {run/len(tl):.4f}  "
              + " ".join(f"{k}={'--' if v != v else f'{v:.3f}'}" for k, v in m.items()),
              flush=True)
        log.append(dict(epoch=ep, train_loss=run / len(tl), **m))
        (out / "log.json").write_text(json.dumps(log, indent=2))

        ck = dict(model=model.state_dict(), opt=opt.state_dict(),
                  sched=sched.state_dict(), epoch=ep, best=best,
                  classes=NUM_CLASSES, arch=args.arch, size=args.size)
        torch.save(ck, out / "last.pth")
        if m["miou"] > best:
            best = m["miou"]
            ck["best"] = best
            torch.save(ck, out / "best.pth")
            print(f"  new best miou {best:.4f}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
