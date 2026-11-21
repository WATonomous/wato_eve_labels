#!/usr/bin/env python3
"""Assemble every labelled frame into one dataset with an explicit session split.

Frames stay where the extractor wrote them (home quota is tight); this writes only
masks plus a manifest that references the frame paths. The `session` field is what
makes a legitimate train/val split possible: recordings are separate drives, so
splitting by session does not leak like a time split inside one loop does.

  assemble_dataset.py --map maps/ring_road_current.osm \
      --map maps/roads_around_campus.osm \
      --input oldbag=~/wato_monorepo/bags/dataset_live \
      --input test3=~/wato_monorepo/bags/ring30/may_30_ring_road_test_3_f0 \
      --input test3=~/wato_monorepo/bags/ring30/may_30_ring_road_test_3_f1 \
      --out ~/wato_monorepo/bags/dataset_v1 --min-coverage 0.10
"""
import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, "/home/raine/wato-cuda-bevfusion")
from lanelet_rasterize import CLASSES, SIZE, load_map, rasterize  # noqa: E402


def merge_maps(paths):
    out = {name: {"polys": [], "bbox": []} for name in CLASSES}
    for p in paths:
        m = load_map(p)
        for name in CLASSES:
            out[name]["polys"].extend(m[name]["polys"])
            if len(m[name]["bbox"]):
                out[name]["bbox"].append(m[name]["bbox"])
    for name in CLASSES:
        out[name]["bbox"] = (np.vstack(out[name]["bbox"]) if out[name]["bbox"]
                             else np.zeros((0, 4)))
    out["_all"] = out["drivable_area"]["polys"] + out["ped_crossing"]["polys"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", action="append", required=True)
    ap.add_argument("--input", action="append", required=True,
                    help="session=dir (repeatable; same session label may appear twice)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-coverage", type=float, default=0.10)
    ap.add_argument("--flip-y", action="store_true")
    args = ap.parse_args()

    m = merge_maps(args.map)
    print("map union: " + ", ".join(f"{k}={len(m[k]['polys'])}" for k in CLASSES))

    out = pathlib.Path(args.out).expanduser()
    (out / "masks").mkdir(parents=True, exist_ok=True)

    manifest, per_session, skipped = [], {}, 0
    for spec in args.input:
        label, _, d = spec.partition("=")
        d = pathlib.Path(d).expanduser()
        if not d.is_dir():
            print(f"  skip {label}: {d} missing")
            continue
        files = [f for f in sorted(list(d.glob("frame_*.npz")) + list(d.glob("sample_*.npz")))
                 if not f.stem.endswith("_mask")]   # skip our own mask files
        kept = 0
        for fp in files:
            f = np.load(fp, allow_pickle=True)
            if "utm" not in f:
                skipped += 1
                continue
            ex, ey, yaw = [float(v) for v in f["utm"]]
            if not np.isfinite([ex, ey, yaw]).all():
                skipped += 1
                continue
            masks, valid = rasterize(m, np.array([ex, ey]), yaw, flip_y=args.flip_y)
            cov = float(valid.mean())
            if cov < args.min_coverage:
                skipped += 1
                continue
            # unique id per (session, source dir, frame)
            sid = f"{label}__{d.name}__{fp.stem}"
            np.savez_compressed(out / "masks" / f"{sid}.npz", valid=valid,
                                **{k: masks[k] for k in CLASSES})
            manifest.append({
                "id": sid,
                "session": label,
                "source_dir": str(d),
                "frame_path": str(fp),
                "utm": [ex, ey],
                "yaw": yaw,
                "travelled_m": float(f["travelled"]) if "travelled" in f else None,
                "speed_ms": float(f["speed"]) if "speed" in f else None,
                "coverage": cov,
                "px": {k: int(masks[k].sum()) for k in CLASSES},
            })
            kept += 1
        per_session[label] = per_session.get(label, 0) + kept
        print(f"  {label:24s} {d.name:32s} kept {kept:4d}/{len(files)}")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\ntotal frames: {len(manifest)}   skipped (coverage/pose): {skipped}")
    for label, n in sorted(per_session.items()):
        print(f"  session {label:22s} {n:5d} frames")
    if manifest:
        print(f"mean coverage: {np.mean([r['coverage'] for r in manifest]):.1%}")
        for k in CLASSES:
            frac = np.mean([r["px"][k] for r in manifest]) / (SIZE * SIZE)
            present = sum(1 for r in manifest if r["px"][k] > 0)
            print(f"  {k:14s} mean {frac*100:5.2f}% of grid, "
                  f"non-empty in {present}/{len(manifest)}")
    print(f"\ndataset: {out}")


if __name__ == "__main__":
    main()
