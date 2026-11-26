# Implementation plan — 2D segmentation + LiDAR-DEM BEV

Replaces the learned camera-LiDAR BEV fusion for the EVE map head. Rationale and
measurements: `~/wato/agent/outbox/2026-11-26-class-parity.md` and AGENTS.md.

```
8 panos ─► SegFormer-B0 ─► per-pixel drivable/walkway/crossing (320x256)
                                    │
LiDAR ─► per-cell ground DEM ───────┼─► back-project each BEV cell's 3D point
      └─► residual-above-ground ────┘   onto the DEM surface into all 8 cameras
                                    │
                       fuse: drivable = camera AND lidar-free
                             occupied  = lidar obstacle
                             unknown   = unseen / occluded
```

## Stage 0 — data (partly blocking)

- [x] king st run (4 mcaps), ring road run (4 mcaps)
- [ ] **night run** (`180622`, 4 mcaps, 1 panel) — same rig, queued for extraction
- [ ] **pose fix**: `extract_sweeps.py` now writes `utm` per frame (the old suite
      used one pose per bag, which invalidated absolute mIoU)
- [ ] **extract all 8 cameras** (we currently drop `ee`/`ww`, leaving a ±90° blind wedge)
- spacing 10 s, single scan (215k points already matches training density)

## Stage 1 — public pretraining  ← RUNNING

`wato-map2d/`, SegFormer-B0 (3.7 M params), 4 classes `other/drivable/walkway/crossing`,
class weights from frequency, ADE20K 20k images, 512², 12 epochs (~20 min/epoch).

Data note: ADE20K is the only large driving-relevant corpus that downloads
without a login. **Cityscapes and ACDC (fog/night/rain) both need a manual
download** — if you grab ACDC and drop it at `~/wato/data/acdc`, stage 1 gets a
second pass on genuinely adverse imagery, which matters for the night target.

- output: `runs/seg2d_ade20k/best.pth`

## Stage 2 — EVE image labels from the BEV GT

No new labelling: project the existing rasterised BEV masks back into each
camera using the LiDAR ground surface.

- for each BEV cell, take `(x, y, z_ground)` from the DEM, project into each
  camera, write the cell's class into that pixel
- dilate 3 px, ignore everything else
- the result is sparse but *correct*, and it doubles as a projection check:
  the labels must land on road in the image or the calibration is wrong
- split **by session**, never by frame

- output: `runs/eve2d/labels/<session>/<cam>/<frame>.npy`

## Stage 3 — EVE fine-tune

- init from stage 1, freeze the encoder for 1 epoch then unfreeze at 1/10 lr
- augmentation: brightness/gamma/contrast jitter (night robustness), flip, small
  rotation; **no** colour inversion — night is not a hue shift
- loss: class-weighted CE + ignore, same weights as stage 1

## Stage 4 — BEV reconstruction + fusion  ← the experiment that matters

Two paths, same weights:

| ablation | warp surface | lidar role |
|---|---|---|
| A | flat plane at z=0 | none (camera only) |
| B | **per-cell DEM** | geometry only |
| C | per-cell DEM | geometry + occupancy gate |

- DEM from the point cloud: per 1 m cell, 5th-percentile z, hole-filled
- occupancy: max residual above ground per 0.25 m cell; free if < 0.20 m
- fusion is a rule, not a network (debuggable, three output states)

## Stage 5 — evaluation and ablations

Metrics, in this order:

1. **drivable BEV IoU** vs our rasterised masks (5-class parity, same classes as
   nuScenes so numbers are comparable)
2. **boundary accuracy vs LiDAR-observed kerb** — the metric that is not
   corrupted by 1-3 m map error
3. **free-space violations**: free cells inside observed obstacles (should be ~0)
4. **night vs day split** — the actual question
5. temporal consistency across revisits (we have poses)

Acceptance: drivable BEV IoU ≥ 0.60 day / ≥ 0.50 night; A→B improves boundary
error ≥ 20 %; C removes ≥ 90 % of free-space violations.

## Cost

| stage | wall clock |
|---|---|
| 1 | ~30 min |
| 2 | half a day of engineering |
| 3 | ~1 GPU-day |
| 4 + 5 | ~1 day |

Total ~2.5 days of GPU plus engineering, versus the BEVFusion fine-tune path
which needs 4-6 k labelled frames before it is even measurable.

## Interoperability

The BEVFusion stack is not thrown away: `head.seg` stays available as a
comparison arm, and the 2D branch can be exported to ONNX and composed the same
way (per-engine, FP16) at roughly a fifth of the latency.
