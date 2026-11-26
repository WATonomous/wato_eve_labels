# map2d — 2D segmentation + LiDAR-DEM BEV

Replacement for the learned camera-LiDAR BEV fusion in `wato-bevfusion`, built
after measuring that the fused map head is *worse* than the camera branch alone
on our car (see `docs/results-2026-11-26.md`).

```
8 panos ─► SegFormer-B0 ─► per-pixel drivable/walkway/crossing
                                    │
LiDAR ─► per-cell ground DEM ───────┼─► back-project each BEV cell onto the DEM
      └─► residual-above-ground ────┘   surface into all 8 cameras
                                    │
                       fuse: drivable = camera AND lidar-free
                             occupied  = lidar obstacle
                             unknown   = unseen / occluded
```

Plan and acceptance criteria: `../docs/plan-2d-bev.md`.

## Layout

| path | role |
|---|---|
| `wato2d/classes.py` | the one class ontology (`other/drivable/walkway/crossing`) + BEV expansion |
| `wato2d/datasets/ade20k.py` | public pretraining corpus |
| `wato2d/datasets/acdc.py` | adverse-weather corpus (Cityscapes labelIds + invalid mask) |
| `wato2d/models/segformer.py` | SegFormer-B0 through smp (3.7 M params) |
| `wato2d/metrics.py` | streaming IoU; averages only over classes with GT pixels |
| `train_2d.py` | stage 1 / 1b training |
| `stage2_labels.py` | BEV GT → per-camera image labels via the ground surface |
| `drive_http.py` | credential-free Drive reader (range requests) |

## Run

```bash
export PYTHONPATH=$PWD
python stage2_labels.py --frames FRAMES --masks MASKS --manifest MANIFEST.json \
    --out eve2d_labels --downscale 4 --overlay 3
python train_2d.py --data ADE20K --acdc ACDC --acdc-ratio 0.6 \
    --init runs/seg2d_ade20k_v2/best.pth --out runs/seg2d_ade20k_acdc \
    --arch mit_b0 --size 512 --epochs 10 --batch 24 --lr 3e-5
```

`stage2_labels.py` needs a separate env from the BEVFusion one (it pulls timm);
it is isolated as `envs/seg2d` on csc3 on purpose.

## Two things that are easy to get wrong

- **mIoU must average only over classes that have GT pixels.** A class with no
  GT is undefined, not zero. Getting this wrong inflated earlier numbers by 30%.
- **Occlusion must be masked out.** Projecting the BEV mask straight into the
  camera paints "drivable" over the car in front; the per-cell
  residual-above-ground test in `stage2_labels.py` removes those cells.
