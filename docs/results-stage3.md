# Stage 3 — first fine-tune on the car (2026-11-26)

2D branch fine-tuned from the ADE20K checkpoint on Eve frames whose labels come
from Stage 2 (BEV ground truth projected through the LiDAR ground surface).

## Data

| | |
|---|---|
| frames | 118 (105 day from the Nov-24 suite + 13 night) |
| views | 768 train / 22 val (8 cameras each) |
| split | **by session** — `ring3` held out |
| labelled pixels | ~17-25 % drivable, 1-3 % walkway, rest unlabelled |

Label coverage per session: `kingst1` 24.7 % drivable / 3.0 % walkway,
`night0_0` 19.5 % drivable / 0.4 % crossing, `ring3` 17.1 % / 1.1 %.

## Result — image-space IoU, held-out session

| epoch | drivable | walkway | mIoU |
|---|---|---|---|
| 22 | 0.754 | 0.592 | 0.673 |
| 24 | 0.797 | 0.541 | 0.669 |
| 29 | 0.786 | 0.531 | 0.659 |

`other` and `crossing` are undefined on the holdout (no GT pixels), so mIoU
averages the classes that exist — the same rule as everywhere else here.

## What this does and does not show

It shows the chain works end to end: BEV GT → labels → training → a model that
segments road and footway on a drive it has never seen.

It does **not** show the thing we actually care about, which is BEV IoU after
warping, and whether the LiDAR DEM beats a flat plane. That is Stage 4.

## Bugs found while building it (both would have been silent)

1. **`other` had zero supervision.** The BEV mask only covers lanelet polygons,
   which are road, so no pixel was ever labelled "not road" — the fine-tune
   would have drifted towards predicting drivable everywhere while the pretrained
   notion of scenery was the only thing holding it back. Fixed by keeping the
   pretrained `other` behaviour (weight 0 rather than eroding it) and by
   labelling occupied BEV cells as `other` where such cells exist.
2. **`torch.tensor(w, np.float32)`** — the second positional arg is `requires_grad`,
   so the class weights were being interpreted as a bool.

## Next

Stage 4: warp these per-camera probabilities onto the DEM surface, compare
against a flat plane, and score BEV IoU on day and night separately.
