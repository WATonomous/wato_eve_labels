# 2D free-space pipeline — status for review (2026-11-26)

## What the pipeline does now

8 pano cameras → SegFormer-B0 → per-pixel drivable / walkway / crossing →
back-projected onto the LiDAR ground surface into a 200×200 @ 0.5 m BEV →
fused with LiDAR occupancy. Trained on ADE20K + ACDC (adverse weather), then
fine-tuned on our own frames, split by session.

## Numbers, held-out session (`ring3`)

| checkpoint | drivable | walkway | mIoU |
|---|---|---|---|
| ADE20K only | 0.767 | 0.817 | 0.792 |
| + ACDC (adverse) | 0.795 | 0.792 | 0.794 |
| **+ EVE fine-tune v2** | **0.817** | 0.453 | 0.635 |

v2 deliberately spends aggregate mIoU to fix a failure mode. In v1 `other` had
**zero** training weight — the BEV mask only covers road polygons, so no pixel is
ever labelled "not road" — and argmax therefore sent everything that was not
drivable to *walkway*. v1 scored a higher mIoU (0.659) while painting sky and
buildings as footway. v2 floors the `other` weight at 0.5, and the pictures
become qualitatively correct.

## Where it is good and where it is not

- **Day, arterial (King St)**: road corridor and footway strips come out roughly
  right; the predicted BEV resembles the GT block layout.
- **Night, ring road**: the model paints essentially the whole forward field as
  drivable. That is the dangerous direction for a planner — false free space.
  Night is also where our labels are weakest (13 frames) and where ACDC marks
  33 % of night pixels invalid.

## What is missing

1. **Labels.** Our GT is lanelet polygons, 1–3 m off the paint, with no walkway
   class; walkway exists only because we added OSM footpaths. Every number above
   is measured against that.
2. **Density.** 118 frames total, 13 of them night.
3. **Occlusion.** The BEV warp has no visibility test yet, so a prediction on a
   cell hidden behind a parked car still counts.
4. **Temporal.** Frames are 10 s apart and independent; nothing accumulates
   evidence across passes yet. That is the next real quality lever after labels.

## Images

- `docs/img/freespace_day_example.png` — King St, day (CAM_FRONT overlay |
  predicted BEV | lanelet GT)
- `docs/img/freespace_night_example.png` — ring road, night (same three panels)
- `~/wato/media/freespace_viz/` — 8 frames (4 day, 4 night) plus a contact sheet

## How to read the figures

Green = drivable, blue = walkway, orange = crossing. The right panel is the
lanelet ground truth for the same frame, which is itself 1–3 m off the real
paint — so treat left-vs-right as "does the shape agree", not as an accuracy
measurement.
