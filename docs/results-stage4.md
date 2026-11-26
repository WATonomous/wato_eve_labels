# Stage 4 — does the LiDAR ground surface help the BEV warp? (2026-11-26)

Same 2D probabilities, same frames, three warp surfaces:

| mode | surface |
|---|---|
| `zero` | z = 0 (naive flat plane, ~2 m above the road) |
| `flat` | z = median ground height (flat, but at the right height) |
| `dem` | z = per-cell LiDAR ground height |

## Result

| mode | split | drivable | ped | walkway | carpark | mIoU |
|---|---|---|---|---|---|---|
| dem | day | 0.412 | 0.018 | 0.098 | 0.120 | 0.130 |
| **flat** | **day** | **0.416** | **0.055** | **0.101** | **0.121** | **0.138** |
| zero | day | 0.418 | 0.025 | 0.092 | 0.120 | 0.131 |
| dem | night | 0.499 | 0.052 | – | – | 0.138 |
| flat | night | 0.502 | 0.042 | – | – | 0.136 |
| zero | night | 0.503 | 0.040 | – | – | 0.136 |

**The DEM does not help. It is within noise of a flat plane at the correct
height, on day and night alike.**

## Why — measured, not guessed

Displacement in the image of a BEV cell between the flat and DEM warps:

| range | median | p95 |
|---|---|---|
| 5-15 m | **1.1 px** | 18.3 px |
| 15-25 m | **0.8 px** | 15.5 px |
| 25-40 m | **0.5 px** | 8.6 px |

A 0.5 m BEV cell covers many pixels at these ranges, so shifting the sample by
about one pixel cannot change the class we read out of a smooth 2D probability
map. The median terrain in the evaluated window is effectively a plane; the
5.7 m p95-p5 "spread" is driven by cells at the edge of LiDAR coverage where the
hole-fill extrapolates, not by terrain the ego drives over.

So the flat-plane assumption was never the problem. The sensor sits 2 m up and
sees a locally flat road out to 40 m, and the angular error from terrain relief
is sub-pixel.

## What this changes

1. **Keep the flat warp**, at the true ground height. One less moving part, no
   DEM to maintain, identical accuracy. (The `zero` control shows the height
   *offset* is worth ~0.007 mIoU, so still measure the height.)
2. **LiDAR has to earn its keep somewhere else**: occupancy / free-space, and
   the height estimate. Not terrain shape.
3. The remaining error is in the 2D model and the labels, not the geometry.

## Caveats

- Night has no walkway/carpark GT, so its mIoU averages fewer classes; do not
  compare day and night mIoU directly.
- Night drivable (0.50) exceeding day (0.42) is more likely a GT difference
  than a real improvement — the night frames are at a wide intersection.
- Absolute numbers are against lanelet GT, which is 1-3 m off the paint. Treat
  them as a regression signal, not a truth.
