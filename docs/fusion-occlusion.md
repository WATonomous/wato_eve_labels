# Cross-camera weighting, occlusion and occupancy

The first BEV back-projection was naive: every camera that could see a cell
contributed with equal weight, nothing checked visibility, and any return above
a global ground threshold counted as an obstacle. Three consequences, all
visible in the early figures: radial seams between cameras, free space painted
behind parked cars, and red speckle over the whole road.

## What changed

**1. Weighted view selection instead of averaging.** Averaging was wrong because
the model is not view-invariant: the same road cell looks like road to
`CAM_FRONT` and like verge to `CAM_LEFT`, and the average of the two is neither.
Each camera's vote is now weighted by

| term | form | what it encodes |
|---|---|---|
| viewing angle | `cos^4` of the angle from the optical axis | a camera sees best what is in front of it |
| distance | `1 / (1 + z/40)` | a cell at 5 m is resolved by far more pixels than at 40 m |
| confidence | `1 / (1 + H)` where `H` is normalised entropy | an unsure camera does not outvote a sure one |

and cells whose ray is more than ~55° off-axis are rejected outright
(`cos^2 < 0.33`), because there the ground is at grazing incidence and one pixel
spans metres.

**2. Occlusion with a LiDAR depth buffer.** Points are projected into each camera
to build a sparse depth image (4× downsampled, min depth per block). A cell is
rejected for a camera if something is observed along the same ray more than
0.8 m closer. This is what stops the map painting free space behind a vehicle.

**3. Occupancy from per-cell height extent.** The first version measured height
above the smoothed ground surface, which on a slope is wrong by tens of
centimetres and turned the road red. Now the test is the cell's own vertical
spread (`max z - min z`, with at least 3 returns): flat tarmac spans a few
centimetres, a car spans about a metre. Independent of the ground model and of
the datum.

**4. Confidence gating and honest unknown.** A cell only gets a class if that
class beats both the threshold *and* the runner-up by a margin, with at least two
views. Cells beyond 40 m, cells no camera resolves, and cells LiDAR marks as
occupied are labelled rather than guessed at.

## Measured effect (clip_day, 3 frames)

| state | before | after |
|---|---|---|
| drivable | 0.231 | 0.069 |
| walkway | 0.213 | 0.189 |
| occupied | 0.119 | 0.043 |
| unknown | 0.436 | 0.699 |

The drop in "drivable" is not a regression, it is the margin rule refusing to
commit where the four classes are close. The previous 0.231 included a lot of
cells the model was guessing at.

## What is still wrong

The map is now *honest*, not *good*. Projecting a cell at 20 m into `CAM_FRONT`
and reading the model's row there gives roughly 45 % walkway on a stretch that is
plainly road: the 2D model itself mislabels mid-range road, which no amount of
fusion tuning fixes. That is the label problem again — the fine-tune set has 118
frames and the walkway labels come from OSM footpaths.
