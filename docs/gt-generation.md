# Ground-truth generation plan

Every experiment so far points at labels, not models: the lanelet maps are 1-3 m
off the paint, they cover only mapped lanes, and they have no walkway class, so
`walkway` was unsupervised in every number we produced.

## The inefficiency to remove

We label **frames**. The ring road is a loop — the same 100 m of road appears in
dozens of frames, and each annotation carries the same hand-drawn error.
**Label the place once and let every frame inherit it.**

```
accumulate clouds in world frame  ->  ground surface + height
                                  ->  kerb polylines (height step)
                                  ->  annotate polygons ONCE on the canvas
                                  ->  rasterise into per-frame BEV masks
```

## Options, ranked

| # | approach | accuracy | cost | covers |
|---|---|---|---|---|
| 1 | **World accumulation + one-time polygon annotation** | ~10 cm | ~4 h human, once | everywhere we drive |
| 2 | **Kerb detection from LiDAR** | ~5 cm | 1 day eng | walkway boundaries |
| 3 | Orthophoto (12 cm) paving classification | 20-50 cm | 1-2 days | walkway where we haven't driven |
| 4 | Pseudo-label + human audit | model-limited | ongoing | scale |
| 5 | CADC snow free-space (boxes + ground) | cm | 1 day | snow, no semantics |
| 6 | Snap lanelet boundaries to observed kerbs | 1-3 m -> ~20 cm | 2 days | drivable, mapped areas |

LiDAR gives free space and geometry to centimetres but not semantics; kerbs give
the semantic split for free (a 10-15 cm step is a kerb, and the raised side is
walkway). That is why 1 and 2 are done together.

## Built: `map2d/accumulate_world.py`

Takes a manifest of frames and produces a world grid (default 0.10 m) of
`zmin / zmax / count / ground` plus a hill-shaded PNG to annotate on.

Validated on the 13 night frames (poses extracted after the bug fix): a coherent
340 x 470 m map with road corridor, building footprints, kerbs and trees visible
— `docs/img/world_accumulation_night.png`.

**It also exposes pose bugs loudly.** The same script on the day suite produced
five radial "bursts" instead of a map, because those 105 frames still carry the
once-per-bag `utm` bug (5 unique poses). A world accumulation is a much better
assertion than a per-frame check: if the map is not coherent, the poses are
wrong.

## Validation without labels

Free space must never contain an observed obstacle — computable on any frame
with no GT at all. That is the metric a planner cares about, and it is immune to
map error. Track it alongside IoU from now on.

## Kerb detection status

`map2d/kerb_detect.py` is written: it takes the world height grid, finds steps
of 4-30 cm between two locally-flat surfaces, and thins to one cell per edge.
Walls and buildings are excluded by the upper bound; noisy rough ground is
excluded by the flat-both-sides test, which took the raw candidate count from
86k to 9.6k cells.

**It cannot be evaluated yet.** The only world map with correct poses is the 13
night frames (6.8 % of cells observed), and at that sparsity the surface is too
fragmented to trace kerb lines. Judging the detector on it would be judging the
data. The ring-road relay is filling that in.

## Next

1. Fix poses (re-extract, or stream odom only and patch the manifests by stamp).
2. Accumulate the whole loop, detect kerbs, render the canvas.
3. Annotate once (road / sidewalk / crossing polygons).
4. Rasterise to per-frame BEV masks and re-run Stages 3-5.
