# ACDC status (2026-11-26)

## What we have

| | fog | night | rain | snow |
|---|---|---|---|---|
| GT train | 400 | 400 | 400 | 400 |
| GT val | 100 | 106 | 100 | 100 |
| **RGB train** | **400** | **400** | **400** | **0** |
| RGB val | 100 | 106 | 100 | 100 |

The RGB archive (`rgb_anon_trainvaltest.zip`, ~20 GB from
`acdc.vision.ee.ethz.ch`) died at 15.76 GB with `NSURLErrorTimedOut` after
3 h 25 m. The signed download token in Safari's `Info.plist` had expired, so
`curl -C -` could not resume (HTTP 400).

**Salvaged with `bsdtar -xf` on the truncated archive:** 7,532 PNGs before the
truncation point, covering fog/night/rain train+val completely, all four test
sets, snow val, and 3,926 reference images. Only **snow/train RGB** is missing
(its GT is intact).

`wato2d/datasets/acdc.py` skips GT entries whose RGB is absent, so training runs
on the 1,506 usable train images rather than crashing.

## Consequence

Stage 1b trains on fog + night + rain and validates on all four conditions
including snow, which is the useful direction: snow is evaluated but not
memorised.

A fresh `rgb_anon_trainvaltest.zip` download would add snow/train (400 images)
and the test splits (unneeded). Not worth re-downloading unless snow training
turns out to matter.

## Licence

Research-only. Cannot redistribute, so the images stay out of this repository.
