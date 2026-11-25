# wato_eve_labels

Vectorised map ground truth for WATonomous **Eve** recordings, derived from Lanelet2
maps. One JSON record per frame: lanes as polylines with boundary attributes,
stop lines, crosswalks, all in the ego frame — plus the raster masks the same
geometry produces.

The point of this repo is to stop duplicating conversion work: if you are
labelling, training or evaluating lane/map perception on Eve bags, start here and
add a session rather than writing a new exporter.

## Layout

| path | contents |
|---|---|
| `vector_gt/vector_gt.jsonl` | **the ground truth**: one JSON object per frame (schema below) |
| `masks/` | 556 raster label files (`npz`): 4 supervised channels + `valid`, 200×200 @ 0.5 m |
| `metadata/frames_manifest.json` | per-frame pose (`utm`, `yaw`), coverage, source dir |
| `metadata/dataset.json` | machine-readable summary: counts, sessions, class/attribute definitions, checksums |
| `metadata/schema/vector_gt.schema.json` | JSON Schema for a frame record (all 556 validate) |
| `maps/` | the Lanelet2 maps the labels were derived from |
| `calib/` | per-source camera/LiDAR calibration used during extraction |
| `scripts/` | the pipeline that produced all of the above |

850 KB of labels per session-equivalent; frames themselves (~2.4 MB each, 1.5 GB
total) are **not** included — only the labels that project onto them.

## What a frame record contains

```json
{"id": "may_30_ring_road_test_5__ring30__test5_f2__frame_0042",
 "session": "test5", "utm": [537012.4, 4813055.1], "yaw": 2.03,
 "lanes": [
   {"id": "1002", "type": "road",
    "centerline": [[x, y], ...], "left_boundary": [...], "right_boundary": [...],
    "left_type": 1, "left_colour": 1, "right_type": 0, "right_colour": 0,
    "speed_limit_mps": 8.3, "turn": null,
    "predecessors": ["998"], "successors": ["1007", "1011"]}],
 "stop_lines": [{"id": "2041", "points": [[x, y], ...]}],
 "crosswalks": [...]}
```

- **ego frame**: x forward, y left, origin at `lidar_cc`, metres — the same frame
  the BEV map head and the planner use.
- **world frame**: UTM 17N (EPSG:32617). `p_ego = R(-yaw) · (p_utm − ego_utm)`.
- **patch**: geometry within **70 m** of the ego position; polylines are clipped
  at the patch edge.
- **boundary type**: `0 dashed, 1 solid, 2 solid_solid, 3 road_edge`;
  **colour**: `0 white, 1 yellow`. See *limitations* — these are derived.

Full field-by-field definition: `metadata/schema/vector_gt.schema.json`.

## Pipeline

Four steps, each script standalone:

```bash
# 1. bag -> frames: stream the bag, one frame every 20 m of travel above 2 m/s
python scripts/stream_extract.py --rclone gdrive:<folder>/<file>.mcap \
    --out frames/<session>/ --spacing 20 --min-speed 2

# 2. frames -> raster labels (4 classes + valid), coverage filter
python scripts/assemble_dataset.py --map maps/ring_road_current.osm \
    --map maps/roads_around_campus.osm --input <session>=frames/<session> \
    --out masks/ --min-coverage 0.10

# 3. frames + poses -> vector ground truth
python scripts/export_vector_gt.py --map maps/ring_road_current.osm \
    --map maps/roads_around_campus.osm --manifest metadata/frames_manifest.json \
    --out vector_gt/ --stats

# 4. (optional) vector GT -> fixed-size training targets
python scripts/vector_targets.py    # K=20 points, clipped to ±49.5 m
```

`lanelet_rasterize.py` does the actual projection (UTM → ego → 200×200 grid) and
is imported by both steps 2 and 3.

## Bags processed

| session | source | frames | lanes/frame | split |
|---|---|---|---|---|
| `oldbag` | earlier ring-road recording (extracted to `dataset_live`) | 181 | 11.9 | train |
| `test3` | `may_30_ring_road_test_3` | 138 | 11.2 | train |
| `test5` | `may_30_ring_road_test_5` | 233 | 11.8 | **val** (held out) |
| `test2` | `may_30_ring_road_test_2` | 4 | 11.5 | unused (coverage filter) |

Total: **556 frames, 6 493 lanes, 253.7 lane-km**, 2.53 crosswalks/frame,
0.74 stop lines/frame.

A second wave, recorded **2026-11-24 with the new VLS-128 LiDAR**, is included
for the frames that fall inside the map:

| session | source | frames | split |
|---|---|---|---|
| `vls24_kingst` | `2026 November 24 (VLS-128 Uncalibrated)/king st run` | 13 | unassigned |
| `vls24_ringday` | `.../ring road run` | 19 | unassigned |

32 of 53 ingested frames passed the coverage filter; the rest left the mapped
area. The VLS-128 turns out to have **uncalibrated reflectivity**, so paint is
not separable from asphalt in the intensity channel — see
[`docs/vls128-2026-11-24.md`](docs/vls128-2026-11-24.md) before planning any
lane extraction from the point cloud. A night session
(`vls24_ringnight`) is recorded but not labelled yet.

Earlier, still unlabelled: 9 further recordings from the Drive bags folder —
only 2 are on the ring road (4 files, ~30 frames pass the coverage filter);
4 are at the test track and 2–3 elsewhere, outside the mapped area.

**Splits are by recording session, never by time.** The ring road is a loop, so a
time split leaks the same geometry into train and val.

## Metadata

`metadata/dataset.json` is generated, not hand-written:

```bash
python3 scripts/make_metadata.py     # needs `jsonschema` to also validate records
```

It records counts, per-session extents, the class and boundary-attribute
definitions, the derivation rules, checksums of every text file, and the caveats
below. If you change the data, re-run it — do not edit it by hand.

## Known limitations

1. **Paint attributes are inferred, not observed.** The source maps carry no paint
   tags (`roads_around_campus.osm` literally says `left_marking: none`), so
   dash/solid and white/yellow come from lane adjacency and direction:
   opposite-direction neighbours → solid yellow, same-direction → dashed white,
   `lane_change=no` → solid white, single lanelet → road edge.
2. **`divider` sits on the lanelet boundary**, which can be 10–30 cm off the
   painted line. That is a systemic label error, not a model error.
3. **Boundaries can be empty or a single point** (~10 %) where the 70 m patch clip
   cut them. Centrelines never degenerate; training targets use the centreline.
4. **Topology is geometrically chained** (end→start within 2 m and 35°), not a
   routing graph. Treat it as a training signal, not routing truth.
5. **Daylight, dry ring-road only.** No night/rain session yet — that is the
   biggest known gap for deployment. (`vls24_ringnight` is recorded but not yet
   labelled.)
6. **The VLS-128 intensity channel is uncalibrated.** On the 2026-11-24
   sessions, white paint measures the same intensity as asphalt
   (p50 22 vs 23), so "bright and near the ground" finds verges, not lane
   markings. Geometry and camera calibration are correct; it is specifically
   reflectivity that is not calibrated.

## Contributing a new session

1. Run steps 1–3 into a **new** `session` name (use the recording folder name).
2. Add the session to the `SESSIONS` table in `scripts/make_metadata.py` with its
   source bag and split.
3. `python3 scripts/make_metadata.py` and commit the regenerated
   `metadata/dataset.json`.
4. In the PR: state the bag, the recording date/conditions, the frame count, and
   whether the recording is inside `maps/` coverage (check `--stats` output — if
   mean coverage is 0 %, the labels are empty and the session should not be added).

Please don't re-export an existing session: check the table above first.
