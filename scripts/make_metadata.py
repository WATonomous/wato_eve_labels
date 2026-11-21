#!/usr/bin/env python3
"""Generate metadata/dataset.json from the data actually in this repo.

Everything in here is computed (counts, extents, boundary-type histograms,
checksums) so the metadata cannot drift from the files. Run it after changing
any data:

    python3 scripts/make_metadata.py

It also validates every line of vector_gt.jsonl against
metadata/schema/vector_gt.schema.json when `jsonschema` is installed.
"""
import collections
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
GT = ROOT / "vector_gt" / "vector_gt.jsonl"
SCHEMA = ROOT / "metadata" / "schema" / "vector_gt.schema.json"
OUT = ROOT / "metadata" / "dataset.json"

BOUNDARY = {0: "dashed", 1: "solid", 2: "solid_solid", 3: "road_edge"}
COLOUR = {0: "white", 1: "yellow"}

# provenance of the sessions we have processed; dates are the recording dates
# where known, otherwise the Drive folder / bag name
SESSIONS = {
    "oldbag": {
        "frames": None,
        "source": "earlier ring-road recording, frames extracted to dataset_live",
        "split": "train",
        "notes": "the original ring-road drive used for the first fine-tune",
    },
    "test2": {
        "frames": None,
        "source": "may_30_ring_road_test_2 (Drive bags folder 1293dKJLuDN4_5KQgk3ToORRmp8fOuODu)",
        "split": "unused",
        "notes": "only 4 frames survived the coverage filter",
    },
    "test3": {
        "frames": None,
        "source": "may_30_ring_road_test_3 (same Drive folder)",
        "split": "train",
        "notes": "second ring-road session, May 30 2026",
    },
    "test5": {
        "frames": None,
        "source": "may_30_ring_road_test_5 (same Drive folder)",
        "split": "val",
        "notes": "held out; different session so there is no time leakage",
    },
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def poly_len(pts):
    return sum(((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5
               for a, b in zip(pts, pts[1:]))


def main():
    rows = [json.loads(l) for l in open(GT)]
    schema = json.loads(SCHEMA.read_text())

    try:
        import jsonschema
        for r in rows:
            jsonschema.validate(r, schema)
        validated = True
    except ImportError:
        validated = None

    per_session = collections.defaultdict(lambda: {
        "frames": 0, "lanes": 0, "lane_m": 0.0,
        "stop_lines": 0, "crosswalks": 0,
        "utm_e": [], "utm_n": [], "with_lane": 0,
    })
    quality = collections.Counter()
    btype = collections.Counter()
    bcol = collections.Counter()
    lane_type = collections.Counter()
    succ = 0

    for r in rows:
        s = per_session[r["session"]]
        s["frames"] += 1
        s["utm_e"].append(r["utm"][0])
        s["utm_n"].append(r["utm"][1])
        s["stop_lines"] += len(r.get("stop_lines", []))
        s["crosswalks"] += len(r.get("crosswalks", []))
        if r["lanes"]:
            s["with_lane"] += 1
        for lane in r["lanes"]:
            s["lanes"] += 1
            quality["lanes"] += 1
            if len(lane["centerline"]) < 2:
                quality["centerline_degenerate"] += 1
            for side in ("left_boundary", "right_boundary"):
                if len(lane[side]) < 2:
                    quality[f"{side}_degenerate"] += 1
            s["lane_m"] += poly_len(lane["centerline"])
            btype[BOUNDARY.get(lane["left_type"], lane["left_type"])] += 1
            btype[BOUNDARY.get(lane["right_type"], lane["right_type"])] += 1
            bcol[COLOUR.get(lane["left_colour"], lane["left_colour"])] += 1
            bcol[COLOUR.get(lane["right_colour"], lane["right_colour"])] += 1
            lane_type[lane["type"]] += 1
            succ += len(lane.get("successors", []))

    sessions = {}
    for name, s in sorted(per_session.items()):
        meta = dict(SESSIONS.get(name, {"source": "unknown", "split": "unknown", "notes": ""}))
        meta.update({
            "frames": s["frames"],
            "lanes_total": s["lanes"],
            "lanes_per_frame": round(s["lanes"] / s["frames"], 1),
            "lane_km": round(s["lane_m"] / 1000, 2),
            "stop_lines_total": s["stop_lines"],
            "crosswalks_total": s["crosswalks"],
            "utm_extent_m": [round(max(s["utm_e"]) - min(s["utm_e"])),
                             round(max(s["utm_n"]) - min(s["utm_n"]))],
            "utm_centre": [round(sum(s["utm_e"]) / s["frames"]),
                           round(sum(s["utm_n"]) / s["frames"])],
        })
        meta.pop("frames_placeholder", None)
        sessions[name] = meta

    files = {}
    for p in sorted(ROOT.rglob("*")):
        if p.is_file() and ".git" not in p.parts and p != OUT:
            rel = str(p.relative_to(ROOT))
            if p.suffix in (".jsonl", ".json", ".osm", ".py", ".md") or p.suffix == ".npz":
                if p.suffix == ".npz":      # 556 of these; hash the manifest instead
                    continue
                files[rel] = sha256(p)

    doc = {
        "$schema": "../metadata/schema/vector_gt.schema.json",
        "name": "wato_eve_labels",
        "version": "0.1.0",
        "generated": "run scripts/make_metadata.py",
        "purpose": "Vectorised map ground truth for lane/map perception on WATonomous Eve "
                   "recordings, derived from Lanelet2 maps. Shared so dataset work can be "
                   "distributed without duplicating conversion effort.",
        "frames_file": "vector_gt/vector_gt.jsonl",
        "frame_schema": "metadata/schema/vector_gt.schema.json",
        "frames_manifest": "metadata/frames_manifest.json",
        "frame_records_validated_against_schema": validated,
        "coordinate_frames": {
            "ego": {"axes": "x forward, y left", "origin": "lidar_cc", "units": "metres",
                    "note": "same frame the BEV map head and the planner use"},
            "world": {"crs": "UTM zone 17N, WGS84", "epsg": 32617, "units": "metres"},
            "transform": "p_ego = R(-yaw) * (p_utm - ego_utm), yaw from the frame record",
        },
        "patch": {
            "clip_radius_m": 70.0,
            "note": "only geometry within 70 m of the ego position is kept; polylines are "
                    "clipped at the patch edge, not truncated to their first point",
        },
        "raster_grid": {
            "size": [200, 200], "resolution_m": 0.5, "extent_m": 100,
            "rows": "+y", "cols": "+x", "flip_y": False,
            "note": "used by masks/ and by the segmentation head; see scripts/lanelet_rasterize.py",
        },
        "classes": {
            "vector": {"lane_type": dict(lane_type)},
            "raster": ["drivable_area", "ped_crossing", "walkway", "stop_line",
                       "carpark_area", "divider"],
            "supervised": ["drivable_area", "ped_crossing", "divider", "stop_line"],
            "note": "walkway and carpark_area are not present in these maps and get zero "
                    "class weight; the six-channel head contract is unchanged",
        },
        "boundary": {
            "type": BOUNDARY,
            "colour": COLOUR,
            "derived": True,
            "derivation": {
                "shared_by_opposite_direction_lanelets": "solid, yellow",
                "shared_by_same_direction_lanelets": "dashed, white",
                "same_direction_with_lane_change=no": "solid, white",
                "boundary_of_a_single_lanelet": "road_edge, white",
            },
            "caveat": "the source maps carry no paint tags (roads_around_campus.osm says "
                      "left_marking/right_marking = none), so dash/solid and white/yellow are "
                      "geometric inferences, not observed paint",
            "observed_counts": {"type": dict(btype), "colour": dict(bcol)},
        },
        "topology": {
            "method": "geometric chaining: end->start within 2 m and heading within 35 deg",
            "successor_edges": succ,
            "note": "OSM Lanelet2 does not store successors; this is an approximation, so "
                    "treat it as a training target, not routing ground truth",
        },
        "splits": {
            "train": [k for k, v in sessions.items() if v["split"] == "train"],
            "val": [k for k, v in sessions.items() if v["split"] == "val"],
            "unused": [k for k, v in sessions.items() if v["split"] == "unused"],
            "rule": "split by recording session, never by time: the ring road is a loop, so "
                    "a time split leaks the same geometry into train and val",
        },
        "sessions": sessions,
        "counts": {
            "frames": len(rows),
            "lanes": sum(len(r["lanes"]) for r in rows),
            "stop_lines": sum(len(r.get("stop_lines", [])) for r in rows),
            "crosswalks": sum(len(r.get("crosswalks", [])) for r in rows),
            "lane_km": round(sum(poly_len(l["centerline"]) for r in rows for l in r["lanes"]) / 1000, 1),
        },
        "contents": {
            "vector_gt/vector_gt.jsonl": "the vector ground truth, one JSON object per frame",
            "masks/": "raster ground truth (4 supervised channels + valid), one npz per frame, "
                      "regenerated by scripts/assemble_dataset.py",
            "calib/": "per-source camera/LiDAR calibration npz used when the frames were extracted",
            "maps/": "Lanelet2 maps the labels were derived from",
            "scripts/": "the pipeline that produced all of the above",
            "metadata/frames_manifest.json": "per-frame pose/coverage record from the extractor",
        },
        "scripts": {
            "stream_extract.py": "bag -> frames: rclone HTTP-range streaming, one frame per 20 m "
                                 "of travel above 2 m/s; writes 6 camera JPEGs + LiDAR points + calib",
            "lanelet_rasterize.py": "Lanelet2 -> BEV raster (4 classes + valid) on the 200x200 @0.5 m grid",
            "assemble_dataset.py": "frames -> masks/manifest, applies the coverage filter (>=10%)",
            "export_vector_gt.py": "frames -> vector_gt.jsonl: lanes/boundaries/topology in ego frame",
            "vector_targets.py": "vector GT -> fixed-size training targets (K=20 points, clipped)",
            "make_metadata.py": "regenerates this file from the data",
        },
        "quality": {
            **dict(quality),
            "note": "boundaries can degenerate to 0-1 points where the 70 m patch clip cut "
                    "them; centrelines never do. Training targets use the centreline, so this "
                    "does not affect the vector model's targets, but a boundary-predicting "
                    "head should filter these.",
        },
        "caveats": [
            "paint attributes (dash/solid, white/yellow) are derived from lane adjacency, not observed",
            "divider labels come from lanelet geometry, which can sit ~10-30 cm off the painted line",
            "topology is geometrically chained, not a routing graph",
            "frames themselves (~2.4 MB each, 1.5 GB total) are not in this repo; only the labels are",
            "sessions were recorded in daylight on the ring road; no night/rain coverage yet",
        ],
        "file_checksums": files,
    }

    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}  ({OUT.stat().st_size / 1024:.0f} KB)")
    print(f"frames {doc['counts']['frames']}  lanes {doc['counts']['lanes']}  "
          f"lane_km {doc['counts']['lane_km']}  schema_validated={validated}")
    print("sessions:", {k: v["frames"] for k, v in sessions.items()})


if __name__ == "__main__":
    main()
