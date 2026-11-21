#!/usr/bin/env python3
"""Export lanelet2 maps to per-frame vector GT in the planner's Lane schema.

The target is `world_modeling/lanelet_msgs/Lanelet.msg`, i.e. what
`lattice_planning` already consumes: geometry as polylines, boundary type and
colour, lane type, speed limit, turn direction and topology. A vector map model
trained on this emits something the planner can read without a rewrite.

Two things the maps do NOT contain, which we derive:

  * boundary type/colour -- derived from adjacency:
        shared by two lanelets travelling in opposite directions -> yellow solid
        shared by two lanelets in the same direction             -> white dashed
        shared by same-direction lanelets with lane_change=no    -> white solid
        boundary of a single lanelet (road edge)                 -> road edge
    (derive_colour=False keeps road_edge/none as-is for maps that do tag paint)

  * topology -- successors/predecessors are not stored in OSM lanelet2, so they
    are chained geometrically: a lanelet that starts where another ends, within
    TOL_M and heading within TOL_DEG.

  export_vector_gt.py --map maps/ring_road_current.osm \
      --manifest /users/gnakamot/wato/map_dataset/manifest.json \
      --map2 maps/roads_around_campus.osm --out vector_gt --stats
"""
import argparse
import json
import math
import pathlib
import sys

import numpy as np
import xml.etree.ElementTree as ET

# boundary type codes, matching lanelet_msgs/Lanelet.msg
DASHED, SOLID, SOLID_SOLID, ROAD_EDGE = 0, 1, 2, 3
# boundary colour codes
WHITE, YELLOW = 0, 1

TOL_M = 2.0          # successor/predecessor chaining tolerance
TOL_DEG = 35.0       # ...and heading agreement
STEP_M = 2.0         # centreline resampling step


# ------------------------------------------------------------------ WGS84 -> UTM
def utm(lat, lon, zone=17):
    """WGS84 -> UTM easting/northing (metres). Same maths as lanelet_rasterize."""
    a, f = 6378137.0, 1 / 298.257223563
    e2 = f * (2 - f)
    k0 = 0.9996
    lat, lon = math.radians(lat), math.radians(lon)
    lam0 = math.radians((zone - 1) * 6 - 180 + 3)
    N = a / math.sqrt(1 - e2 * math.sin(lat) ** 2)
    T = math.tan(lat) ** 2
    C = e2 / (1 - e2) * math.cos(lat) ** 2
    A = (lon - lam0) * math.cos(lat)
    M = a * ((1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256) * lat
             - (3 * e2 / 8 + 3 * e2 ** 2 / 32 + 45 * e2 ** 3 / 1024) * math.sin(2 * lat)
             + (15 * e2 ** 2 / 256 + 45 * e2 ** 3 / 1024) * math.sin(4 * lat)
             - (35 * e2 ** 3 / 3072) * math.sin(6 * lat))
    east = k0 * N * (A + (1 - T + C) * A ** 3 / 6
                     + (5 - 18 * T + T ** 2 + 72 * C - 58 * e2) * A ** 5 / 120) + 500000.0
    north = k0 * (M + N * math.tan(lat) * (A ** 2 / 2 + (5 - T + 9 * C + 4 * C ** 2) * A ** 4 / 24
                                          + (61 - 58 * T + T ** 2 + 600 * C - 330 * e2) * A ** 6 / 720))
    return east, north


# ------------------------------------------------------------------ geometry
def resample(poly, step=STEP_M):
    """Resample a polyline at a fixed arc-length step; keeps both endpoints."""
    p = np.asarray(poly, dtype=float)
    if len(p) < 2:
        return p
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-6:
        return p[:1]
    n = max(2, int(math.ceil(s[-1] / step)) + 1)
    t = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], axis=1)


def resample_n(poly, n):
    """Interpolate a polyline to exactly n points, evenly spaced by arc length."""
    p = np.asarray(poly, dtype=float)
    if len(p) < 2:
        return np.repeat(p[:1], n, axis=0)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-6:
        return np.repeat(p[:1], n, axis=0)
    t = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], axis=1)


def heading(p):
    d = np.asarray(p[-1]) - np.asarray(p[0])
    return math.atan2(d[1], d[0])


def poly_len(p):
    p = np.asarray(p, float)
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) if len(p) > 1 else 0.0


def ang_diff(a, b):
    d = abs(a - b) % (2 * math.pi)
    return math.degrees(min(d, 2 * math.pi - d))


def clip(poly, ego, radius):
    """Keep only the runs of a polyline that lie inside the radius."""
    p = np.asarray(poly, float)
    keep = np.linalg.norm(p - np.asarray(ego, float), axis=1) <= radius
    if not keep.any():
        return p[:0]
    runs, start = [], None
    for i, k in enumerate(keep):
        if k and start is None:
            start = i
        elif not k and start is not None:
            runs.append(p[start:i]); start = None
    if start is not None:
        runs.append(p[start:])
    return max(runs, key=len) if runs else p[:0]


def in_radius(poly, ego, radius):
    p = np.asarray(poly, float)
    return bool(np.min(np.linalg.norm(p - ego, axis=1)) <= radius)


# ------------------------------------------------------------------ map load
class LaneMap:
    def __init__(self, path, derive_colour=True):
        root = ET.parse(path).getroot()
        self.nodes = {}
        for n in root.findall("node"):
            if n.get("lat") and n.get("lon"):
                self.nodes[n.get("id")] = utm(float(n.get("lat")), float(n.get("lon")))
        self.ways, self.way_tags = {}, {}
        for w in root.findall("way"):
            nd = [self.nodes[r.get("ref")] for r in w.findall("nd") if r.get("ref") in self.nodes]
            if len(nd) >= 2:
                self.ways[w.get("id")] = np.array(nd, dtype=float)
                self.way_tags[w.get("id")] = {c.get("k"): c.get("v") for c in w.findall("tag")}

        self.lanelets = {}        # id -> dict(left_way, right_way, tags)
        self.way_use = {}         # way id -> [lanelet ids]
        self.stop_lines, self.crosswalk_ways, self.traffic = [], [], []
        for rel in root.findall("relation"):
            tags = {c.get("k"): c.get("v") for c in rel.findall("tag")}
            t = tags.get("type")
            if t == "lanelet":
                left = right = None
                for m in rel.findall("member"):
                    if m.get("type") != "way":
                        continue
                    if m.get("role") == "left":
                        left = m.get("ref")
                    elif m.get("role") == "right":
                        right = m.get("ref")
                if left in self.ways and right in self.ways:
                    lid = rel.get("id")
                    self.lanelets[lid] = {"left": left, "right": right, "tags": tags}
                    self.way_use.setdefault(left, []).append(lid)
                    self.way_use.setdefault(right, []).append(lid)
            elif t in ("pedestrian_marking", "zebra_marking"):
                for m in rel.findall("member"):
                    if m.get("type") == "way" and m.get("ref") in self.ways:
                        self.crosswalk_ways.append(m.get("ref"))
        for lid, L in self.lanelets.items():      # crosswalk lanelets (ring-road map)
            if L["tags"].get("subtype") == "crosswalk":
                self.crosswalk_ways.extend([L["left"], L["right"]])
        for wid, tags in self.way_tags.items():
            if tags.get("type") == "stop_line":
                self.stop_lines.append(wid)
            elif tags.get("type") in ("pedestrian_marking", "zebra_marking"):
                self.crosswalk_ways.append(wid)

        # lanelet geometry, direction, boundary attributes
        self.info = {}
        for lid, L in self.lanelets.items():
            lw, rw = self.ways[L["left"]], self.ways[L["right"]]
            # lanelet2 orders both boundaries along the direction of travel, but
            # be safe: if right is reversed relative to left, flip it
            if np.linalg.norm(lw[0] - rw[0]) > np.linalg.norm(lw[0] - rw[-1]):
                rw = rw[::-1].copy()
                self.ways[L["right"]] = rw
            n = max(8, int(round(poly_len(lw) / STEP_M)) + 1)
            cl = (resample_n(lw, n) + resample_n(rw, n)) / 2.0
            self.info[lid] = {
                "left": lw, "right": rw, "centerline": cl,
                "heading": heading(cl), "length": poly_len(cl),
                "tags": L["tags"],
            }

        if derive_colour:
            self._derive_paint()
        else:
            for lid, i in self.info.items():
                for side in ("left", "right"):
                    i[f"{side}_type"] = ROAD_EDGE
                    i[f"{side}_colour"] = WHITE

        self._chain_topology()

    # -- paint attributes from adjacency + direction --------------------
    def _derive_paint(self):
        for lid, info in self.info.items():
            h = info["heading"]
            for side in ("left", "right"):
                wid = self.lanelets[lid][side]
                tags = self.way_tags.get(wid, {})
                sub = tags.get("subtype")
                users = [u for u in self.way_use.get(wid, []) if u != lid]
                if sub in ("solid", "solid_solid", "dashed"):
                    btype = {"dashed": DASHED, "solid": SOLID,
                             "solid_solid": SOLID_SOLID}[sub]
                    bcol = YELLOW if "yellow" in str(tags.get("color", "")) else WHITE
                elif not users:
                    btype, bcol = ROAD_EDGE, WHITE
                else:
                    other = self.info.get(users[0])
                    opposite = other is not None and ang_diff(h, other["heading"]) > 120.0
                    no_lc = str(self.lanelets[lid]["tags"].get("lane_change", "")).lower() in ("no", "none")
                    btype = SOLID if (opposite or no_lc) else DASHED
                    bcol = YELLOW if opposite else WHITE
                info[f"{side}_type"] = btype
                info[f"{side}_colour"] = bcol

    # -- geometric predecessor/successor chaining -----------------------
    def _chain_topology(self):
        ids = list(self.info)
        for lid in ids:
            i = self.info[lid]
            i["successors"], i["predecessors"] = [], []
        for lid in ids:
            i = self.info[lid]
            end, start, h = i["centerline"][-1], i["centerline"][0], i["heading"]
            for oid in ids:
                if oid == lid:
                    continue
                o = self.info[oid]
                if (np.linalg.norm(o["centerline"][0] - end) <= TOL_M
                        and ang_diff(h, o["heading"]) <= TOL_DEG):
                    i["successors"].append(oid)
                    o["predecessors"].append(lid)

    # -- per-frame export -------------------------------------------------
    def frame(self, ego, yaw, radius=70.0):
        c, s = math.cos(-yaw), math.sin(-yaw)
        ego = np.asarray(ego, float)

        def to_ego(poly):
            d = np.asarray(poly, float) - ego
            return np.stack([d[:, 0] * c - d[:, 1] * s,
                             d[:, 0] * s + d[:, 1] * c], axis=1)

        lanes = []
        for lid, i in self.info.items():
            if not in_radius(i["centerline"], ego, radius):
                continue
            cl = clip(i["centerline"], ego, radius)
            if len(cl) < 2:
                continue
            lanes.append({
                "id": lid,
                "type": i["tags"].get("subtype", "road"),
                "location": i["tags"].get("location"),
                "one_way": i["tags"].get("one_way"),
                "centerline": to_ego(cl).round(3).tolist(),
                "left_boundary": to_ego(clip(i["left"], ego, radius)).round(3).tolist(),
                "right_boundary": to_ego(clip(i["right"], ego, radius)).round(3).tolist(),
                "left_type": int(i["left_type"]), "left_colour": int(i["left_colour"]),
                "right_type": int(i["right_type"]), "right_colour": int(i["right_colour"]),
                "speed_limit_mps": _f(i["tags"].get("speed_limit")),
                "turn": i["tags"].get("turn_direction"),
                "predecessors": i["predecessors"],
                "successors": i["successors"],
            })

        def lines(wids):
            out = []
            for wid in wids:
                w = self.ways.get(wid)
                if w is not None and in_radius(w, ego, radius):
                    out.append({"id": wid, "points": to_ego(w).round(3).tolist()})
            return out

        return {"lanes": lanes,
                "stop_lines": lines(self.stop_lines),
                "crosswalks": lines(self.crosswalk_ways)}


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", action="append", required=True, help="lanelet2 .osm (repeatable)")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--radius", type=float, default=70.0)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    maps = [LaneMap(m) for m in args.map]
    recs = json.loads(pathlib.Path(args.manifest).read_text())
    if args.max_frames:
        recs = recs[: args.max_frames]
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "vector_gt.jsonl", "w") as fh:
        for r in recs:
            best = {"lanes": [], "stop_lines": [], "crosswalks": []}
            for m in maps:
                got = m.frame(r["utm"], r["yaw"], args.radius)
                for k in best:
                    best[k].extend(got[k])
            if not best["lanes"]:
                continue
            seen, kept = set(), []
            for l in best["lanes"]:
                if l["id"] in seen:
                    continue
                seen.add(l["id"])
                c = np.asarray(l["centerline"], float)
                dup = False
                for k in kept:
                    kc = np.asarray(k["centerline"], float)
                    d = np.linalg.norm(c[:, None, :] - kc[None, :, :], axis=2).min(axis=1)
                    if (d < 2.0).mean() > 0.6:      # same physical lane
                        dup = True
                        break
                if not dup:
                    kept.append(l)
            best["lanes"] = kept
            fh.write(json.dumps({"id": r["id"], "session": r["session"],
                                 "utm": r["utm"], "yaw": r["yaw"], **best}) + "\n")

    if args.stats:
        stats(out / "vector_gt.jsonl")


def stats(path):
    rows = [json.loads(l) for l in open(path)]
    if not rows:
        print("no frames exported"); return
    n_lanes = [len(r["lanes"]) for r in rows]
    km = sum(sum(poly_len(l["centerline"]) for l in r["lanes"]) for r in rows) / 1000
    succ = sum(len(l["successors"]) for r in rows for l in r["lanes"])
    lt, rt, col = {}, {}, {}
    for r in rows:
        for l in r["lanes"]:
            lt[l["left_type"]] = lt.get(l["left_type"], 0) + 1
            rt[l["right_type"]] = rt.get(l["right_type"], 0) + 1
            col[(l["left_colour"], l["right_colour"])] = col.get((l["left_colour"], l["right_colour"]), 0) + 1
    names = {DASHED: "dashed", SOLID: "solid", SOLID_SOLID: "solid_solid", ROAD_EDGE: "road_edge"}
    print(f"\nframes exported      {len(rows)}")
    print(f"lanes per frame      mean {np.mean(n_lanes):.1f}  min {min(n_lanes)}  max {max(n_lanes)}")
    print(f"lane length per frame {km * 1000 / len(rows):.0f} m   (total {km:.1f} km)")
    print(f"topology edges       {succ} successors ({succ / max(1, sum(n_lanes)):.2f} per lane)")
    print(f"left boundary types  { {names.get(k, k): v for k, v in sorted(lt.items())} }")
    print(f"right boundary types { {names.get(k, k): v for k, v in sorted(rt.items())} }")
    print(f"colour pairs (L,R)   {col}")
    print(f"stop lines/frame     {np.mean([len(r['stop_lines']) for r in rows]):.2f}")
    print(f"crosswalks/frame     {np.mean([len(r['crosswalks']) for r in rows]):.2f}")


if __name__ == "__main__":
    main()
