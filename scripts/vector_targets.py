"""Pair our frames with the exported lanelet vector GT.

Targets come from export_vector_gt.py (per frame: lanes with clipped centreline
polylines, lane type, boundary attributes, topology). A vector head needs fixed
size targets, so each centreline is resampled to K ordered points in ego metres.

Classes match what the planner distinguishes, plus a background slot:
    0 road   1 crosswalk   2 other   3 background
"""
import json
import pathlib

import numpy as np

CLASS_NAMES = ["road", "crosswalk", "other"]
BACKGROUND = len(CLASS_NAMES)


def resample_polyline(pts, k):
    p = np.asarray(pts, dtype=float)
    if len(p) < 2:
        return np.repeat(p[:1], k, axis=0) if len(p) else np.zeros((k, 2))
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-6:
        return np.repeat(p[:1], k, axis=0)
    t = np.linspace(0.0, s[-1], k)
    return np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], axis=1)


class VectorTargets:
    def __init__(self, jsonl, k=20, extent=50.0):
        self.k = k
        self.limit = extent - 0.5      # the head predicts inside the BEV grid
        rows = [json.loads(l) for l in open(jsonl)]
        self.by_id = {r["id"]: r for r in rows}

    def __len__(self):
        return len(self.by_id)

    def get(self, frame_id):
        """-> (points (G,K,2) float32, labels (G,) int64, session, coverage of lanes)"""
        r = self.by_id.get(frame_id)
        if r is None:
            return np.zeros((0, self.k, 2), np.float32), np.zeros(0, np.int64), None, 0
        pts, labs = [], []
        for lane in r["lanes"]:
            c = lane["centerline"]
            if len(c) < 2:
                continue
            p = resample_polyline(c, self.k)
            if np.abs(p).max() > self.limit + 8.0:   # lane lives outside the grid
                continue
            pts.append(np.clip(p, -self.limit, self.limit))
            t = lane.get("type") or "other"
            labs.append(CLASS_NAMES.index(t) if t in CLASS_NAMES else 2)
        if not pts:
            return np.zeros((0, self.k, 2), np.float32), np.zeros(0, np.int64), r["session"], 0
        return (np.stack(pts).astype(np.float32), np.asarray(labs, np.int64),
                r["session"], len(pts))
