"""Shared BEV geometry: ground surface, sampling, ego<->world.

These were duplicated in three scripts before; they are the part most likely to
be subtly wrong (grid convention, hole filling), so they live in one place now.

Grid convention everywhere: 0.5 m cells over [-50, 50] m, rows +y, cols +x,
rotated into the ego heading — the same one the BEV map head uses.
"""
import math

import numpy as np

RES, EXTENT = 0.5, 50.0
SIZE = int(2 * EXTENT / RES)


def ground_surface(pts_ego, cell=2.0, extent=60.0):
    """Per-cell 10th-percentile z, hole-filled and median-smoothed.

    The LiDAR sits ~2 m up and the ground height varies per recording, so a
    global threshold does not work; this gives z(x, y) in the ego frame.
    """
    n = int(2 * extent / cell)
    c = np.clip(((pts_ego[:, 0] + extent) / cell).astype(int), 0, n - 1)
    r = np.clip(((extent - pts_ego[:, 1]) / cell).astype(int), 0, n - 1)
    flat = r * n + c
    order = np.argsort(flat, kind="stable")
    fs, zs = flat[order], pts_ego[order, 2]
    starts = np.searchsorted(fs, np.arange(n * n), side="left")
    ends = np.searchsorted(fs, np.arange(n * n), side="right")
    g = np.full(n * n, np.nan, np.float32)
    for i in np.flatnonzero(ends > starts):
        g[i] = np.percentile(zs[starts[i]:ends[i]], 10)
    g = g.reshape(n, n)
    for _ in range(8):
        bad = ~np.isfinite(g)
        if not bad.any():
            break
        pad = np.pad(g, 1, constant_values=np.nan)
        st = np.stack([pad[1 + dr:n + 1 + dr, 1 + dc:n + 1 + dc]
                       for dr in (-1, 0, 1) for dc in (-1, 0, 1)])
        with np.errstate(all="ignore"):
            fill = np.nanmedian(st, axis=0)
        g = np.where(bad, fill, g)
    if not np.isfinite(g).all():
        g = np.where(np.isfinite(g), g, np.nanmedian(g))
    pad = np.pad(g, 2, mode="edge")
    g = np.median(np.lib.stride_tricks.sliding_window_view(pad, (5, 5)), axis=(2, 3))
    return g, cell, extent


def sample_grid(grid, x, y):
    g, cell, extent = grid
    n = g.shape[0]
    c = np.clip(((x + extent) / cell).astype(int), 0, n - 1)
    r = np.clip(((extent - y) / cell).astype(int), 0, n - 1)
    return g[r, c]


def bev_cells(extent=EXTENT, res=RES):
    """Cell centres of the BEV window in ego metres."""
    n = int(2 * extent / res)
    rows, cols = np.mgrid[0:n, 0:n]
    x = (cols * res - extent).ravel()
    y = (extent - rows * res).ravel()
    return x, y, n


def ego_to_world(pts_ego, utm, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    x = pts_ego[:, 0] * c - pts_ego[:, 1] * s + utm[0]
    y = pts_ego[:, 0] * s + pts_ego[:, 1] * c + utm[1]
    return x, y


def world_to_ego(x, y, utm, yaw):
    dx, dy = x - utm[0], y - utm[1]
    c, s = math.cos(-yaw), math.sin(-yaw)
    return dx * c - dy * s, dx * s + dy * c
