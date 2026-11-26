"""Cross-camera fusion and occlusion for the BEV back-projection.

The first version took "whatever camera covers this cell" with equal weight and
no visibility test. That produced the radial wedges and, worse, free space
behind parked cars. Three fixes:

1. **Weighted blending instead of winner-takes-all.**
   Each camera contributes with a weight built from
     * viewing angle  — cos^2 of the angle from the optical axis, so a camera
       sees best what is in front of it and fades at its edges. This alone
       removes the hard wedge seams.
     * distance       — 1/(1 + z/40): a cell 5 m away is resolved by many more
       pixels than the same cell at 40 m.
     * confidence     — 1/(1 + normalised entropy) of the 4-class distribution,
       so an unsure camera does not outvote a sure one.

2. **Occlusion via a LiDAR depth buffer.** Points are projected into each
   camera to build a sparse depth image; a BEV cell is rejected for that camera
   if something is observed along the same ray significantly closer. This is
   what stops the map painting free space behind a vehicle.

3. **Occupancy, so "free" means free.** Per BEV cell, the tallest return above
   the ground. Cells with more than `occ_max` are marked occupied regardless of
   what the cameras say, and cells with no camera coverage are `unknown` rather
   than silently `other`.

The rendered result has five states: drivable / walkway / crossing / occupied /
unknown.
"""
import numpy as np
import torch
import torch.nn.functional as F

from .bev_geom import RES, EXTENT, SIZE, ground_surface, sample_grid
from .classes import CLASSES

STATE_DRIVABLE, STATE_WALKWAY, STATE_CROSSING, STATE_OCCUPIED, STATE_UNKNOWN = 0, 1, 2, 3, 4


def depth_buffer(pts_ego, c2e, K, W, H, scale=4, min_z=1.0):
    """Sparse depth image per camera: min observed depth in each downsampled px."""
    h, w = H // scale, W // scale
    buf = np.full((h, w), np.inf, np.float32)
    P = np.concatenate([pts_ego[:, :3], np.ones((len(pts_ego), 1))], 1)
    pc = (np.linalg.inv(c2e) @ P.T).T
    z = pc[:, 2]
    ok = z > min_z
    if not ok.any():
        return buf, scale
    uv = (K @ (pc[ok, :3] / z[ok, None]).T).T[:, :2]
    u = (uv[:, 0] / scale).astype(np.int32)
    v = (uv[:, 1] / scale).astype(np.int32)
    inb = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    np.minimum.at(buf, (v[inb], u[inb]), z[ok][inb].astype(np.float32))
    return buf, scale


def cell_weights(pts_cam, K, W, H, depth, dscale, octol=0.8):
    """Projection + visibility + per-cell weight for one camera.

    Returns (u, v, z, weight, visible) for the given camera-space points.
    """
    z = pts_cam[:, 2]
    ok = z > 1.0
    uv = (K @ (pts_cam[:, :3] / z[:, None]).T).T[:, :2]
    u, v = uv[:, 0], uv[:, 1]
    inb = ok & (u > 1) & (u < W - 2) & (v > 1) & (v < H - 2)

    # occlusion: is something observed closer along this ray?
    bu = np.clip((u / dscale).astype(np.int32), 0, depth.shape[1] - 1)
    bv = np.clip((v / dscale).astype(np.int32), 0, depth.shape[0] - 1)
    obs = depth[bv, bu]
    occluded = inb & np.isfinite(obs) & (obs < z - octol)
    visible = inb & ~occluded

    f = 0.5 * (K[0, 0] + K[1, 1])
    cx, cy = K[0, 2], K[1, 2]
    r2 = (u - cx) ** 2 + (v - cy) ** 2
    cos2 = (f * f) / (f * f + r2 + 1e-6)          # 1 at the optical axis
    # a camera is unreliable past ~55 deg off-axis: the ground is at grazing
    # incidence and the sampled pixel covers metres of road
    visible = visible & (cos2 > 0.33)
    w = (cos2 ** 2) * (1.0 / (1.0 + z / 40.0))     # cos^4: the best camera wins
    return u, v, z, w, visible


def fuse(probs, c2e, K, W, H, pts_ego_cells, depth_buffers, dscale,
         min_weight=0.02, entropy_scale=1.0, top_k=1):
    """Select the best view per cell instead of averaging across cameras.

    Averaging was the source of the mottling: the same road cell looks like road
    to CAM_FRONT and like verge to CAM_LEFT, and an average of the two is neither.
    The model is not view-invariant, so the honest thing is to trust the camera
    that sees the cell best — by angle, distance and confidence — and ignore the
    rest. `top_k=1` is a hard select; 2 blends the two best.

    Returns (4, N) probs, (N,) weight of the selected view, (N,) ok, (N,) n_views.
    """
    n = len(pts_ego_cells)
    P = np.concatenate([pts_ego_cells, np.ones((n, 1))], 1)
    C = len(probs)
    Wt = np.zeros((C, n), np.float32)
    UV = np.zeros((C, n, 2), np.float32)
    ok_all = np.zeros((C, n), bool)

    for ci, (T, Kc) in enumerate(zip(c2e, K)):
        pc = (np.linalg.inv(T) @ P.T).T
        u, v, z, w, visible = cell_weights(pc, Kc, W, H, depth_buffers[ci], dscale)
        Wt[ci] = np.where(visible, w, 0.0)
        UV[ci, :, 0] = u
        UV[ci, :, 1] = v
        ok_all[ci] = visible

    nviews = ok_all.sum(0)
    out = np.zeros((4, n), np.float32)
    wsel = np.zeros(n, np.float32)

    order = np.argsort(-Wt, axis=0)            # best camera per cell
    for k in range(top_k):
        pick = order[k]                        # (N,) camera index
        valid = Wt[pick, np.arange(n)] > 0
        if not valid.any():
            continue
        idx = np.flatnonzero(valid)
        cam = pick[idx]
        gx = (UV[cam, idx, 0] / W) * 2 - 1
        gy = (UV[cam, idx, 1] / H) * 2 - 1
        grid = torch.from_numpy(np.stack([gx, gy], 1)[None, None].astype(np.float32))
        # gather each cell's chosen camera feature map
        s = np.zeros((4, len(idx)), np.float32)
        for ci in range(C):
            sel = cam == ci
            if not sel.any():
                continue
            pr = probs[ci].unsqueeze(0)
            g = grid[:, :, sel]
            if pr.device != g.device:
                pr = pr.to(g.device)
            sm = F.grid_sample(pr, g, mode="bilinear", align_corners=False,
                               padding_mode="border")[0, :, 0, :].cpu().numpy()
            s[:, sel] = sm
        w = Wt[cam, idx][None, :]
        out[:, idx] += s * w
        wsel[idx] += w[0]

    m = wsel > min_weight
    out[:, m] /= wsel[m][None, :]
    return out, wsel, m, nviews


def occupancy(pts_ego, ground=None, occ_cell=0.5, occ_max=0.50, min_points=3):
    """Vertical extent of the returns in each BEV cell.

    The first version measured height above the *smoothed* ground surface, which
    on a slope is wrong by tens of centimetres and painted the whole road red.
    Using the cell's own spread (max - min z) is independent of the ground model
    and of the datum: flat tarmac spans a few centimetres, a car spans a metre.
    """
    m = np.abs(pts_ego[:, :2]).max(1) < EXTENT
    p = pts_ego[m]
    n = SIZE
    if len(p) == 0:
        return np.zeros((n, n), np.float32)
    c = np.clip(((p[:, 0] + EXTENT) / occ_cell).astype(int), 0, n - 1)
    r = np.clip(((EXTENT - p[:, 1]) / occ_cell).astype(int), 0, n - 1)
    flat = r * n + c
    zmin = np.full(n * n, np.inf, np.float32)
    zmax = np.full(n * n, -np.inf, np.float32)
    cnt = np.zeros(n * n, np.int32)
    np.minimum.at(zmin, flat, p[:, 2].astype(np.float32))
    np.maximum.at(zmax, flat, p[:, 2].astype(np.float32))
    np.add.at(cnt, flat, 1)
    spread = np.where(cnt >= min_points, zmax - zmin, 0.0)
    return spread.reshape(n, n)


def smooth_states(st, keep=(3,)):
    """3x3 majority filter on the state map.

    Cosmetic but honest: the per-cell decision comes from a handful of pixels and
    is noisy; a planner would apply temporal filtering anyway. Observed obstacles
    are never smoothed away.
    """
    counts = np.zeros((5,) + st.shape, np.int32)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            sh = np.roll(np.roll(st, dr, axis=0), dc, axis=1)
            for k in range(5):
                counts[k] += (sh == k)
    out = counts.argmax(0).astype(np.int8)
    for k in keep:
        out[st == k] = k
    return out


def to_states(prob, weight, occ, thr=(0.35, 0.35, 0.45), occ_max=0.50,
              min_cams=2, seen=None, margin=0.12, max_range=40.0):
    """4-class probabilities + occupancy -> a 5-state map for rendering.

    Order matters: an observed obstacle beats any camera prediction, and no
    coverage is `unknown`, never silently "other".
    """
    st = np.full(SIZE * SIZE, STATE_UNKNOWN, np.int8)
    m = weight > 0
    if seen is not None:
        m &= seen >= min_cams
    dr, wa, cr = prob[CLASSES.index("drivable")], prob[CLASSES.index("walkway")], \
        prob[CLASSES.index("crossing")]
    st[m] = STATE_UNKNOWN
    # a class must beat the threshold *and* the runner-up, and at least
    # `min_cams` views must agree, or the cell stays unknown
    order = np.argsort(prob, axis=0)
    top, second = order[-1], order[-2]
    pmax = prob[top, np.arange(prob.shape[1])]
    psec = prob[second, np.arange(prob.shape[1])]
    confident = m & ((pmax - psec) > margin)
    st[confident & (top == CLASSES.index("crossing")) & (cr > thr[2])] = STATE_CROSSING
    st[confident & (top == CLASSES.index("walkway")) & (wa > thr[1])] = STATE_WALKWAY
    st[confident & (top == CLASSES.index("drivable")) & (dr > thr[0])] = STATE_DRIVABLE
    occf = occ.ravel()
    st[occf > occ_max] = STATE_OCCUPIED
    st = st.reshape(SIZE, SIZE)
    # nothing beyond the usable range: the geometry, not the model, runs out
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    rng = np.hypot(xx * RES - EXTENT, EXTENT - yy * RES)
    st[rng > max_range] = STATE_UNKNOWN
    return st
