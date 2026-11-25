#!/usr/bin/env python3
"""Extract lane markings from a VLS-128 point cloud — no training, no map.

Why this exists
---------------
The ring-road / King St recordings have retro-reflective paint that the VLS-128
resolves at ~5-10 cm, while our lanelet2 maps are hand-drawn and 1-3 m off. For
*paint* the LiDAR is the better source of truth and needs no supervised training
on a 500-frame dataset.

What is hard about it
---------------------
Intensity alone is not enough on this data. Worn paint on dark, wet asphalt
returns *weaker* than grass, snow and concrete sidewalk, so a plain "bright and
near the ground" filter is dominated by false positives (verified by projecting
the candidates into the cameras: the road itself came back almost empty while
the verges lit up).

What works is structure. Lane paint is a **long, thin, straight-ish ridge**;
grass, snow, kerbs and sidewalk slabs are **blobs**. So the pipeline votes for
lines instead of thresholding points:

  1. ground model    per 2 m cell 5th-percentile z, hole-filled, median-smoothed
                     (the sensor sits ~2 m above the road and the offset changes
                     per recording, so no global z threshold works)
  2. candidates      0.02-0.25 m above the local ground, brighter than the local
                     asphalt median + --contrast, and locally *planar*
                     (grass/snow roughness is rejected by the dz spread)
  3. Hough vote      (theta, rho) accumulator with 1 deg / 0.10 m bins; a real
                     line collects 8+ collinear returns, a blobby verge does not
  4. segments        for each peak, take the inliers within --inlier m, order
                     them along the line, split at gaps, and simplify
  5. chain           join segments that continue each other (endpoint gap and
                     tangent agreement), which recovers curves as polylines

Output: one .npz per frame with polylines in ego metres plus per-line attributes.
`--png` writes a BEV overlay, `--cams` projects the candidates into the camera
images (the fastest sanity check that a line is really paint).

  lane_from_lidar.py FRAME.npz --out lanes/ --png --cams
"""
import argparse
import io
import json
import math
import pathlib

import numpy as np
from PIL import Image, ImageDraw

RES = 0.10
EXTENT = 60.0
SIZE = int(2 * EXTENT / RES)


# --------------------------------------------------------------------- ground
def ground_model(xyz, cell=2.0, extent=EXTENT):
    n = int(2 * extent / cell)
    c = np.clip(((xyz[:, 0] + extent) / cell).astype(int), 0, n - 1)
    r = np.clip(((extent - xyz[:, 1]) / cell).astype(int), 0, n - 1)
    flat = r * n + c
    order = np.argsort(flat, kind="stable")
    f_sorted, z_sorted = flat[order], xyz[order, 2]
    starts = np.searchsorted(f_sorted, np.arange(n * n), side="left")
    ends = np.searchsorted(f_sorted, np.arange(n * n), side="right")
    g = np.full(n * n, np.nan, np.float32)
    for i in range(n * n):
        if ends[i] > starts[i]:
            g[i] = np.percentile(z_sorted[starts[i]:ends[i]], 5)
    g = g.reshape(n, n)
    for _ in range(8):
        bad = ~np.isfinite(g)
        if not bad.any():
            break
        pad = np.pad(g, 1, constant_values=np.nan)
        stack = np.stack([pad[1 + dr:n + 1 + dr, 1 + dc:n + 1 + dc]
                          for dr in (-1, 0, 1) for dc in (-1, 0, 1)])
        with np.errstate(all="ignore"):
            fill = np.nanmedian(stack, axis=0)
        g = np.where(bad, fill, g)
    if not np.isfinite(g).all():
        g = np.where(np.isfinite(g), g, np.nanmedian(g))
    pad = np.pad(g, 2, mode="edge")
    g = np.median(np.lib.stride_tricks.sliding_window_view(pad, (5, 5)), axis=(2, 3))
    return g, cell, extent


def sample_grid(model, x, y):
    g, cell, extent = model
    n = g.shape[0]
    c = np.clip(((x + extent) / cell).astype(int), 0, n - 1)
    r = np.clip(((extent - y) / cell).astype(int), 0, n - 1)
    return g[r, c]


# ------------------------------------------------------------------ candidates
def _local_median(vals, keys, nbin, min_count=15, fallback=None):
    med = np.full(nbin, np.nan, np.float32)
    order = np.argsort(keys, kind="stable")
    ks, vs = keys[order], vals[order]
    starts = np.searchsorted(ks, np.arange(nbin), side="left")
    ends = np.searchsorted(ks, np.arange(nbin), side="right")
    for i in np.flatnonzero(ends > starts):
        if ends[i] - starts[i] >= min_count:
            med[i] = np.median(vs[starts[i]:ends[i]])
    fill = np.nanmedian(vals) if fallback is None else fallback
    return np.where(np.isfinite(med), med, fill)


def candidates(points, dz_lo=0.02, dz_hi=0.25, contrast=6.0, rmax=EXTENT,
               rough_max=0.045, contrast_range=6.0):
    p = points[np.isfinite(points[:, :3]).all(1)]
    p = p[np.hypot(p[:, 0], p[:, 1]) < rmax]
    if len(p) < 500:
        return np.zeros((0, 5), np.float32)
    dz = p[:, 2] - sample_grid(ground_model(p), p[:, 0], p[:, 1])
    band = (dz > dz_lo) & (dz < dz_hi)
    if band.sum() < 200:
        return np.zeros((0, 5), np.float32)

    q = p[band]
    dzq = dz[band]

    # asphalt median per 3 m cell (intensity contrast)
    cell = 3.0
    n = int(2 * rmax / cell)
    c = np.clip(((q[:, 0] + rmax) / cell).astype(int), 0, n - 1)
    r = np.clip(((rmax - q[:, 1]) / cell).astype(int), 0, n - 1)
    local = _local_median(q[:, 3], r * n + c, n * n)[(r * n + c)]

    # per-range baseline: within one beam ring the asphalt intensity is fairly
    # constant, so the median over all azimuths at that range is the baseline
    # and painted stripes are the bumps. This is what kills the ring pattern.
    rng = np.hypot(q[:, 0], q[:, 1])
    nrb = int(rmax / 0.5) + 1
    rb = np.clip((rng / 0.5).astype(int), 0, nrb - 1)
    base = _local_median(q[:, 3], rb, nrb, min_count=30)
    bright = (q[:, 3] > local + contrast) & (q[:, 3] > base[rb] + contrast_range)

    # local roughness of the ground surface: grass, snow, gravel and kerbs are
    # rough within a 2 m cell, a road surface (painted or not) is flat
    rc = 2.0
    nr = int(2 * rmax / rc)
    dz_all = dz[(np.abs(dz) < 0.6)]
    pall = p[(np.abs(dz) < 0.6)]
    rc2 = np.clip(((pall[:, 0] + rmax) / rc).astype(int), 0, nr - 1)
    rr2 = np.clip(((rmax - pall[:, 1]) / rc).astype(int), 0, nr - 1)
    key = rr2 * nr + rc2
    cnt = np.bincount(key, minlength=nr * nr).astype(np.float64)
    s1 = np.bincount(key, weights=dz_all, minlength=nr * nr)
    s2 = np.bincount(key, weights=dz_all ** 2, minlength=nr * nr)
    with np.errstate(all="ignore"):
        mean = np.where(cnt > 5, s1 / cnt, 0.0)
        var = np.where(cnt > 5, s2 / cnt - mean ** 2, 1.0)
    rough = np.sqrt(np.maximum(var, 0.0))
    qc = np.clip(((q[:, 0] + rmax) / rc).astype(int), 0, nr - 1)
    qr = np.clip(((rmax - q[:, 1]) / rc).astype(int), 0, nr - 1)
    flat = rough[qr * nr + qc] < rough_max

    keep = bright & flat
    out = np.concatenate([q[keep][:, :4], dzq[keep, None]], 1)
    return out


# --------------------------------------------------------------------- hough
def hough(xy, theta_step=1.0, rho_step=0.10, rmax=None):
    if len(xy) == 0:
        return np.zeros((0, 3))
    rmax = (rmax or float(np.abs(xy).max())) * math.sqrt(2.0)
    thetas = np.deg2rad(np.arange(0, 180, theta_step))
    cos, sin = np.cos(thetas), np.sin(thetas)
    nrho = int(2 * rmax / rho_step) + 3
    acc = np.zeros((len(thetas), nrho), np.int32)
    for i in range(len(thetas)):
        rho = xy[:, 0] * cos[i] + xy[:, 1] * sin[i]
        idx = np.clip(((rho + rmax) / rho_step).astype(int), 0, nrho - 1)
        acc[i] = np.bincount(idx, minlength=nrho)[:nrho]
    return acc, thetas, rho_step, rmax


def _peaks(acc, thetas, rho_step, rmax, min_votes, nms_theta=4, nms_rho=6):
    order = np.argsort(acc, axis=None)[::-1]
    taken = np.zeros_like(acc, bool)
    out = []
    for flat in order:
        ti, ri = np.unravel_index(flat, acc.shape)
        if acc[ti, ri] < min_votes or taken[ti, ri]:
            continue
        out.append((int(acc[ti, ri]), float(thetas[ti]), (ri * rho_step) - rmax))
        t0, t1 = max(0, ti - nms_theta), min(acc.shape[0], ti + nms_theta + 1)
        r0, r1 = max(0, ri - nms_rho), min(acc.shape[1], ri + nms_rho + 1)
        taken[t0:t1, r0:r1] = True
        if len(out) >= 400:
            break
    return out


def _split_gaps(chain, max_gap):
    runs, cur = [], [chain[0]]
    for a, b in zip(chain, chain[1:]):
        if np.linalg.norm(b - a) > max_gap:
            runs.append(np.array(cur))
            cur = [b]
        else:
            cur.append(b)
    runs.append(np.array(cur))
    return [r for r in runs if len(r) >= 2]


def _isolation(xy, run, inlier, radius=0.7):
    """Fraction of nearby candidates that actually lie on the line.

    A painted stripe is surrounded by dark asphalt, so almost every candidate
    within 0.7 m of it is an inlier. Grass, snow and sidewalk slabs are dense
    blobs, so the ratio collapses. This is the filter that separates the two.
    """
    mid = run[:: max(1, len(run) // 60)]
    d = np.hypot(xy[:, None, 0] - mid[None, :, 0], xy[:, None, 1] - mid[None, :, 1])
    near = (d < radius).any(1)
    if near.sum() == 0:
        return 1.0
    dperp = np.min(np.abs(xy[near, 0][:, None] * 0 + 0), axis=1) if False else None
    on = run
    dd = np.hypot(xy[near][:, None, 0] - on[None, :, 0], xy[near][:, None, 1] - on[None, :, 1])
    on_line = dd.min(1) < inlier
    return float(on_line.mean())


def segments_from_peaks(cand, peaks, inlier=0.20, min_len=5.0, max_gap=1.2,
                        min_isolation=0.55):
    xy = cand[:, :2]
    segs = []
    for votes, th, rho in peaks:
        d = np.abs(xy[:, 0] * math.cos(th) + xy[:, 1] * math.sin(th) - rho)
        sel = d < inlier
        if sel.sum() < 6:
            continue
        p = xy[sel]
        # order along the line direction
        u = np.array([-math.sin(th), math.cos(th)])
        t = p @ u
        p = p[np.argsort(t)]
        for run in _split_gaps(p, max_gap):
            seg = np.linalg.norm(np.diff(run, axis=0), axis=1).sum()
            if seg < min_len:
                continue
            iso = _isolation(xy, run, inlier)
            if iso < min_isolation:
                continue
            segs.append({"xy": run, "votes": int(votes), "theta": th,
                         "rho": rho, "isolation": round(iso, 3)})
    segs.sort(key=lambda s: -s["votes"])
    # drop duplicates: same line found at a slightly different rho
    kept = []
    for s in segs:
        dup = False
        for k in kept:
            if abs(s["rho"] - k["rho"]) < 0.30 and abs(
                    math.degrees(s["theta"] - k["theta"])) < 4.0:
                a = np.array([np.mean(k["xy"][:, 0]), np.mean(k["xy"][:, 1])])
                b = np.array([np.mean(s["xy"][:, 0]), np.mean(s["xy"][:, 1])])
                if np.linalg.norm(a - b) < 8.0:
                    dup = True
                    break
        if not dup:
            kept.append(s)
    return kept


# --------------------------------------------------------------- chaining
def _ends(chain):
    a, b = chain[0], chain[-1]
    v = b - a
    L = np.linalg.norm(v)
    return a, b, (v / L if L > 1e-9 else np.array([1.0, 0.0]))


def chain_segments(segs, gap=4.0, ang=28.0, lat=0.7):
    """Join segments that continue each other — recovers curves as polylines."""
    changed = True
    while changed:
        changed = False
        for i in range(len(segs)):
            for j in range(i + 1, len(segs)):
                A, B = segs[i]["xy"], segs[j]["xy"]
                best = None
                for pa in (0, 1):
                    for pb in (0, 1):
                        ea = A[0] if pa == 0 else A[-1]
                        eb = B[0] if pb == 0 else B[-1]
                        g = np.linalg.norm(ea - eb)
                        if g > gap:
                            continue
                        _, _, da = _ends(A)
                        _, _, db = _ends(B)
                        v = eb - ea
                        L = np.linalg.norm(v)
                        if L < 1e-6:
                            continue
                        v = v / L
                        a1 = math.degrees(math.acos(min(1.0, abs(float(v @ da)))))
                        a2 = math.degrees(math.acos(min(1.0, abs(float(v @ db)))))
                        if a1 < ang and a2 < ang and L * math.sin(math.radians(a1)) < lat:
                            if best is None or g < best[0]:
                                best = (g, pa, pb)
                if best is None:
                    continue
                _, pa, pb = best
                left = A if pa == 0 else A[::-1]
                right = B if pb == 1 else B[::-1]
                segs[i]["xy"] = np.concatenate([left, right])
                segs[i]["votes"] = max(segs[i]["votes"], segs[j]["votes"])
                segs.pop(j)
                changed = True
                break
            if changed:
                break
    return segs


def simplify(xy, eps=0.12):
    if len(xy) < 3:
        return xy
    keep = np.zeros(len(xy), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(xy) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        seg = xy[b] - xy[a]
        L = np.linalg.norm(seg)
        if L < 1e-9:
            d = np.linalg.norm(xy[a + 1:b] - xy[a], axis=1)
        else:
            d = np.abs(np.cross(seg, xy[a + 1:b] - xy[a])) / L
        i = int(np.argmax(d))
        if d[i] > eps:
            keep[a + 1 + i] = True
            stack += [(a, a + 1 + i), (a + 1 + i, b)]
    return xy[keep]


def extract(points, min_length=5.0, min_votes=8):
    cand = candidates(points)
    if len(cand) < 40:
        return [], cand
    acc, thetas, rho_step, rmax = hough(cand[:, :2])
    peaks = _peaks(acc, thetas, rho_step, rmax, min_votes)
    segs = segments_from_peaks(cand, peaks, min_len=min_length * 0.6)
    if not segs:
        return [], cand
    segs = chain_segments(segs)
    lines = []
    for s in segs:
        xy = s["xy"]
        length = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
        if length < min_length:
            continue
        d = xy - xy.mean(0)
        _, vecs = np.linalg.eigh(d.T @ d)
        width = float(np.abs(d @ vecs[:, 0]).max() * 2)
        # support along the line: paint is continuous, a blobby verge is not
        runs = _split_gaps(xy, 0.9)
        cover = float(length / (runs[0][0] @ np.array([1, 0]) * 0 + 1e-9)) if False else None
        gaps = max(0, len(runs) - 1)
        poly = simplify(xy, eps=0.12)
        lines.append({
            "points": np.round(poly, 3).tolist(),
            "length_m": round(length, 2),
            "width_m": round(width, 3),
            "votes": s["votes"],
            "n_points": int(len(xy)),
            "style": "solid" if gaps == 0 else "dashed",
            "n_gaps": gaps,
            "isolation": s.get("isolation", 1.0),
        })
    lines.sort(key=lambda d: -d["length_m"])
    return lines, cand


# ------------------------------------------------------------------- rendering
def render_bev(points, lines, cand, path, title=""):
    img = Image.new("RGB", (1000, 1000), (12, 12, 14))
    dr = ImageDraw.Draw(img)

    def px(x, y):
        return ((x + EXTENT) / (2 * EXTENT) * 1000, (EXTENT - y) / (2 * EXTENT) * 1000)

    for g in range(-50, 51, 10):
        dr.line([px(g, -55), px(g, 55)], fill=(28, 28, 32))
        dr.line([px(-55, g), px(55, g)], fill=(28, 28, 32))
    for x, y in points[:, :2]:
        if abs(x) < EXTENT and abs(y) < EXTENT:
            dr.point(px(x, y), fill=(45, 45, 50))
    for x, y in cand[:, :2]:
        dr.point(px(x, y), fill=(120, 120, 130))
    for l in lines:
        col = (0, 240, 120) if l["style"] == "solid" else (255, 170, 0)
        dr.line([px(*q) for q in l["points"]], fill=col, width=3)
    dr.line([px(0, 0), px(0, 25)], fill=(255, 70, 70), width=3)
    dr.line([px(0, 0), px(12, 0)], fill=(70, 130, 255), width=3)
    if title:
        dr.text((8, 8), title, fill=(200, 200, 200))
    img.save(path)


def render_cams(frame, cand, cal, prefix):
    if len(cand) == 0:
        return
    l2e = cal["lidar2ego"].astype(np.float64)
    ph = np.concatenate([cand[:, :3], np.ones((len(cand), 1))], 1)
    pe = (l2e @ ph.T).T
    for i, name in enumerate(cal["names"]):
        c2e = cal["camera2ego"][i].astype(np.float64)
        K = cal["K"][i].astype(np.float64)
        pc = (np.linalg.inv(c2e) @ pe.T).T
        ok = pc[:, 2] > 1.0
        if not ok.any():
            continue
        uv = (K @ (pc[ok][:, :3] / pc[ok, 2:3]).T).T[:, :2]
        im = Image.open(io.BytesIO(bytes(frame["jpegs"][i]))).convert("RGB")
        W, H = im.size
        dr = ImageDraw.Draw(im)
        for u, v in uv:
            if 0 <= u < W and 0 <= v < H:
                dr.ellipse([u - 2, v - 2, u + 2, v + 2], fill=(255, 0, 255))
        im.save(f"{prefix}_{name}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--png", action="store_true")
    ap.add_argument("--cams", action="store_true")
    ap.add_argument("--min-length", type=float, default=5.0)
    ap.add_argument("--min-votes", type=int, default=8)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    total = 0
    for f in args.frames:
        d = np.load(f, allow_pickle=True)
        pts = d["points"].astype(np.float32)
        lines, cand = extract(pts, min_length=args.min_length,
                              min_votes=args.min_votes)
        tag = f"{pathlib.Path(f).parent.name}__{pathlib.Path(f).stem}"
        np.savez_compressed(
            out / f"{tag}.npz",
            lines=np.array(json.dumps(lines)),
            candidates=cand.astype(np.float32),
            utm=d["utm"] if "utm" in d else np.zeros(3),
        )
        total += len(lines)
        solid = sum(1 for l in lines if l["style"] == "solid")
        print(f"{tag}: {len(lines):2d} lines ({solid} solid), "
              f"{len(cand):5d} candidates", flush=True)
        if args.png:
            render_bev(pts, lines, cand, out / f"{tag}.png", tag)
        if args.cams:
            cal = np.load(pathlib.Path(f).parent / "calib.npz", allow_pickle=True)
            render_cams(d, cand, cal, str(out / tag))
    print(f"\n{len(args.frames)} frames, {total} lines -> {out}")


if __name__ == "__main__":
    main()
