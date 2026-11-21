#!/usr/bin/env python3
"""Stream a huge Google-Drive bag over HTTP ranges and keep only extracted frames.

The home quota (~16 GB) cannot hold a 65 GB bag, and full GETs of this file are
blocked by Drive's download quota anyway -- but Range requests are served. So read
the mcap sequentially through a range-backed file object, and write out only what
we need: frames sampled by *distance travelled*, so the set spans the whole drive
instead of the stationary stretch at the start.

Frame selection:
  * ego motion from /novatel/oem7/odom (poses are already UTM)
  * emit a frame every --spacing metres of travel
  * only when speed > --min-speed, so parked/staging segments are skipped
  * cameras matched to the lidar stamp within --tol-ms

  stream_extract.py --id FILE_ID --out /bags/ring_frames --spacing 20 --min-speed 2
  stream_extract.py --local /bags/ring_road_30s --out /tmp/test --spacing 5   # test
"""
import argparse
import concurrent.futures as cf
import json
import subprocess
import io
import math
import pathlib
import re
import sys
import threading
import time
import urllib.request

import numpy as np

CAMS = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
        "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
PANOS = {
    "/camera_pano_nn/image_rect_compressed": 0,
    "/camera_pano_ne/image_rect_compressed": 1,
    "/camera_pano_nw/image_rect_compressed": 2,
    "/camera_pano_ss/image_rect_compressed": 3,
    "/camera_pano_sw/image_rect_compressed": 4,
    "/camera_pano_se/image_rect_compressed": 5,
}
INFO = {k.replace("image_rect_compressed", "camera_info"): v for k, v in PANOS.items()}
LIDAR, ODOM = "/lidar_cc/velodyne_points", "/novatel/oem7/odom"
FRAME_TO_SLOT = {"camera_pano_nn": 0, "camera_pano_ne": 1, "camera_pano_nw": 2,
                 "camera_pano_ss": 3, "camera_pano_sw": 4, "camera_pano_se": 5,
                 "lidar_cc": "lidar"}


# ------------------------------------------------------------- range-backed file
COOKIE_HEADER = ""       # set by --cookies


def load_cookies(path):
    """Netscape cookies.txt -> a single Cookie header value."""
    pairs = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 7:
                continue
            domain, _flag, _path, _secure, _exp, name, value = parts[:7]
            if "google.com" in domain:
                pairs.append(f"{name}={value}")
    return "; ".join(pairs)


def _headers(extra=None):
    h = dict(extra or {})
    if COOKIE_HEADER:
        h["Cookie"] = COOKIE_HEADER
    return h


def resolve_url(file_id):
    page = f"https://drive.google.com/uc?export=download&id={file_id}"
    req = urllib.request.Request(page, headers=_headers())
    with urllib.request.urlopen(req, timeout=60) as r:
        html = r.read().decode("utf-8", "replace")
    m = re.search(r'uuid" value="([^"]+)"', html)
    uuid = m.group(1) if m else ""
    return (f"https://drive.usercontent.google.com/download?id={file_id}"
            f"&export=download&confirm=t&uuid={uuid}")


class HttpRangeFile(io.RawIOBase):
    """Read-only seekable file over HTTP ranges with parallel prefetch.

    Drive throttles a single connection to ~4 MiB/s but scales ~8x with 8 of
    them, so blocks are fetched by a small thread pool slightly ahead of the
    read cursor. Tokens are refreshed under a lock when a fetch fails.
    """

    def __init__(self, file_id, block=16 << 20, workers=8, lookahead=12):
        # several byte-identical copies can be given; Drive's quota is per file
        # *and* per byte region, so failing over rides around throttled blocks
        self.file_ids = file_id if isinstance(file_id, (list, tuple)) else [file_id]
        self._rr = 0
        self.file_id = self.file_ids[0]
        self._lock = threading.Lock()
        self.url = resolve_url(self.file_id)
        self.block = block
        self.workers = workers
        self.lookahead = lookahead
        self.pos = 0
        self._cur = b""
        self._cur_start = 0
        self.size = self._probe()
        self._pool = cf.ThreadPoolExecutor(max_workers=workers)
        self._futures = {}
        self._fetches = 0
        self._prefetch(self.pos)

    def _fetch(self, start, end):
        for attempt in range(12):
            with self._lock:
                url = self.url
            try:
                req = urllib.request.Request(
                    url, headers=_headers({"Range": f"bytes={start}-{end}"}))
                with urllib.request.urlopen(req, timeout=180) as r:
                    data = r.read()
                if len(data) != end - start + 1:
                    raise IOError(f"short block {len(data)} != {end-start+1}")
                return data
            except Exception as exc:  # noqa: BLE001
                with self._lock:                      # rotate to the next copy
                    self._rr = (self._rr + 1) % len(self.file_ids)
                    self.file_id = self.file_ids[self._rr]
                    try:
                        self.url = resolve_url(self.file_id)
                    except Exception:                 # keep the old url
                        pass
                print(f"  [range] {exc}; -> file {self._rr} ({self.file_id[:8]})",
                      file=sys.stderr)
                time.sleep(0.5 + 0.5 * attempt)
        raise RuntimeError(f"range fetch failed at {start}")

    def _prefetch(self, pos):
        first = (pos // self.block) * self.block
        for i in range(self.lookahead):
            off = first + i * self.block
            if off >= self.size or off in self._futures:
                continue
            end = min(off + self.block, self.size) - 1
            self._futures[off] = self._pool.submit(self._fetch, off, end)

    def _block(self, off):
        f = self._futures.pop(off, None)
        if f is None:
            end = min(off + self.block, self.size) - 1
            f = self._pool.submit(self._fetch, off, end)
        return f.result()

    def _probe(self):
        req = urllib.request.Request(self.url, headers=_headers({"Range": "bytes=0-0"}))
        with urllib.request.urlopen(req, timeout=120) as r:
            cr = r.headers.get("Content-Range", "")
        return int(cr.split("/")[-1])

    def close(self):
        """Ignored: mcap closes the stream it was handed."""
        pass

    @property
    def closed(self):
        return False

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        new = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        if (new // self.block) != (self.pos // self.block):
            self._prefetch(new)
        self.pos = new
        return self.pos

    def readinto(self, b):
        n = len(b)
        if self.pos >= self.size or n == 0:
            return 0
        n = min(n, self.size - self.pos)
        filled = 0
        while filled < n:
            start = (self.pos // self.block) * self.block
            if start != self._cur_start or not self._cur:
                self._cur = self._block(start)
                self._cur_start = start
                self._fetches += 1
                self._prefetch(self.pos)      # keep the pipeline full
                if self._fetches % 40 == 0:
                    print(f"  [range] {self.pos / 2**30:.2f}/"
                          f"{self.size / 2**30:.2f} GiB", file=sys.stderr)
            off = self.pos - self._cur_start
            take = min(len(self._cur) - off, n - filled)
            b[filled:filled + take] = self._cur[off:off + take]
            self.pos += take
            filled += take
        return filled



class RcloneFile(io.RawIOBase):
    """Read-only seekable file over an rclone remote, using `rclone cat` ranges.

    Avoids transferring credentials to the cluster: the OAuth refresh token lives
    in ~/.config/rclone/rclone.conf and rclone does the authenticated reads. Each
    block costs one `rclone cat --offset --count` process; blocks are prefetched
    by a small thread pool (Drive's API rate-limits harder than plain HTTP, so
    keep workers modest).
    """

    def __init__(self, remote, block=32 << 20, workers=8, lookahead=10):
        self.remote = remote
        self.block = block
        self.workers = workers
        self.lookahead = lookahead
        self.pos = 0
        self._cur = b""
        self._cur_start = 0
        self.size = self._size()
        self._pool = cf.ThreadPoolExecutor(max_workers=workers)
        self._futures = {}
        self._prefetch(0)

    def _size(self):
        out = subprocess.check_output(["rclone", "size", "--json", self.remote],
                                      stderr=subprocess.STDOUT)
        return int(json.loads(out)["bytes"])

    def _fetch(self, off, n):
        for attempt in range(5):
            try:
                out = subprocess.run(
                    ["rclone", "cat", "--offset", str(off), "--count", str(n), self.remote],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if out.returncode == 0 and len(out.stdout) == n:
                    return out.stdout
                print(f"  [rclone] block {off} rc={out.returncode} "
                      f"got {len(out.stdout)}/{n}: "
                      f"{out.stderr.decode('utf-8', 'replace')[:120]}", file=sys.stderr)
            except Exception as exc:  # noqa: BLE001
                print(f"  [rclone] {exc}", file=sys.stderr)
            time.sleep(2 + 2 * attempt)
        raise RuntimeError(f"rclone block failed at {off}")

    def _prefetch(self, pos):
        first = (pos // self.block) * self.block
        for i in range(self.lookahead):
            off = first + i * self.block
            if off >= self.size or off in self._futures:
                continue
            self._futures[off] = self._pool.submit(
                self._fetch, off, min(self.block, self.size - off))

    def _block(self, off):
        f = self._futures.pop(off, None)
        if f is None:
            f = self._pool.submit(self._fetch, off, min(self.block, self.size - off))
        return f.result()

    def close(self):
        pass

    @property
    def closed(self):
        return False

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        new = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        self._prefetch(new)
        self.pos = new
        return self.pos

    def readinto(self, b):
        n = len(b)
        if self.pos >= self.size or n == 0:
            return 0
        n = min(n, self.size - self.pos)
        filled = 0
        while filled < n:
            start = (self.pos // self.block) * self.block
            if start != self._cur_start or not self._cur:
                self._cur = self._block(start)
                self._cur_start = start
                self._prefetch(self.pos)
                print(f"  [rclone] {self.pos / 2**30:.2f}/{self.size / 2**30:.2f} GiB",
                      file=sys.stderr)
            off = self.pos - self._cur_start
            take = min(len(self._cur) - off, n - filled)
            b[filled:filled + take] = self._cur[off:off + take]
            self.pos += take
            filled += take
        return filled


# ------------------------------------------------------------------ extraction
def cloud_to_numpy(msg):
    ps, n = msg.point_step, msg.height * msg.width
    raw = np.frombuffer(msg.data, dtype=np.uint8, count=n * ps).reshape(n, ps)
    offs = {f.name: f.offset for f in msg.fields}
    out = np.zeros((n, 5), dtype=np.float32)
    for k, name in enumerate(("x", "y", "z", "intensity")):
        if name in offs:
            o = offs[name]
            out[:, k] = raw[:, o:o + 4].copy().view("<f4").ravel()
    return out[np.isfinite(out).all(axis=1)]


def quat_to_mat(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", action="append", default=[], help="Drive file id (repeatable: byte-identical copies for quota failover)")
    ap.add_argument("--local", default=None, help="local bag dir or .mcap (testing)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--spacing", type=float, default=20.0, help="metres between frames")
    ap.add_argument("--min-speed", type=float, default=2.0, help="m/s")
    ap.add_argument("--tol-ms", type=float, default=200.0)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--max-gib", type=float, default=0.0, help="stop after reading this much")
    ap.add_argument("--cookies", default=None,
                    help="Netscape cookies.txt for authenticated (owner-quota) download")
    ap.add_argument("--rclone", default=None,
                    help="rclone remote path, e.g. gdrive:ring_road/ring_road_corrected_0.mcap "
                         "(uses the server-side OAuth token; no credentials needed here)")
    ap.add_argument("--block-mb", type=int, default=32)
    ap.add_argument("--rclone-workers", type=int, default=12,
                    help="parallel rclone streams; Drive throttles each one hard")
    args = ap.parse_args()

    from mcap_ros2.reader import read_ros2_messages

    if args.cookies:
        global COOKIE_HEADER
        COOKIE_HEADER = load_cookies(args.cookies)
        print(f"cookies: {len(COOKIE_HEADER.split('; '))} entries from {args.cookies}")

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.rclone:
        stream = RcloneFile(args.rclone, block=args.block_mb << 20,
                            workers=args.rclone_workers)
        print(f"rclone stream {stream.size / 2**30:.2f} GiB from {args.rclone}")
    elif args.local:
        bag = pathlib.Path(args.local)
        mcap = next(bag.glob("*.mcap")) if bag.is_dir() else bag
        stream = open(mcap, "rb")
    else:
        stream = HttpRangeFile(args.id if len(args.id) > 1 else args.id[0])
        print(f"streaming {stream.size / 2**30:.2f} GiB from Drive")

    K = np.zeros((6, 3, 3))
    got_info = np.zeros(6, bool)
    cam2ego = np.zeros((6, 4, 4))
    lidar2ego = np.zeros((4, 4))
    edges = {}

    cam_buf = {i: [] for i in range(6)}
    dbg = {"pending": 0, "lidar": 0, "emitted": 0, "cam_tol": 0, "nocams": 0}
    want_emit = False
    last_pose = None
    travelled = 0.0
    next_emit = 0.0
    written = 0
    t_start = time.time()
    total_msgs = 0

    for m in read_ros2_messages(stream):
        topic, r = m.channel.topic, m.ros_msg
        total_msgs += 1
        h = getattr(r, "header", None)
        stamp = (h.stamp.sec * 10**9 + h.stamp.nanosec) if h is not None else 0

        if topic in PANOS:
            buf = cam_buf[PANOS[topic]]
            buf.append((stamp, bytes(r.data)))
            if len(buf) > 40:
                del buf[:-20]
        elif topic in INFO and not got_info[INFO[topic]]:
            K[INFO[topic]] = np.array(r.k, dtype=np.float64).reshape(3, 3)
            got_info[INFO[topic]] = True
        elif topic in ("/tf", "/tf_static"):
            for tr in r.transforms:
                T = np.eye(4)
                T[:3, :3] = quat_to_mat(tr.transform.rotation.x, tr.transform.rotation.y,
                                        tr.transform.rotation.z, tr.transform.rotation.w)
                T[:3, 3] = [tr.transform.translation.x, tr.transform.translation.y,
                            tr.transform.translation.z]
                edges[tr.child_frame_id] = (tr.header.frame_id, T)
        elif topic == ODOM:
            q = r.pose.pose.orientation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                             1 - 2 * (q.y * q.y + q.z * q.z))
            pos = np.array([r.pose.pose.position.x, r.pose.pose.position.y])
            if last_pose is not None:
                dt = (stamp - last_pose[2]) / 1e9
                d = float(np.linalg.norm(pos - last_pose[:2]))
                speed = d / dt if dt > 1e-3 else 0.0
                if speed > args.min_speed:
                    travelled += d
            else:
                speed = 0.0
            last_pose = (pos[0], pos[1], stamp)
            if travelled >= next_emit and speed > args.min_speed and not want_emit:
                next_emit = travelled + args.spacing
                pending = (pos.copy(), yaw, travelled, speed)
                want_emit = True
                dbg["pending"] += 1
        # emit on the lidar message nearest the pending trigger
        if topic == LIDAR:
            dbg["lidar"] += 1
            if want_emit:
                jpegs, ok = [], True
                for i in range(6):
                    if not cam_buf[i]:
                        dbg["nocams"] += 1
                        ok = False
                        break
                    best = min(cam_buf[i], key=lambda e: abs(e[0] - stamp))
                    if abs(best[0] - stamp) > args.tol_ms * 1e6:
                        dbg["cam_tol"] += 1
                        ok = False
                        break
                    jpegs.append(best[1])
                if ok:
                    ppos, pyaw, ptrav, pspeed = pending
                    pts = cloud_to_numpy(r)
                    if not cam2ego.any() or not lidar2ego.any():
                        def pose_in_base(frame, depth=0):
                            if frame not in edges or depth > 8:
                                return None
                            parent, T = edges[frame]
                            if parent == "base_link":
                                return T
                            up = pose_in_base(parent, depth + 1)
                            return None if up is None else up @ T
                        for name, slot in FRAME_TO_SLOT.items():
                            P = pose_in_base(name)
                            if P is None:
                                continue
                            if slot == "lidar":
                                lidar2ego = P
                            elif not cam2ego[slot].any():
                                cam2ego[slot] = P
                        np.savez_compressed(out / "calib.npz", K=K, camera2ego=cam2ego,
                                            lidar2ego=lidar2ego,
                                            names=np.array(CAMS))
                    np.savez_compressed(
                        out / f"frame_{written:04d}.npz",
                        jpegs=np.array(jpegs, dtype=object), points=pts,
                        stamp=stamp,
                        utm=np.array([ppos[0], ppos[1], pyaw], np.float64),
                        travelled=np.float64(ptrav), speed=np.float64(pspeed))
                    written += 1
                    dbg["emitted"] += 1
                    want_emit = False
                    el = time.time() - t_start
                    print(f"[{written:4d}] t={ptrav:7.0f} m v={pspeed:4.1f} m/s "
                          f"pts={len(pts):6d} ({el/60:.1f} min)", flush=True)
                    if args.max_frames and written >= args.max_frames:
                        print("reached --max-frames")
                        break

        if args.max_gib and args.local is None:
            if getattr(stream, "pos", 0) > args.max_gib * 2**30:
                print(f"reached --max-gib ({stream.pos / 2**30:.1f} GiB read)")
                break
        if total_msgs % 200000 == 0:
            pos = stream.tell() if hasattr(stream, "tell") else 0
            print(f"  ... {total_msgs} msgs, {pos / 2**30:.1f} GiB, "
                  f"{written} frames, travelled {travelled:.0f} m", flush=True)

    print(f"\nwrote {written} frames to {out}; total travel {travelled:.0f} m")
    print(f"debug: {dbg}")


if __name__ == "__main__":
    main()
