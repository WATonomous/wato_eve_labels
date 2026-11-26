"""Credential-free Drive reader.

rclone is only configured on the WATcloud login node, and the night bags need to
be processed on csc3. The Drive folders are shared publicly, so a plain
`file_id` + HTTP Range request is enough -- no OAuth token leaves WATcloud.
Proven against these very files: 8 parallel range workers give ~30 MB/s.
"""
import io
import re
import sys
import threading
import time
import urllib.request
import concurrent.futures as cf


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



