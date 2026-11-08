"""Download only the RealSense files of the XYZ-IBD validation split (about 90 MB instead of 7.7 GB),
reading single members of the remote zip with HTTP range requests.

    python -m datasets.download_xyzibd --out data/xyzibd --every 5

base and models are small: unzip xyzibd_base.zip and xyzibd_models.zip into the same folder.
"""
import argparse
import os
import re
import struct
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor

import requests
from remotezip import RemoteZip

URL = "https://huggingface.co/datasets/bop-benchmark/xyzibd/resolve/main/xyzibd_val.zip"
KEEP = re.compile(r"(scene_(camera|gt|gt_info)_realsense\.json$)|"
                  r"(/(depth_realsense)/(\d{6})\.png$)|(/mask_visib_realsense/(\d{6})_\d{6}\.png$)")


def scene_of(name):
    p = name.split("/")
    for i in range(len(p) - 2):
        if p[i].startswith("val") and p[i + 1].isdigit():
            return p[i + 1]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=URL)
    ap.add_argument("--out", default="data/xyzibd")
    ap.add_argument("--every", type=int, default=5, help="keep frames whose id is a multiple of N")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    with RemoteZip(a.url) as z:
        todo = []
        for n in z.namelist():
            m = KEEP.search(n)
            if scene_of(n) is None or not m:
                continue
            fid = m.group(5) or m.group(7)
            if fid is not None and int(fid) % a.every:
                continue
            todo.append(n)
        info = {n: (z.getinfo(n).header_offset, z.getinfo(n).compress_size, z.getinfo(n).compress_type) for n in todo}
    print(f"{len(todo)} files, {sum(v[1] for v in info.values()) / 1e6:.0f} MB")
    local = threading.local()

    def fetch(n):
        p = n.split("/")
        dst = os.path.join(a.out, "/".join(p[next(i for i, s in enumerate(p) if s.startswith("val")):]))
        if os.path.exists(dst):
            return
        if not hasattr(local, "s"):
            local.s = requests.Session()
        off, size, ctype = info[n]
        rng = f"bytes={off}-{off + 30 + len(n.encode()) + 4096 + size}"
        for attempt in range(8):
            with local.s.get(a.url, headers={"Range": rng}, timeout=60, stream=True) as r:
                if r.status_code == 206:
                    buf = r.content
                    break
            time.sleep(2 ** attempt)    # rate limited (HTTP 429) or transient error
        else:
            raise RuntimeError(f"failed: {n}")
        nlen, xlen = struct.unpack("<HH", buf[26:30])
        raw = buf[30 + nlen + xlen: 30 + nlen + xlen + size]
        data = raw if ctype == 0 else zlib.decompress(raw, -15)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst + ".part", "wb") as f:
            f.write(data)
        os.replace(dst + ".part", dst)

    with ThreadPoolExecutor(a.workers) as ex:
        for i, _ in enumerate(ex.map(fetch, todo)):
            if i % 200 == 0:
                print(f"{i}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
