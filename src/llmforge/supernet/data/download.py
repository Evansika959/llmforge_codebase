#!/usr/bin/env python
"""M5: fetch the FULL FineWeb-Edu `sample-10BT` subset (14 parquet shards, 28.5 GB).

Why a new script rather than data/download_data.py::bucket_fineweb: that one streams the same
config but caps at 3 GB of raw text (~301M packed tokens), which is what limited the M2/M4 runs
to ~1.8 epochs over the fineweb bucket. M5 shifts the mix to 55% fineweb and wants headroom for
a clean from-scratch run, so we take the whole 10BT sample.

We pull PARQUET into the HF cache instead of re-emitting a ~40 GB raw/*.jsonl:
  * snapshot_download is parallel and natively resumable (re-run to continue an interrupted pull);
  * data/pack_stream.py reads parquet directly, so we skip a 40 GB write + 40 GB read;
  * nothing is duplicated on disk -- we read from the cache path this script prints.

  python -m llmforge.supernet.data.download          # fetch (idempotent; safe to re-run)
  python -m llmforge.supernet.data.download --check  # report what is already local, fetch nothing
"""
import argparse
import os
import time

from huggingface_hub import snapshot_download
from huggingface_hub import HfApi

REPO = "HuggingFaceFW/fineweb-edu"
PATTERN = "sample/10BT/*.parquet"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report local state only, download nothing")
    ap.add_argument("--workers", type=int, default=8, help="parallel shard downloads")
    args = ap.parse_args()

    api = HfApi()
    info = api.repo_info(REPO, repo_type="dataset", files_metadata=True)
    want = {s.rfilename: (s.size or 0) for s in info.siblings
            if s.rfilename.startswith("sample/10BT/") and s.rfilename.endswith(".parquet")}
    log(f"{REPO}:{PATTERN} -> {len(want)} shards, {sum(want.values())/1e9:.1f} GB")

    if args.check:
        try:
            path = snapshot_download(REPO, repo_type="dataset", allow_patterns=PATTERN,
                                     local_files_only=True)
            have = sorted(f for f in os.listdir(os.path.join(path, "sample", "10BT"))
                          if f.endswith(".parquet"))
            got = sum(os.path.getsize(os.path.join(path, "sample", "10BT", f)) for f in have)
            log(f"LOCAL {len(have)}/{len(want)} shards, {got/1e9:.1f} GB at {path}")
        except Exception as e:
            log(f"nothing local yet ({type(e).__name__})")
        return

    t0 = time.time()
    path = snapshot_download(REPO, repo_type="dataset", allow_patterns=PATTERN,
                             max_workers=args.workers)
    d = os.path.join(path, "sample", "10BT")
    shards = sorted(f for f in os.listdir(d) if f.endswith(".parquet"))
    got = sum(os.path.getsize(os.path.join(d, f)) for f in shards)
    dt = time.time() - t0
    log(f"DONE {len(shards)}/{len(want)} shards, {got/1e9:.1f} GB in {dt/60:.1f} min "
        f"({got/1e6/max(dt,1):.0f} MB/s)")
    log(f"parquet dir: {d}")
    log(f"next: python -m llmforge.supernet.data.pack_stream --parquet-dir {d} --out fineweb10bt")


if __name__ == "__main__":
    main()
