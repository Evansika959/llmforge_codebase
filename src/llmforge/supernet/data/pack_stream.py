#!/usr/bin/env python
"""M5: streaming, memory-safe doc-aware tokenize+pack. Same output contract as data/pack.py.

Why a new packer instead of extending data/pack.py: that one accumulates the whole token stream
in a Python list before `np.array(...)`. At 10B tokens that list alone is ~360 GB (8 B/pointer +
28 B/PyLong), so it cannot run -- it was fine for the 300M-token buckets it was written for.
This version never holds more than one batch: tokens append straight to a uint32 .bin, and the
.npy is materialized once at the end from a known shape.

Output is byte-identical in FORMAT to pack.py so data/datamix.py + data/collate.py work unchanged:
  data/packed/<out>_tokens.npy    uint32 [n_seq, SEQLEN]  (doc-aware, EOS-separated)
  data/packed/<out>_seglens.jsonl one JSON list of segment lengths per sequence, summing to SEQLEN

Segment lengths drive the block-diagonal attention mask and per-doc position_ids at train time, so
attention never crosses a document boundary and RoPE restarts per doc.

  python -m llmforge.supernet.data.pack_stream --parquet-dir <dir> --out fineweb10bt
  python -m llmforge.supernet.data.pack_stream --parquet-dir <dir> --out smoke --max-tokens 20_000_000
  python -m llmforge.supernet.data.pack_stream --jsonl data/raw/code.jsonl --out code_big
"""
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")  # rust batch-encode uses all cores

import argparse
import glob
import json
import time

import numpy as np
from transformers import AutoTokenizer

from ..paths import PACKED as _PACKED
OUT = str(_PACKED)
# All Qwen3 scales share one tokenizer and vocab_size=151936, so a single pack serves 0.6B/1.7B/4B.
MODEL = os.environ.get("PACK_TOKENIZER", "Qwen/Qwen3-1.7B-Base")
SEQLEN = 4096
COPY_CHUNK = 65536  # sequences per .bin -> .npy copy chunk (~1 GB at 4096 uint32)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def iter_parquet_texts(d, column="text"):
    """Yield texts row-group at a time so a 2 GB shard never lands in RAM whole."""
    import pyarrow.parquet as pq
    for fp in sorted(glob.glob(os.path.join(d, "*.parquet"))):
        f = pq.ParquetFile(fp)
        for rg in range(f.num_row_groups):
            col = f.read_row_group(rg, columns=[column])[column]
            for v in col:
                t = v.as_py()
                if t:
                    yield t
        log(f"  finished shard {os.path.basename(fp)}")


def iter_jsonl_texts(path):
    with open(path) as f:
        for line in f:
            try:
                t = json.loads(line).get("text")
            except Exception:
                continue
            if t:
                yield t


def write_seglens(path, boundaries, n_seq, seqlen):
    """Merge-walk sorted doc starts against sequence windows -- O(n_docs + n_seq), streamed to disk.

    Mirrors pack.py: a doc starting exactly on a window boundary does not emit a duplicate start.
    """
    i, n_b = 0, len(boundaries)
    with open(path, "w") as f:
        for s in range(n_seq):
            lo, hi = s * seqlen, (s + 1) * seqlen
            while i < n_b and boundaries[i] <= lo:
                i += 1
            starts, j = [lo], i
            while j < n_b and boundaries[j] < hi:
                starts.append(boundaries[j]); j += 1
            starts.append(hi)
            f.write(json.dumps([starts[k + 1] - starts[k] for k in range(len(starts) - 1)]) + "\n")


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--parquet-dir")
    src.add_argument("--jsonl")
    ap.add_argument("--out", required=True, help="bucket name -> data/packed/<out>_tokens.npy")
    ap.add_argument("--max-tokens", type=int, default=0, help="0 = no cap")
    ap.add_argument("--seqlen", type=int, default=SEQLEN)
    ap.add_argument("--batch", type=int, default=2000, help="docs per tokenizer call")
    ap.add_argument("--flush", type=int, default=8_000_000, help="tokens buffered before disk write")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    bin_path = os.path.join(OUT, f".{args.out}_tokens.bin")
    npy_path = os.path.join(OUT, f"{args.out}_tokens.npy")
    seg_path = os.path.join(OUT, f"{args.out}_seglens.jsonl")

    tok = AutoTokenizer.from_pretrained(MODEL)
    eos = tok.eos_token_id
    texts = iter_parquet_texts(args.parquet_dir) if args.parquet_dir else iter_jsonl_texts(args.jsonl)
    log(f"tokenizer={MODEL} eos={eos} seqlen={args.seqlen} -> {npy_path}")

    boundaries = []          # global start offset of each doc; ~10M entries at 10B tokens
    total = 0                # tokens written so far
    buf = []                 # token buffer, flushed every --flush tokens
    t0 = time.time()
    hit_cap = False

    def flush(fh):
        nonlocal buf
        if buf:
            np.asarray(buf, dtype=np.uint32).tofile(fh)
            buf = []

    with open(bin_path, "wb") as fh:
        batch = []
        for text in texts:
            batch.append(text)
            if len(batch) < args.batch:
                continue
            for ids in tok(batch, add_special_tokens=False).input_ids:
                boundaries.append(total)
                buf.extend(ids); buf.append(eos)
                total += len(ids) + 1
            batch = []
            if len(buf) >= args.flush:
                flush(fh)
                el = time.time() - t0
                log(f"  {total/1e9:.3f}B tokens | {len(boundaries)/1e6:.2f}M docs | "
                    f"{total/1e6/max(el,1):.1f}M tok/s | {el/60:.1f} min")
            if args.max_tokens and total >= args.max_tokens:
                hit_cap = True
                break
        if batch and not hit_cap:
            for ids in tok(batch, add_special_tokens=False).input_ids:
                boundaries.append(total)
                buf.extend(ids); buf.append(eos)
                total += len(ids) + 1
        flush(fh)

    n_seq = total // args.seqlen
    log(f"tokenized {total/1e9:.3f}B tokens from {len(boundaries)/1e6:.2f}M docs in "
        f"{(time.time()-t0)/60:.1f} min -> {n_seq} sequences ({n_seq*args.seqlen/1e9:.3f}B kept)")

    log("materializing .npy ...")
    arr = np.lib.format.open_memmap(npy_path, mode="w+", dtype=np.uint32, shape=(n_seq, args.seqlen))
    with open(bin_path, "rb") as fh:
        done = 0
        while done < n_seq:
            m = min(COPY_CHUNK, n_seq - done)
            chunk = np.fromfile(fh, dtype=np.uint32, count=m * args.seqlen)
            arr[done:done + m] = chunk.reshape(m, args.seqlen)
            done += m
    arr.flush(); del arr
    os.remove(bin_path)

    log("writing seglens ...")
    write_seglens(seg_path, boundaries, n_seq, args.seqlen)

    # verify the contract datamix/collate rely on
    chk = np.load(npy_path, mmap_mode="r")
    with open(seg_path) as f:
        n_lines = sum(1 for _ in f)
    with open(seg_path) as f:
        sums = [sum(json.loads(l)) for _, l in zip(range(1000), f)]
    ok = chk.shape == (n_seq, args.seqlen) and n_lines == n_seq and set(sums) == {args.seqlen}
    log(f"[{args.out}] {n_seq} seqs x {args.seqlen} = {n_seq*args.seqlen/1e9:.3f}B tokens | "
        f"dtype={chk.dtype} | seglens lines={n_lines} | contract={'OK' if ok else 'FAILED'}")
    if not ok:
        raise SystemExit(f"contract check FAILED: shape={chk.shape} lines={n_lines} "
                         f"distinct_seglen_sums={sorted(set(sums))[:5]}")
    log(f"total wall {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
