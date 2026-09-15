#!/usr/bin/env python
"""Correctness gate for data/pack_stream.py: must reproduce data/pack.py EXACTLY.

pack.py is re-implemented inline here rather than imported/executed, because running it would
overwrite the existing data/packed/{code,rag,fineweb,math}_* buckets that every M2/M4 result
depends on. This reads the same raw jsonl, applies pack.py's algorithm verbatim (lines 42-84),
and diffs against pack_stream.py's output token-for-token and seglen-for-seglen.

  python experiments/supernet_fidelity/verify_pack_stream.py   # 20M tokens of fineweb_edu.jsonl
  python experiments/supernet_fidelity/verify_pack_stream.py --src data/raw/code.jsonl --max-tokens 10000000
"""
import argparse
import bisect
import json
import os
import subprocess
import sys

import numpy as np
from transformers import AutoTokenizer


from llmforge.supernet.paths import PACKED as _P, ROOT as _R, RAW as _RAW
PACKED = str(_P)
MODEL = os.environ.get("PACK_TOKENIZER", "Qwen/Qwen3-1.7B-Base")
SEQLEN = 4096
NAME = "_verify_stream"


def iter_texts(path):
    with open(path) as f:
        for line in f:
            try:
                t = json.loads(line).get("text")
            except Exception:
                continue
            if t:
                yield t


def reference_pack(path, tok, max_tokens, batch=1000):
    """data/pack.py::pack_bucket, verbatim (in-memory; only safe for small caps)."""
    eos = tok.eos_token_id
    stream, boundaries = [], []

    def add_docs(texts):
        for ids in tok(texts, add_special_tokens=False).input_ids:
            boundaries.append(len(stream))
            stream.extend(ids)
            stream.append(eos)

    buf = []
    for t in iter_texts(path):
        buf.append(t)
        if len(buf) >= batch:
            add_docs(buf); buf = []
            if len(stream) >= max_tokens:
                break
    if buf and len(stream) < max_tokens:
        add_docs(buf)

    n_seq = len(stream) // SEQLEN
    arr = np.array(stream[: n_seq * SEQLEN], dtype=np.uint32).reshape(n_seq, SEQLEN)
    seglens_all = []
    for s in range(n_seq):
        lo, hi = s * SEQLEN, (s + 1) * SEQLEN
        starts = [lo]
        i = bisect.bisect_right(boundaries, lo)
        while i < len(boundaries) and boundaries[i] < hi:
            starts.append(boundaries[i]); i += 1
        starts.append(hi)
        seglens_all.append([starts[k + 1] - starts[k] for k in range(len(starts) - 1)])
    return arr, seglens_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None)
    ap.add_argument("--max-tokens", type=int, default=20_000_000)
    args = ap.parse_args()
    ROOT = str(_R)
    if args.src is None: args.src = os.path.join(str(_RAW), "fineweb_edu.jsonl")

    # pack.py batches docs 1000 at a time and stops on the first batch that crosses max_tokens;
    # pack_stream must be given the same batch size or the two consume different doc counts.
    print(f"[verify] running pack_stream.py on {os.path.basename(args.src)} "
          f"(cap {args.max_tokens/1e6:.0f}M, batch 1000)", flush=True)
    r = subprocess.run([sys.executable, "-m", "llmforge.supernet.data.pack_stream",
                        "--jsonl", args.src, "--out", NAME,
                        "--max-tokens", str(args.max_tokens), "--batch", "1000"],
                       cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit("pack_stream.py failed")

    got = np.load(os.path.join(PACKED, f"{NAME}_tokens.npy"), mmap_mode="r")
    with open(os.path.join(PACKED, f"{NAME}_seglens.jsonl")) as f:
        got_seg = [json.loads(l) for l in f]

    print("[verify] running pack.py's algorithm inline for reference ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    ref, ref_seg = reference_pack(args.src, tok, args.max_tokens)

    print(f"[verify] shapes  stream={got.shape} dtype={got.dtype} | ref={ref.shape} dtype={ref.dtype}")
    ok_shape = got.shape == ref.shape and got.dtype == ref.dtype
    ok_tok = ok_shape and np.array_equal(np.asarray(got), ref)
    ok_seg = got_seg == ref_seg
    ok_sum = all(sum(s) == SEQLEN for s in got_seg)

    if not ok_tok and ok_shape:
        bad = np.argwhere(np.asarray(got) != ref)
        print(f"[verify] first token mismatches: {bad[:5].tolist()}")
    if not ok_seg:
        for i, (a, b) in enumerate(zip(got_seg, ref_seg)):
            if a != b:
                print(f"[verify] first seglen mismatch at seq {i}: stream={a} ref={b}")
                break
        print(f"[verify] seglen list lengths: stream={len(got_seg)} ref={len(ref_seg)}")

    print(f"\n[verify] shape/dtype match : {ok_shape}")
    print(f"[verify] tokens identical  : {ok_tok}")
    print(f"[verify] seglens identical : {ok_seg}")
    print(f"[verify] all seglens sum to {SEQLEN}: {ok_sum}")
    verdict = ok_shape and ok_tok and ok_seg and ok_sum
    print(f"\n[verify] VERDICT: {'PASS -- pack_stream reproduces pack.py exactly' if verdict else 'FAIL'}")

    for suf in ("_tokens.npy", "_seglens.jsonl"):
        p = os.path.join(PACKED, NAME + suf)
        if os.path.exists(p):
            os.remove(p)
    print("[verify] cleaned up temp artifacts")
    if not verdict:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
