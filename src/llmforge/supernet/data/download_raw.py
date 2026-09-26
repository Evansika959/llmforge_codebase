#!/usr/bin/env python
"""Download the four raw buckets of the supernet training mixture as jsonl.

The SmolLM2 recipes and the Qwen3-0.6B recipe mix web, code, math and retrieval text at 40, 25, 20
and 15 percent. This module fetches the raw text of each bucket with the same sources, filters,
formats and byte caps that produced the packs behind the paper. Each bucket is written to
`$LLMFORGE_DATA/raw/<bucket>.jsonl`, one {"text": ..., "source": ...} object per line, which is
what `scripts/supernet/pack_smollm.sh` and `scripts/supernet/pack_qwen.sh` read.

| bucket       | source                                   | filter and format                         | cap  |
|--------------|------------------------------------------|-------------------------------------------|------|
| fineweb_edu  | HuggingFaceFW/fineweb-edu, sample-10BT   | document text as is                       | 3 GB |
| code         | bigcode/the-stack-smol                   | source files of at least 2,000 characters | 3 GB |
| openmath     | nvidia/OpenMathInstruct-2                | "Problem: ... Solution: ..."              | 2 GB |
| rag_hotpot   | hotpotqa/hotpot_qa, distractor, train    | "Documents: ... Question: ... Answer: ..." | 3 GB |

Every source is streamed in its published order and cut at the first document that reaches the
cap, so a rerun reproduces the same documents as long as the upstream datasets are unchanged.
HotpotQA ends before its cap. bigcode/the-stack-smol is gated: accept its terms of use on the
Hugging Face Hub and log in with `huggingface-cli login` before fetching the code bucket.

The Qwen3 recipes read their web bucket from the complete sample-10BT parquet instead, which
`python -m llmforge.supernet.data.download` fetches.

  python -m llmforge.supernet.data.download_raw                    # all four buckets
  python -m llmforge.supernet.data.download_raw --only code rag_hotpot
"""
import argparse
import json
import time

from datasets import load_dataset

from ..paths import DATA

RAW = DATA / "raw"
GB = 1024 ** 3


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def write_stream(name, examples, text_fn, cap_bytes, extra=None):
    """Write each example that `text_fn` keeps until the written text reaches `cap_bytes`."""
    path = RAW / f"{name}.jsonl"
    n = nbytes = 0
    t0 = time.time()
    with open(path, "w") as f:
        for ex in examples:
            try:
                text = text_fn(ex)
            except Exception:
                continue
            if not text:
                continue
            row = {"text": text, "source": name}
            if extra:
                row.update(extra(ex))
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
            nbytes += len(text.encode("utf-8"))
            if n % 50000 == 0:
                log(f"  {name}: {n} docs, {nbytes / 1e6:.0f} MB")
            if nbytes >= cap_bytes:
                break
    log(f"DONE {name}: {n} docs, {nbytes / 1e6:.0f} MB in {time.time() - t0:.0f}s -> {path}")


def fineweb_edu():
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    write_stream("fineweb_edu", ds, lambda e: e.get("text"), cap_bytes=3 * GB)


def code():
    # Long files only. The short ones are mostly configuration and boilerplate.
    ds = load_dataset("bigcode/the-stack-smol", split="train", streaming=True)

    def text(e):
        c = e.get("content")
        return c if c and len(c) >= 2000 else None

    write_stream("code", ds, text, cap_bytes=3 * GB, extra=lambda e: {"lang": e.get("lang")})


def openmath():
    ds = load_dataset("nvidia/OpenMathInstruct-2", split="train", streaming=True)

    def text(e):
        prob, sol = e.get("problem"), e.get("generated_solution")
        return f"Problem:\n{prob}\n\nSolution:\n{sol}" if prob and sol else None

    write_stream("openmath", ds, text, cap_bytes=2 * GB)


def rag_hotpot():
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")

    def text(e):
        ctx = e.get("context") or {}
        paras = [f"[{t}] " + " ".join(s) for t, s in zip(ctx.get("title", []), ctx.get("sentences", []))]
        docs, q, a = "\n".join(paras), e.get("question", ""), e.get("answer", "")
        if not docs or not q:
            return None
        return f"Documents:\n{docs}\n\nQuestion: {q}\nAnswer: {a}"

    write_stream("rag_hotpot", ds, text, cap_bytes=3 * GB)


BUCKETS = {"fineweb_edu": fineweb_edu, "code": code, "openmath": openmath, "rag_hotpot": rag_hotpot}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=sorted(BUCKETS), help="fetch these buckets only")
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    for name in args.only or BUCKETS:
        log(f"=== {name} ===")
        BUCKETS[name]()


if __name__ == "__main__":
    main()
