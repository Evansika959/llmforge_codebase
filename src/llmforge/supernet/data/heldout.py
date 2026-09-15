"""Held-out documents for scoring supernet slices.

Documents come from the last FineWeb-Edu sample-10BT shard. Supernet training reads shards in
sorted order starting from the first, so the last shard is unseen. Documents of 500 characters or
fewer are skipped and every document is truncated to `max_chars`.

Frozen copies of the leading documents ship in assets/heldout, so scores reproduce without
downloading the shard. heldout_texts.json holds the first 96 documents and heldout_ab.json the
first 192. A search scores the first `n` documents and validation
re-scores a disjoint slice selected with `skip`.
"""
import glob
import json
import os
from pathlib import Path

from ...paths import HELDOUT
from ..paths import RAW

FROZEN = HELDOUT / "heldout_texts.json"
FROZEN_AB = HELDOUT / "heldout_ab.json"
FROZEN_MAX_CHARS = 4000


def _from_parquet(n, max_chars, skip):
    import pyarrow.parquet as pq

    cand = sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--HuggingFaceFW--fineweb-edu/snapshots/*/sample/10BT/*.parquet")))
    if not cand:
        cand = sorted(glob.glob(str(RAW / "*.parquet")))
    if not cand:
        raise FileNotFoundError("requested documents exceed the frozen copy in assets/heldout "
                                "and no FineWeb-Edu sample-10BT parquet shard was found")
    f = pq.ParquetFile(cand[-1])
    out, seen = [], 0
    for rg in range(f.num_row_groups):
        for v in f.read_row_group(rg, columns=["text"])["text"]:
            t = v.as_py()
            if t and len(t) > 500:
                seen += 1
                if seen > skip:
                    out.append(t[:max_chars])
            if len(out) >= n:
                return out
    return out


def heldout_texts(n, max_chars=4000, skip=0, source=None):
    """Return documents `skip` to `skip + n` of the held-out stream."""
    if source:
        paths = [Path(source)]
    else:
        paths = [FROZEN, FROZEN_AB] if max_chars <= FROZEN_MAX_CHARS else []
    for path in paths:
        if path.exists():
            docs = json.loads(path.read_text())
            if skip + n <= len(docs):
                return [d[:max_chars] for d in docs[skip:skip + n]]
    return _from_parquet(n, max_chars, skip)
