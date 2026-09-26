"""M1: ratio-mixed sampler over the packed buckets.

Every recipe passes its mix through --mix. The default is the Qwen3-0.6B recipe's web, code, math and
retrieval mix at 40, 25, 20 and 15 percent. The SmolLM2 recipes use the same mix over the sl_ buckets.
Each batch element independently picks a bucket by ratio, then a random packed sequence.
Returns (tokens [B, SEQLEN] long, seglens_batch: list[list[int]]) -> feed to data.collate.
"""
import os, json, random
import numpy as np
import torch

from ..paths import PACKED as _PACKED
PACKED = str(_PACKED)
DEFAULT_RATIOS = {"fineweb10bt": 0.40, "code": 0.25, "math": 0.20, "rag": 0.15}


class DataMix:
    def __init__(self, ratios=DEFAULT_RATIOS, seed=0):
        self.rng = random.Random(seed)
        self.names, self.weights = [], []
        self.tokens, self.seglens = {}, {}
        for name, w in ratios.items():
            tpath = os.path.join(PACKED, f"{name}_tokens.npy")
            spath = os.path.join(PACKED, f"{name}_seglens.jsonl")
            if not os.path.exists(tpath):
                print(f"[datamix] skip {name} (not packed yet)")
                continue
            self.tokens[name] = np.load(tpath, mmap_mode="r")
            with open(spath) as f:
                self.seglens[name] = [json.loads(l) for l in f]
            self.names.append(name); self.weights.append(w)
            print(f"[datamix] {name}: {self.tokens[name].shape[0]} seqs")
        assert self.names, "no packed buckets found -- run python -m llmforge.supernet.data.pack_stream first"
        # A skipped bucket silently renormalises the mix, which changes what the run is actually
        # trained on. Print the EFFECTIVE ratios so that shows up in the log rather than in the
        # results three days later.
        tot = sum(self.weights)
        eff = ", ".join(f"{n}={w / tot:.3f}" for n, w in zip(self.names, self.weights))
        miss = [n for n in ratios if n not in self.tokens]
        # Exposed as an attribute, not only printed, so a run can record what it ACTUALLY trained
        # on. Four n_kv reference models were once dispatched two per machine, the machines had
        # different buckets on disk, and the rungs ended up alternating between two recipes --
        # confounding the axis under study with the data. Their arch.json recorded the request.
        self.effective = {n: w / tot for n, w in zip(self.names, self.weights)}
        self.missing = miss
        print(f"[datamix] effective mix: {eff}"
              + (f"   MISSING {miss} -- requested mix NOT honoured" if miss else ""))
        self.seqlen = self.tokens[self.names[0]].shape[1]

    def _one(self):
        name = self.rng.choices(self.names, weights=self.weights, k=1)[0]
        i = self.rng.randrange(self.tokens[name].shape[0])
        return np.asarray(self.tokens[name][i], dtype=np.int64), self.seglens[name][i]

    def batch(self, B):
        toks, segs = [], []
        for _ in range(B):
            t, s = self._one()
            toks.append(t); segs.append(s)
        return torch.from_numpy(np.stack(toks)), segs
