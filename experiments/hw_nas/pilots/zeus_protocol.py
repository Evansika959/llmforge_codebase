"""Pilots that chose the GPU measurement protocol: dynamic range against measurement noise.

For each scale the script first checks the cached decode paths in fp32: eager cached decoding against
one uncached forward pass, and CUDA-graph replay against eager cached decoding, on a uniform and a
heterogeneous architecture. It then measures the full, a middle and the smallest uniform slice plus two
random heterogeneous architectures under three protocols at batch 64 with a 512-token prompt.

    eager-decode   128 eager decode steps in one window
    graph-decode   128 decode steps replayed from CUDA graphs, windows repeated to 2 s
    prefill        prefill only, windows repeated to 2 s

Two more independent passes over the full and the smallest slice give run-to-run noise. A protocol is
usable when the full-to-smallest ratio is large against the spread across repeats and passes, and
heterogeneous architectures cost what their size suggests.

    python experiments/hw_nas/pilots/zeus_protocol.py --out runs/pilots/zeus_protocol.jsonl
"""
import argparse
import json
import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch

from llmforge.evaluators.cache import gpu_lock
from llmforge.evaluators.hw_zeus import HwZeus
from llmforge.hw.zeus.kv_cache import (attach_iha_kv_cache, detach_iha_kv_cache, graph_parity_check,
                                       parity_check)
from llmforge.hw.zeus.measure import build_model_from_individual
from llmforge.paths import CONFIGS, RUNS
from llmforge.search.elastic_space import ElasticSearchSpace
from llmforge.search.individual import Individual

KEEP = ("ttft_ms", "tpot_ms", "energy_per_token_uJ", "prefill_energy_per_token_uJ", "power_W",
        "hw_feasible", "zeus_error", "zeus_repeats_energy_cv", "zeus_prefill_energy_cv",
        "zeus_prefill_passes", "zeus_decode_passes")
PROTOCOLS = {"eager-decode": dict(decode_len=128, cuda_graphs=False, min_window_s=0.0),
             "graph-decode": dict(decode_len=128, cuda_graphs=True, min_window_s=2.0),
             "prefill": dict(decode_len=0, cuda_graphs=False, min_window_s=2.0)}


def architectures(model):
    u = ElasticSearchSpace.from_yaml(str(CONFIGS / "search_spaces" / f"{model}_uniform.yaml"))
    b = ElasticSearchSpace.from_yaml(str(CONFIGS / "search_spaces" / f"{model}.yaml"), seed=7)
    mid = u.uniform(**{k: u.grids[k][len(u.grids[k]) // 2] for k in u.grids})
    return [("full", u.full()), ("mid", mid), ("smallest", u.smallest()), ("het0", b.sample()), ("het1", b.sample())]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(RUNS / "pilots" / "zeus_protocol.jsonl"))
    ap.add_argument("--models", nargs="+", default=["smollm2-135m", "qwen3-1.7b", "qwen3-4b"])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--prompt", type=int, default=512)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    log = open(a.out, "a")

    def emit(rec):
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)

    base = os.path.join(os.path.dirname(a.out), "zeus_protocol_cache")
    for model in a.models:
        archs = architectures(model)
        with gpu_lock():
            for name, ind in (archs[1], archs[3]):
                m = build_model_from_individual(ind, 64, torch.device("cuda"), torch.float32)
                attach_iha_kv_cache(m)
                ok, max_abs, _ = parity_check(m, prefill_len=24, decode_len=8, atol=1e-3, rtol=1e-3, verbose=False)
                graph_diff = graph_parity_check(m, prefill_len=24, decode_len=8, batch_size=2)
                emit({"tag": "parity", "model": model, "arch": name, "cached_vs_uncached_pass": ok,
                      "cached_vs_uncached_max_abs": max_abs, "graph_vs_eager_max_abs": graph_diff})
                detach_iha_kv_cache(m)
                del m
                torch.cuda.empty_cache()
        for proto, kw in PROTOCOLS.items():
            hz = HwZeus(prefill_len=a.prompt, batch_size=a.batch, n_repeats=3, cache_dir=f"{base}/sweep", **kw)
            for name, ind in archs:
                t0 = time.time()
                r = hz.evaluate([ind])[0]
                emit({"tag": "sweep", "protocol": proto, "model": model, "arch": name,
                      "params_M": round(Individual.from_dict(ind).estimate_params() / 1e6, 1),
                      "wall_s": round(time.time() - t0, 1), **{k: r.get(k) for k in KEEP}})
    for rep in (1, 2):
        for model in a.models:
            archs = architectures(model)
            for proto in ("graph-decode", "prefill"):
                hz = HwZeus(prefill_len=a.prompt, batch_size=a.batch, n_repeats=3,
                            cache_dir=f"{base}/pass{rep}", **PROTOCOLS[proto])
                for name, ind in (archs[0], archs[2]):
                    r = hz.evaluate([ind])[0]
                    emit({"tag": f"pass{rep}", "protocol": proto, "model": model, "arch": name,
                          **{k: r.get(k) for k in KEEP}})


if __name__ == "__main__":
    main()
