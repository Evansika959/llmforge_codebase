"""Does supernet training FLATTEN per-layer sensitivity, or was the base's spread an artifact?

The NSGA-II search over the trained 4B supernet found that per-layer heterogeneity does not beat
a uniform grid, and per-layer sensitivity is 6x flatter on the trained supernet (CV 0.429) than on
the untrained base (CV 2.580). Two incompatible readings:

  H1  training flattens real differences. Sandwich sampling draws per-layer widths uniformly at
      random, so every layer is optimized to tolerate shrinking equally. Predicts CV declines
      PROGRESSIVELY across checkpoints.
  H2  the base's spread is itself an artifact. An untrained checkpoint cannot be sliced at all, so
      the probe measures "how badly does this break an unsliceable model", not layer capacity.
      Predicts CV collapses at the FIRST checkpoint and is then flat.

The shape of CV vs training step discriminates them. Both models are measured because the 1.7B
saves every 500 steps, giving finer resolution exactly where the two curves separate.

Two probe families, because a result that holds for only one probe is a fact about the probe:
  attn  shrink one layer's d_qk and d_v to the floor
  mlp   shrink one layer's d_mlp to the floor

  python experiments/supernet_fidelity/sensitivity_trajectory.py --model qwen3-1.7b
"""
import argparse, glob, json, os, re, statistics as st
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmforge.supernet.config import SPECS
from llmforge.supernet.elastic import build_pair_order, enable_elastic, set_elastic_backend
from llmforge.supernet.data.heldout import heldout_texts
from llmforge.supernet.paths import RUNS
from llmforge.supernet.search.nsga import load_supernet, make_evaluator
from llmforge.supernet.space import ElasticConfig


def probes(spec, kind):
    """One config per layer, that layer alone shrunk to the floor."""
    out = []
    for L in range(spec.n_layers):
        qk = [spec.head_dim] * spec.n_layers; v = list(qk)
        mlp = [spec.d_mlp] * spec.n_layers
        if kind == "attn": qk[L] = v[L] = spec.qk_grid[0]
        else:              mlp[L] = spec.mlp_grid[0]
        out.append(ElasticConfig(spec, qk, v, [spec.n_kv] * spec.n_layers,
                                 [spec.n_q] * spec.n_layers, mlp))
    return out


def measure(model, tok, order, spec, texts):
    loss_of, _ = make_evaluator(model, tok, order, spec, texts)
    base = loss_of(ElasticConfig.full(spec))
    res = {}
    for kind in ("attn", "mlp"):
        d = [loss_of(c) - base for c in probes(spec, kind)]
        res[kind] = {"damage": d, "cv": st.pstdev(d) / st.mean(d),
                     "spread": max(d) / max(min(d), 1e-9), "mean": st.mean(d)}
    res["full_loss"] = base
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-1.7b", choices=sorted(SPECS))
    ap.add_argument("--n-texts", type=int, default=24)
    a = ap.parse_args()
    spec = SPECS[a.model]
    texts = heldout_texts(a.n_texts)
    d = f"{RUNS}/{a.model}_ab"
    steps = sorted((int(re.search(r"step(\d+)", p).group(1)), p)
                   for p in glob.glob(d + "/step*") if os.path.isdir(p))

    rows = []
    # step 0 = the published base, elastic-patched but untrained
    mb = AutoModelForCausalLM.from_pretrained(
        spec.repo, torch_dtype=torch.bfloat16, attn_implementation="eager").to("cuda").eval()
    enable_elastic(mb); set_elastic_backend("eager"); mb.config.use_cache = False
    r = measure(mb, AutoTokenizer.from_pretrained(spec.repo), build_pair_order("hilo"), spec, texts)
    r["step"] = 0; rows.append(r)
    print(f"step {0:>5}  full {r['full_loss']:.4f} | "
          f"attn CV {r['attn']['cv']:.3f} spread {r['attn']['spread']:6.1f}x | "
          f"mlp CV {r['mlp']['cv']:.3f} spread {r['mlp']['spread']:6.1f}x", flush=True)
    del mb; torch.cuda.empty_cache()

    for s, p in steps:
        m, t, o = load_supernet(p, spec)
        r = measure(m, t, o, spec, texts); r["step"] = s; rows.append(r)
        print(f"step {s:>5}  full {r['full_loss']:.4f} | "
              f"attn CV {r['attn']['cv']:.3f} spread {r['attn']['spread']:6.1f}x | "
              f"mlp CV {r['mlp']['cv']:.3f} spread {r['mlp']['spread']:6.1f}x", flush=True)
        del m; torch.cuda.empty_cache()

    b = rows[0]
    print("\ncorrelation of per-layer damage with the BASE ordering:")
    for r in rows[1:]:
        print(f"  step {r['step']:>5}  attn r={np.corrcoef(r['attn']['damage'],b['attn']['damage'])[0,1]:+.3f}"
              f"   mlp r={np.corrcoef(r['mlp']['damage'],b['mlp']['damage'])[0,1]:+.3f}")
    out = RUNS / f"sensitivity_trajectory_{a.model}.json"
    json.dump({"model": a.model, "n_texts": a.n_texts, "rows": rows}, open(out, "w"), indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
