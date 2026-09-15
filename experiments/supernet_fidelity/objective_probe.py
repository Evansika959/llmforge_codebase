"""Is there a cheap search objective that tracks CAPABILITY, not just fluency?

The NSGA-II search scored candidates by cross-entropy on held-out web text. That metric never
mis-ranks the configurations -- it is monotone against Minerva MATH -- but its resolution is
useless: +0.54 nats corresponds to losing 80% of mathematical ability, so a search reading it
treats a crippled model as a mild compromise.

A capability benchmark cannot replace it (40 min per candidate against 2.4 s). So the question is
whether cross-entropy measured on IN-DOMAIN text -- the reasoning traces themselves -- separates
the same configurations more sharply at identical cost. If it does, the search objective changes
and nothing else has to.

  python experiments/supernet_fidelity/objective_probe.py --ckpt runs/supernet/qwen3-4b_ab/final
"""
import argparse, json
import torch

from llmforge.supernet.config import SPECS
from llmforge.supernet.elastic.sampler import head_grid
from llmforge.supernet.data.heldout import heldout_texts
from llmforge.supernet.paths import RUNS
from llmforge.supernet.search.nsga import load_supernet, make_evaluator
from llmforge.supernet.space import ElasticConfig


def math_texts(n):
    """Problem + worked solution from MATH-500 -- the same distribution the benchmark scores."""
    from datasets import load_dataset
    for name, cfg in [("HuggingFaceH4/MATH-500", None), ("EleutherAI/hendrycks_math", "algebra")]:
        try:
            ds = load_dataset(name, cfg, split="test") if cfg else load_dataset(name, split="test")
            key = "solution" if "solution" in ds.column_names else "answer"
            out = [f"Problem:\n{d['problem']}\n\nSolution:\n{d[key]}" for d in ds.select(range(min(n, len(ds))))]
            print(f"[data] {name}: {len(out)} math documents", flush=True)
            return out
        except Exception as e:
            print(f"[data] {name} unavailable ({type(e).__name__})", flush=True)
    raise SystemExit("no math corpus available")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-4b")
    ap.add_argument("--ckpt", default=f"{RUNS}/qwen3-4b_ab/final")
    ap.add_argument("--n", type=int, default=96)
    a = ap.parse_args()
    spec = SPECS[a.model]
    hg, g = head_grid(spec), spec.mlp_grid
    lo = spec.qk_grid[0]
    cfgs = [("full", ElasticConfig.full(spec)),
            ("w75", ElasticConfig.uniform(spec, 96, 96, n_h=hg[-1], d_mlp=g[2])),
            ("w50", ElasticConfig.uniform(spec, 64, 64, n_h=hg[-1], d_mlp=g[1])),
            ("min", ElasticConfig.uniform(spec, lo, lo, n_h=hg[0], d_mlp=g[0]))]
    MATH = {"full": 49.5, "w75": 10.0, "w50": 0.0, "min": 0.0}

    model, tok, order = load_supernet(a.ckpt, spec)
    web, _ = make_evaluator(model, tok, order, spec, heldout_texts(a.n))
    mth, nm = make_evaluator(model, tok, order, spec, math_texts(a.n), max_len=768)

    rows = {}
    for name, c in cfgs:
        rows[name] = {"web": web(c), "math": mth(c), "acc": MATH[name],
                      "w": c.weight_params(include_embed=False) /
                           ElasticConfig.full(spec).weight_params(include_embed=False)}
        print(f"  {name:5} web {rows[name]['web']:.3f}  math {rows[name]['math']:.3f}", flush=True)

    b = rows["full"]
    print(f"\n{'cfg':6} {'MATH':>7} {'web NLL':>9} {'Δweb':>7} {'math NLL':>9} {'Δmath':>7} "
          f"{'ratio':>7}")
    for k in ["full", "w75", "w50", "min"]:
        r = rows[k]
        dw, dm = r["web"] - b["web"], r["math"] - b["math"]
        print(f"{k:6} {r['acc']:6.1f}% {r['web']:9.3f} {dw:7.3f} {r['math']:9.3f} {dm:7.3f} "
              f"{(dm/dw if dw else 0):7.2f}x")
    print("\nratio > 1 means the math objective separates the configurations more sharply "
          "than web text\nat identical evaluation cost.")
    json.dump(rows, open(RUNS / f"objective_probe_{spec.key}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
