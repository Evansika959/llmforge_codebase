"""Score the ground-truth architectures as SUPERNET SLICES -- the predictor side of rank fidelity.

Pairs with experiments/supernet_fidelity/groundtruth.py, which trains the same architectures on their own weights.
The correlation between the two is the whole question: can slicing order architectures the way
dedicated training does?

Benchmark choice matters more than raw spread suggests. On the 5-point probe MMLU had the widest
range (59.0 -> 22.8) but most of that is the fall TO CHANCE: at W <= 0.75 it sits at 23 against a
25% baseline and carries no information. arc_challenge and winogrande floor the same way. Only
hellaswag (45.2 -> 28.5 against 25) and piqa (74.0 -> 54.0 against 50) stay above chance and
monotone across the whole range, and our architectures span W 0.375-0.875 -- mostly inside the
region where MMLU is already dead. hellaswag and piqa are therefore primary; the rest are kept
for reference and to show the flooring.
"""
import argparse, json, pathlib
import torch

from llmforge.supernet.config import SPECS
from llmforge.supernet.elastic import build_pair_order, set_elastic_config
from llmforge.supernet.paths import RUNS
from llmforge.supernet.search.nsga import load_supernet
from llmforge.supernet.space import ElasticConfig

CHANCE = {"hellaswag": 25.0, "minerva_math500": 0.0, "piqa": 50.0, "arc_challenge": 25.0,
          "mmlu": 25.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-1.7b")
    ap.add_argument("--ckpt", default=f"{RUNS}/qwen3-1.7b_ab/final")
    ap.add_argument("--tasks", default="hellaswag,minerva_math500")
    ap.add_argument("--out", default=None, help="separate file per supernet version")
    ap.add_argument("--only", default=None,
                    help="comma-separated config names; 'full' plus a few slices is enough to ask "
                         "whether a recipe change preserved capability, and the full 23 would take "
                         "six hours per version")
    ap.add_argument("--limit", type=int, default=0,
                    help="0 = full set. HellaSwag's within-cost-level signal is ~1.4 points and "
                         "at limit=500 the two-model detection threshold is 5.9, so a truncated "
                         "run measures noise. The full 10,042 brings it to 1.3.")
    a = ap.parse_args()
    spec = SPECS[a.model]
    order = build_pair_order("hilo")

    import sys
    sys.argv = [sys.argv[0]]
    from llmforge.supernet.eval.archsets import arch_set
    archs = arch_set(spec)
    W0 = ElasticConfig.full(spec).weight_params(include_embed=False)

    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    model, tok, _ = load_supernet(a.ckpt, spec)
    model.config.use_cache = True
    tasks = a.tasks.split(",")

    out_path = pathlib.Path(a.out) if a.out else (RUNS / f"slice_scores_{spec.key}.json")
    res = json.load(open(out_path)) if out_path.exists() else {}
    items = [("full", None)] + list(archs.items())
    if a.only:
        want = set(a.only.split(","))
        items = [(n, A) for n, A in items if n in want]
    for name, A in items:
        if name in res:
            print(f"  {name}: done, skip", flush=True); continue
        c = (ElasticConfig.full(spec) if A is None else
             ElasticConfig.uniform(spec, A["d_qk"], A["d_qk"], n_h=A["n_h"], d_mlp=A["d_mlp"]))
        set_elastic_config(model, c.d_qk, c.d_v, order, n_h=c.n_h, d_mlp=c.d_mlp)
        r = simple_evaluate(model=HFLM(pretrained=model, tokenizer=tok, batch_size=16),
                            tasks=tasks, limit=(a.limit or None), bootstrap_iters=1000)
        row = {"w": c.weight_params(include_embed=False) / W0, "kv": c.kv_frac()}
        for t in tasks:
            m = r["results"].get(t, {})
            v = next((x for k, x in m.items()
                      if k.split(",")[0] in ("acc_norm", "acc", "exact_match")
                      and "stderr" not in k and isinstance(x, float)), None)
            se = next((x for k, x in m.items() if "stderr" in k and isinstance(x, float)), None)
            row[t + "_se"] = None if se is None else se * 100
            row[t] = None if v is None else v * 100
        res[name] = row
        json.dump(res, open(out_path, "w"), indent=1)
        print(f"  {name:6} W={row['w']:.3f} " +
              " ".join(f"{t[:4]}={row[t]:.1f}" for t in tasks if row[t] is not None), flush=True)

    print(f"\n{'task':14} {'range over the 16':>18} {'chance':>7}  usable range?")
    for t in tasks:
        vs = [res[k][t] for k in res if k != "full" and res[k].get(t) is not None]
        if not vs: continue
        above = sum(1 for v in vs if v > CHANCE[t] + 3)
        print(f"{t:14} {min(vs):7.1f} - {max(vs):5.1f} {CHANCE[t]:7.1f}  "
              f"{above}/{len(vs)} above chance")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
