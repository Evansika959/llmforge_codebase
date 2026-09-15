"""Which benchmark actually separates supernet slices? Cheap ones first.

We do not need mathematics specifically -- any benchmark that DIFFERENTIATES architectures works
as ground truth. Likelihood-style benchmarks (MMLU, HellaSwag, ARC, WinoGrande, PIQA) score fixed
continuations instead of generating, so they cost 10-50x less than MATH or GSM8K.

But they cannot simply be assumed usable here. This project's central finding is that the failure
compounds over GENERATED tokens, and six teacher-forced likelihood metrics all misranked the same
configuration. A likelihood benchmark may inherit exactly that blindness, so the point of this
probe is to measure the SPREAD each benchmark shows across slices, and to check whether the cheap
ones agree with the generative one.

A benchmark is useful here if slices spread widely and do not floor at chance.
"""
import argparse, json
import torch

from llmforge.supernet.config import SPECS
from llmforge.supernet.elastic import build_pair_order, set_elastic_config
from llmforge.supernet.elastic.sampler import head_grid
from llmforge.supernet.paths import RUNS
from llmforge.supernet.search.nsga import load_supernet
from llmforge.supernet.space import ElasticConfig

CHANCE = {"mmlu": 25.0, "hellaswag": 25.0, "arc_challenge": 25.0,
          "winogrande": 50.0, "piqa": 50.0, "gsm8k": 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-1.7b")
    ap.add_argument("--ckpt", default=f"{RUNS}/qwen3-1.7b_ab/final")
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--tasks", default="hellaswag,arc_challenge,winogrande,piqa,mmlu,gsm8k")
    a = ap.parse_args()
    spec = SPECS[a.model]
    order = build_pair_order("hilo")
    hg, g, hd = head_grid(spec), spec.mlp_grid, spec.head_dim
    U = lambda q, h, m: ElasticConfig.uniform(spec, q, q, n_h=h, d_mlp=m)
    CFG = [("full", ElasticConfig.full(spec)),
           ("w85", U(96, hg[-1], g[3])),
           ("w70", U(96, hg[-1], g[2])),
           ("w55", U(64, hg[0], g[2])),
           ("min", U(32, hg[0], g[0]))]

    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    model, tok, _ = load_supernet(a.ckpt, spec)
    model.config.use_cache = True
    tasks = a.tasks.split(",")
    W0 = ElasticConfig.full(spec).weight_params(include_embed=False)

    res = {}
    for name, c in CFG:
        set_elastic_config(model, c.d_qk, c.d_v, order, n_h=c.n_h, d_mlp=c.d_mlp)
        r = simple_evaluate(model=HFLM(pretrained=model, tokenizer=tok, batch_size=16),
                            tasks=tasks, limit=a.limit, bootstrap_iters=0)
        row = {"w": c.weight_params(include_embed=False) / W0}
        for t in tasks:
            m = r["results"].get(t, {})
            v = next((x for k, x in m.items()
                      if k.split(",")[0] in ("acc", "acc_norm", "exact_match")
                      and "stderr" not in k and isinstance(x, float)), None)
            row[t] = None if v is None else v * 100
        res[name] = row
        print(f"  {name:5} W={row['w']:.3f} " +
              " ".join(f"{t}={row[t]:.1f}" if row[t] is not None else f"{t}=NA"
                       for t in tasks), flush=True)

    print(f"\n{'task':16} {'full':>7} {'min':>7} {'spread':>8} {'chance':>7}  useful?")
    for t in tasks:
        vs = [res[n][t] for n in res if res[n][t] is not None]
        if not vs:
            continue
        sp = max(vs) - min(vs)
        ch = CHANCE.get(t, 0.0)
        ok = "YES" if (sp >= 8 and max(vs) - ch >= 10) else "no"
        print(f"{t:16} {res['full'][t]:7.1f} {res['min'][t]:7.1f} {sp:8.1f} {ch:7.1f}  {ok}")
    json.dump(res, open(RUNS / f"bench_probe_{spec.key}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
