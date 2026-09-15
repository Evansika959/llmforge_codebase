"""Is 'calibration' literally the whole story? Fit one scalar temperature and see.

Claim under test: supernet training mainly repaired the SCALE of the output distribution, not
which tokens it ranks. If so, fitting a single temperature to the naively-truncated base should
recover most of the cross-entropy gap to the trained supernet -- while leaving the token RANKING
(top-1 agreement with the full model) untouched, because a temperature cannot reorder anything.

Three quantities per config:
  CE            cross-entropy, sensitive to both ranking and scale
  CE(T*)        cross-entropy after fitting the single best temperature -- scale removed
  top-1 agree   fraction of positions whose argmax matches the FULL model: pure ranking,
                completely invariant to temperature
"""
import argparse
import torch
import torch.nn.functional as F

from llmforge.supernet.config import SPECS
from llmforge.supernet.elastic import set_elastic_config, build_pair_order
from llmforge.supernet.elastic.sampler import head_grid
from llmforge.supernet.data.heldout import heldout_texts
from llmforge.supernet.search.nsga import load_supernet
from llmforge.supernet.space import ElasticConfig
from llmforge.supernet.paths import RUNS


@torch.no_grad()
def collect(model, order, cfg, ids):
    set_elastic_config(model, cfg.d_qk, cfg.d_v, order, n_h=cfg.n_h, d_mlp=cfg.d_mlp)
    lg, tg = [], []
    for i in ids:
        lg.append(model(input_ids=i).logits[:, :-1].float().cpu())
        tg.append(i[:, 1:].cpu())
    return lg, tg


def ce_at_T(lg, tg, T):
    tot = n = 0.0
    for a, b in zip(lg, tg):
        x = (a / T).reshape(-1, a.shape[-1])
        tot += F.cross_entropy(x, b.reshape(-1), reduction="sum").item()
        n += b.numel()
    return tot / n


def fit_T(lg, tg):
    lo, hi = 0.2, 5.0
    for _ in range(40):                      # golden-section on a 1-D convex-ish objective
        m1, m2 = lo + (hi - lo) * .382, lo + (hi - lo) * .618
        if ce_at_T(lg, tg, m1) < ce_at_T(lg, tg, m2): hi = m2
        else: lo = m1
    T = (lo + hi) / 2
    return T, ce_at_T(lg, tg, T)


def top1(lg, ref):
    ok = n = 0
    for a, b in zip(lg, ref):
        ok += (a.argmax(-1) == b.argmax(-1)).sum().item(); n += b[..., 0].numel()
    return ok / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-4b")
    ap.add_argument("--n", type=int, default=16)
    a = ap.parse_args()
    spec = SPECS[a.model]; order = build_pair_order("hilo")
    hg, g = head_grid(spec), spec.mlp_grid
    U = lambda q, v, h, m: ElasticConfig.uniform(spec, q, v, n_h=h, d_mlp=m)
    CFG = [("qk96", U(96, 128, hg[-1], g[3])), ("v64", U(128, 64, hg[-1], g[3])),
           ("w75", U(96, 96, hg[-1], g[2])), ("mlp7296", U(128, 128, hg[-1], g[2]))]

    out = {}
    for tag, ck in (("base", None), ("trained", f"{RUNS}/qwen3-4b_ab/final")):
        if ck is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from llmforge.supernet.elastic import enable_elastic, set_elastic_backend
            model = AutoModelForCausalLM.from_pretrained(
                spec.repo, torch_dtype=torch.bfloat16, attn_implementation="eager").to("cuda").eval()
            enable_elastic(model); set_elastic_backend("eager")
            tok = AutoTokenizer.from_pretrained(spec.repo)
        else:
            model, tok, _ = load_supernet(ck, spec)
        model.config.use_cache = False
        texts = heldout_texts(a.n)
        ids = [tok(t, return_tensors="pt", truncation=True, max_length=768).input_ids.to("cuda")
               for t in texts]
        full_lg, _ = collect(model, order, ElasticConfig.full(spec), ids)
        for nm, c in CFG:
            lg, tg = collect(model, order, c, ids)
            T, ceT = fit_T(lg, tg)
            out[(tag, nm)] = (ce_at_T(lg, tg, 1.0), T, ceT, top1(lg, full_lg))
        del model; torch.cuda.empty_cache()

    print(f"\n{'cfg':9} | {'base truncated':^34} | {'trained supernet':^34}")
    print(f"{'':9} | {'CE':>7} {'T*':>5} {'CE(T*)':>8} {'top1':>9} | "
          f"{'CE':>7} {'T*':>5} {'CE(T*)':>8} {'top1':>9}")
    print("-" * 84)
    for nm, _ in CFG:
        b = out[("base", nm)]; t = out[("trained", nm)]
        print(f"{nm:9} | {b[0]:7.3f} {b[1]:5.2f} {b[2]:8.3f} {b[3]:8.1%} | "
              f"{t[0]:7.3f} {t[1]:5.2f} {t[2]:8.3f} {t[3]:8.1%}")
    print("\nCE(T*) close between base and trained  => training mostly fixed SCALE (calibration)")
    print("top1 much higher for trained           => training also fixed RANKING (real repair)")


if __name__ == "__main__":
    main()
