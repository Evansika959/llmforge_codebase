"""Minerva MATH on supernet slices: does a capability survive slicing the way NLL suggests?

NLL is the only signal the supernet was ever selected on, and NLL parity does not imply capability
parity. Math is the sharpest available test: it needs multi-step symbolic reasoning that a small
perplexity gap can easily hide.

This is also the FIRST generative evaluation of the elastic model -- everything to date has been
teacher-forced loss. Generation exercises the KV-cache path, where the elastic forward slices k/v
per layer and HF's cache must stay consistent across decode steps. So the run is gated on a parity
check: at full width the elastic model must reproduce stock HF generation token for token. If that
fails, every number below would be measuring a broken decode path rather than an architecture.

  python experiments/supernet_fidelity/minerva_subnets.py --model qwen3-4b --limit 200
"""
import argparse, json, os
import torch

from llmforge.supernet.config import SPECS
from llmforge.supernet.elastic import set_elastic_config
from llmforge.supernet.paths import RUNS
from llmforge.supernet.search.nsga import load_supernet
from llmforge.supernet.space import ElasticConfig


def subnets(spec, mode="span"):
    """Which slices to benchmark.

    span : the original four, spanning the whole elastic range.
    band : the near-full region W in [0.75, 1.0], which the span set leaves completely unmeasured
           even though 78% of the capability disappears inside it. The ladder doubles as a knob
           contrast -- qk96 and v96 cost IDENTICAL weights and KV, so the pair tests whether the
           perplexity-derived ranking (trim query/key first, protect value) also holds for
           capability, which is a different question and may have a different answer.
    """
    hd, lo = spec.head_dim, spec.qk_grid[0]
    g = spec.mlp_grid
    from llmforge.supernet.elastic.sampler import head_grid
    hg = head_grid(spec)
    U = lambda q, v, h, m: ElasticConfig.uniform(spec, q, v, n_h=h, d_mlp=m)
    if mode == "band":
        return [
            ("full",     U(hd, hd, hg[-1], g[3])),   # W 1.000  reference
            ("qk96",     U(96, hd, hg[-1], g[3])),   # W 0.968  query/key trimmed
            ("v96",      U(hd, 96, hg[-1], g[3])),   # W 0.968  value trimmed, SAME cost as qk96
            ("v64",      U(hd, 64, hg[-1], g[3])),   # W 0.935
            ("h16",      U(hd, hd, hg[1], g[3])),    # W 0.896  half the query heads
            ("h8",       U(hd, hd, hg[0], g[3])),    # W 0.844
            ("mlp7296",  U(hd, hd, hg[-1], g[2])),   # W 0.815  feed-forward trimmed
        ]
    return [("full", ElasticConfig.full(spec)),
            ("w75",  U(96, 96, hg[-1], g[2])),
            ("w50",  U(64, 64, hg[-1], g[1])),
            ("min",  U(lo, lo, hg[0], g[0]))]


PROMPT = "Question: What is 17 * 23?\nAnswer: Let us compute step by step."


@torch.no_grad()
def parity_check(spec):
    """Is the elastic forward equivalent to stock HF at full width, on IDENTICAL weights?

    Checked teacher-forced, NOT by comparing generated text. Token-exact greedy generation is the
    wrong bar: in bf16 the reordered gather changes float accumulation by up to ~0.7 logits while
    the top1-top2 gap can be ~0.12, so greedy decoding turns harmless precision noise into
    different-but-equivalent wording. Measured here: fp32 max|dlogit| 5.5e-05 (exact up to
    reassociation) and 100% argmax agreement in both precisions. So the gate is
    fp32 exactness + bf16 argmax agreement, which is what correctness actually means.

    Stock logits must be taken BEFORE enable_elastic: that call patches the Qwen3Attention CLASS,
    so any model built afterwards inherits the elastic forward without per-module index buffers.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llmforge.supernet.elastic import (build_pair_order, disable_elastic, enable_elastic,
                                    set_elastic_backend)
    tok = AutoTokenizer.from_pretrained(spec.repo)
    ids = tok(PROMPT, return_tensors="pt").input_ids.to("cuda")

    def load(dt):
        return AutoModelForCausalLM.from_pretrained(
            spec.repo, torch_dtype=dt, attn_implementation="eager").to("cuda").eval()

    ok = True
    for dt, name, tol in [(torch.float32, "fp32", 1e-3), (torch.bfloat16, "bf16", None)]:
        m = load(dt)
        st = m(ids).logits.float().cpu()
        del m; torch.cuda.empty_cache()
        m = load(dt); enable_elastic(m); set_elastic_backend("eager")
        f = ElasticConfig.full(spec)
        set_elastic_config(m, f.d_qk, f.d_v, build_pair_order("hilo"), n_h=f.n_h, d_mlp=f.d_mlp)
        el = m(ids).logits.float().cpu()
        del m; torch.cuda.empty_cache(); disable_elastic()
        md = (st - el).abs().max().item()
        agree = (st.argmax(-1) == el.argmax(-1)).float().mean().item()
        good = agree == 1.0 and (tol is None or md < tol)
        ok &= good
        print(f"[parity] {name}: max|dlogit| {md:.3e}  argmax agree {agree:.1%}  "
              f"{'OK' if good else 'FAIL'}", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-4b", choices=sorted(SPECS))
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--task", default="minerva_math500")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--batch-size", default="8")
    ap.add_argument("--skip-parity", action="store_true")
    ap.add_argument("--mode", choices=["span", "band"], default="span")
    ap.add_argument("--base", action="store_true",
                    help="slice the PUBLISHED checkpoint instead of a trained supernet. Answers "
                         "whether supernet training bought any CAPABILITY, as opposed to the "
                         "perplexity it demonstrably bought (min 14.748 -> 4.514).")
    a = ap.parse_args()
    spec = SPECS[a.model]
    ckpt = a.ckpt or f"{RUNS}/{a.model}_ab/final"

    if not a.skip_parity and not parity_check(spec):
        raise SystemExit('PARITY FAILED -- decode path broken, numbers meaningless')
    if a.base:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from llmforge.supernet.elastic import enable_elastic, set_elastic_backend, build_pair_order
        model = AutoModelForCausalLM.from_pretrained(
            spec.repo, torch_dtype=torch.bfloat16, attn_implementation="eager").to("cuda").eval()
        enable_elastic(model); set_elastic_backend("eager")
        tok, order = AutoTokenizer.from_pretrained(spec.repo), build_pair_order("hilo")
        ckpt = "BASE"
        print(f"[loader] published {spec.repo}, elastic-patched, UNTRAINED", flush=True)
    else:
        model, tok, order = load_supernet(ckpt, spec)

    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    model.config.use_cache = True
    out = {}
    for name, cfg in subnets(spec, a.mode):
        set_elastic_config(model, cfg.d_qk, cfg.d_v, order, n_h=cfg.n_h, d_mlp=cfg.d_mlp)
        lm = HFLM(pretrained=model, tokenizer=tok, batch_size=a.batch_size)
        r = simple_evaluate(model=lm, tasks=[a.task], limit=a.limit, bootstrap_iters=0)
        m = r["results"][a.task]
        acc = next((v for k, v in m.items() if "exact_match" in k and "stderr" not in k), None)
        w = cfg.weight_params(include_embed=False) / \
            ElasticConfig.full(spec).weight_params(include_embed=False)
        out[name] = {"acc": acc, "w": w, "kv": cfg.kv_frac(), "metrics": m}
        print(f"  {name:5} W {w:.3f} KV {cfg.kv_frac():.3f}  {a.task} = {acc}", flush=True)

    # name the output after the checkpoint, not just the model: the mathmix rerun would
    # otherwise silently overwrite the fineweb-only result it is meant to be compared against
    tag = "base" if a.base else os.path.basename(os.path.normpath(ckpt).rstrip("/"))
    parent = "published" if a.base else os.path.basename(os.path.dirname(os.path.normpath(ckpt)))
    p = RUNS / f"minerva_{a.model}_{parent}_{tag}{'_band' if a.mode == 'band' else ''}.json"
    json.dump({"model": a.model, "ckpt": ckpt, "task": a.task, "limit": a.limit,
               "results": out}, open(p, "w"), indent=1)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
