"""Does our supernet implementation actually do what we claim? Three checks never run before.

Every capability number in this project came from CACHED GENERATION at REDUCED width. The only
correctness test we ever ran was full width, teacher-forced, no cache. So the entire conclusion
"slicing destroys capability" rests on a code path that was never verified.

T1  cache vs no cache, at reduced width. Incremental decoding must reproduce the teacher-forced
    logits. If it does not, every Minerva number is measuring a broken decoder.
T2  sliced elastic model vs an INDEPENDENTLY BUILT dense model of the same shape. Copy the
    corresponding weights into a real Qwen3 and compare. This is the definitive check: it tests
    our slicing against HuggingFace's own attention rather than against our own assumptions.
    Two subtleties, both handled, both of which would otherwise produce a false failure:
      - a native head_dim=D model derives RoPE frequencies for D dims; our slice keeps D of the
        128-dim model's pairs, which is a DIFFERENT frequency set. The reference model's inv_freq
        is overridden with the selected subset.
      - the elastic model folds a learned temperature into q. It is folded into the reference's
        q_proj weights instead.
T3  GQA grouping under head reduction: every surviving query head must still read its own KV group.

  python experiments/supernet_fidelity/validate_slicing.py --model qwen3-4b --ckpt runs/supernet/qwen3-4b_ab/final
"""
import argparse, glob, os
import torch
import torch.nn.functional as F

from llmforge.supernet.config import SPECS
from llmforge.supernet.elastic import set_elastic_config, build_pair_order
from llmforge.supernet.elastic.attention import head_index, qk_gather_index
from llmforge.supernet.search.nsga import load_supernet
from llmforge.supernet.space import ElasticConfig
from llmforge.supernet.paths import RUNS

OK = lambda b: "\033[32mPASS\033[0m" if b else "\033[31mFAIL\033[0m"


def prompt_ids(tok, dev, n=48):
    t = ("The Fibonacci sequence begins 1, 1, 2, 3, 5, 8. Each term is the sum of the two "
         "preceding terms, and the ratio between consecutive terms approaches the golden ratio.")
    return tok(t, return_tensors="pt").input_ids[:, :n].to(dev)


@torch.no_grad()
def t1_cache(model, tok, order, spec, cfgs):
    """Cached incremental decoding must equal a single teacher-forced pass."""
    print("\n=== T1: KV-cache path at reduced width ===", flush=True)
    ids = prompt_ids(tok, "cuda")
    allok = True
    for name, c in cfgs:
        set_elastic_config(model, c.d_qk, c.d_v, order, n_h=c.n_h, d_mlp=c.d_mlp)
        model.config.use_cache = False
        ref = model(input_ids=ids).logits[:, -1].float()
        model.config.use_cache = True
        out = model(input_ids=ids[:, :-1], use_cache=True)
        step = model(input_ids=ids[:, -1:], past_key_values=out.past_key_values,
                     use_cache=True).logits[:, -1].float()
        d = (ref - step).abs().max().item()
        agree = (ref.argmax(-1) == step.argmax(-1)).all().item()
        good = agree and d < 0.5      # bf16 accumulation differs between the two paths
        allok &= good
        print(f"  {name:9} max|dlogit| {d:.3e}  argmax match {bool(agree)}  {OK(good)}", flush=True)
    model.config.use_cache = False
    return allok


@torch.no_grad()
def t2_standalone(spec, ckpt, order, cfg, name):
    """Rebuild the slice as a real dense Qwen3 and compare. Independent of our own code."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from transformers.models.qwen3 import modeling_qwen3 as mq
    dqk, dv, nh, dm = cfg.d_qk[0], cfg.d_v[0], cfg.n_h[0], cfg.d_mlp[0]
    assert dqk == dv, "reference model needs one head_dim, so d_qk must equal d_v"

    model, tok, _ = load_supernet(ckpt, spec)
    model = model.float()   # fp32: bf16 kernel-order noise otherwise swamps a real discrepancy
    set_elastic_config(model, cfg.d_qk, cfg.d_v, order, n_h=cfg.n_h, d_mlp=cfg.d_mlp)
    ids = prompt_ids(tok, "cuda")
    ours = model(input_ids=ids).logits.float().cpu()
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    rope_inv = model.model.rotary_emb.inv_freq.clone()
    del model; torch.cuda.empty_cache()

    n_q, n_kv, hd = spec.n_q, spec.n_kv, spec.head_dim
    idx = qk_gather_index(order, dqk)
    qi = head_index(n_q, n_kv, nh) if nh != n_q else torch.arange(n_q)

    # enable_elastic patched the Qwen3Attention CLASS, so a freshly built model would inherit the
    # elastic forward without any config and die on _qk_idx. Restore stock behaviour first -- the
    # whole point is to compare against HuggingFace's own implementation, not ours.
    from llmforge.supernet.elastic import disable_elastic
    disable_elastic()

    cf = AutoConfig.from_pretrained(spec.repo)
    cf.head_dim, cf.num_attention_heads, cf.intermediate_size = dqk, nh, dm
    cf.num_key_value_heads = n_kv
    cf._attn_implementation = "eager"   # match our backend; SDPA accumulates differently
    ref = AutoModelForCausalLM.from_config(cf, torch_dtype=torch.float32).to("cuda").eval()

    # RoPE: a native head_dim=dqk model would use its OWN frequencies. Ours keeps a subset of the
    # 128-dim model's pairs, so transplant those or the comparison fails for the wrong reason.
    # qk_gather_index lays the head out as [first-halves of selected pairs | second-halves], the
    # convention HF rotate_half expects. So the frequency index of a selected pair IS the entry in
    # the first half -- no division. (Dividing by 2 assumes the interleaved (2i, 2i+1) convention,
    # which Qwen3 does not use; that mistake made even full width fail.)
    sel = idx[: dqk // 2].to("cuda")
    ref.model.rotary_emb.inv_freq = rope_inv.to("cuda")[sel]
    if hasattr(ref.model.rotary_emb, "original_inv_freq"):
        ref.model.rotary_emb.original_inv_freq = ref.model.rotary_emb.inv_freq

    new = {}
    for k, v in sd.items():
        if "logit_scale" in k:
            continue
        if k.endswith("q_proj.weight"):
            new[k] = v.view(n_q, hd, -1)[qi][:, idx, :].reshape(nh * dqk, -1)
        elif k.endswith("k_proj.weight"):
            new[k] = v.view(n_kv, hd, -1)[:, idx, :].reshape(n_kv * dqk, -1)
        elif k.endswith("v_proj.weight"):
            new[k] = v.view(n_kv, hd, -1)[:, :dv, :].reshape(n_kv * dv, -1)
        elif k.endswith("o_proj.weight"):
            new[k] = v.view(-1, n_q, hd)[:, qi, :dv].reshape(-1, nh * dv)
        elif k.endswith("q_norm.weight"):
            # the learned temperature must be folded in HERE, not into q_proj: q_norm is RMSNorm
            # and therefore scale-invariant, so a factor placed on q_proj is normalised straight
            # back out. q_norm.weight multiplies AFTER normalisation, which is where the elastic
            # forward applies it.
            from llmforge.supernet.elastic.attention import grid_index
            L = int(k.split(".")[2])
            w = v[idx]
            ls = sd.get(f"model.layers.{L}.self_attn.logit_scale")
            if ls is not None:
                w = w * torch.exp(ls[grid_index(dqk, spec.head_dim)].float()).to(w.dtype)
            new[k] = w
        elif k.endswith("k_norm.weight"):
            new[k] = v[idx]
        elif k.endswith("gate_proj.weight") or k.endswith("up_proj.weight"):
            new[k] = v[:dm]
        elif k.endswith("down_proj.weight"):
            new[k] = v[:, :dm]
        else:
            new[k] = v
    miss, unexp = ref.load_state_dict(new, strict=False)
    theirs = ref(input_ids=ids).logits.float().cpu()
    del ref; torch.cuda.empty_cache()

    d = (ours - theirs).abs().max().item()
    agree = (ours.argmax(-1) == theirs.argmax(-1)).float().mean().item()
    good = agree == 1.0 and d < 1e-2      # fp32: only float reassociation should remain
    print(f"  {name:22} max|dlogit| {d:.3e}  argmax agree {agree:.1%}  "
          f"missing={len(miss)} unexpected={len(unexp)}  {OK(good)}", flush=True)
    return good


def t3_grouping(spec):
    """Each surviving query head must still be paired with the KV group it belongs to."""
    print("\n=== T3: GQA grouping under head reduction ===", flush=True)
    n_q, n_kv = spec.n_q, spec.n_kv
    from llmforge.supernet.elastic.sampler import head_grid
    allok = True
    for nh in head_grid(spec):
        if nh == n_q:
            continue
        qi = head_index(n_q, n_kv, nh)
        groups_after = (torch.arange(nh) // (nh // n_kv)).tolist()   # what repeat_kv will pair
        groups_true = (qi // (n_q // n_kv)).tolist()                 # the group each head is in
        good = groups_after == groups_true
        allok &= good
        print(f"  n_h={nh:3}  heads {qi.tolist()[:6]}...  pairing correct {OK(good)}", flush=True)
    return allok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-4b")
    ap.add_argument("--ckpt", default=f"{RUNS}/qwen3-4b_ab/final")
    a = ap.parse_args()
    spec = SPECS[a.model]
    order = build_pair_order("hilo")
    from llmforge.supernet.elastic.sampler import head_grid
    hg, g = head_grid(spec), spec.mlp_grid
    U = lambda q, v, h, m: ElasticConfig.uniform(spec, q, v, n_h=h, d_mlp=m)
    cfgs = [("full", ElasticConfig.full(spec)),
            ("qk96", U(96, 128, hg[-1], g[3])),
            ("v64", U(128, 64, hg[-1], g[3])),
            ("h16", U(128, 128, hg[1], g[3])),
            ("w75", U(96, 96, hg[-1], g[2])),
            ("w50", U(64, 64, hg[-1], g[1])),
            ("min", U(32, 32, hg[0], g[0]))]

    r3 = t3_grouping(spec)
    model, tok, _ = load_supernet(a.ckpt, spec)
    r1 = t1_cache(model, tok, order, spec, cfgs)
    del model; torch.cuda.empty_cache()

    print("\n=== T2: sliced elastic vs independently built dense model ===", flush=True)
    r2 = True
    for nm, c in [("full 128/32/9728", ElasticConfig.full(spec)),
                  ("w75  96/32/7296", U(96, 96, hg[-1], g[2])),
                  ("w50  64/32/4864", U(64, 64, hg[-1], g[1])),
                  ("h16  128/16/9728", U(128, 128, hg[1], g[3]))]:
        r2 &= t2_standalone(spec, a.ckpt, order, c, nm)

    print(f"\n{'='*60}\nT1 cache path      {OK(r1)}\nT2 standalone      {OK(r2)}\n"
          f"T3 GQA grouping    {OK(r3)}\n{'='*60}", flush=True)


if __name__ == "__main__":
    main()
