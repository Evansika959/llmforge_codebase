"""Evaluate the dedicated-training reference models.

These are plain dense checkpoints, so no elastic patching is involved. Two measurements per model:

  held-out CE   on the SAME documents the supernet probes use, so slice and reference are
                directly comparable
  Minerva MATH  the capability number. Required rather than optional: six cheap likelihood
                metrics (CE, p(target), mean/median target rank, top-1 accuracy, top-1 agreement)
                were tested against capability and ALL SIX misrank the same configuration, so a
                likelihood proxy cannot stand in here.

Resumable -- an architecture with a result file is skipped.
"""
import argparse, glob, json, os
import torch


def restore_rope(model, d, parent_head_dim=None):
    """HF stores rotary inv_freq as a NON-PERSISTENT buffer: save_pretrained drops it and
    from_pretrained recomputes it natively from head_dim. Our extracted models keep a SUBSET of
    the 128-dim parent's frequency pairs, which is a different set entirely -- for head_dim=64 the
    second entry should be 1.24e-06 and the native recompute gives 0.649, five orders out. The
    model then loads, runs, and returns confident nonsense: held-out CE 6.0 against 2.4 at the end
    of its own training, and 0% on every capability benchmark.

    Recomputed here from the architecture record rather than re-saving 16 checkpoints.

    The parent's head_dim is NOT recoverable from the extracted model -- `config.head_dim` is the
    child's d_qk. It used to be hard-coded to 128, which is right for Qwen3 and silently wrong for
    SmolLM2: every index stays in range, so there is no exception, just an inv_freq up to 14.9x off
    at d_qk 64. Runs record it in arch.json; older Qwen3 records predate the field and default to
    128, which is what they were.
    """
    from llmforge.supernet.elastic.attention import build_pair_order, qk_gather_index
    if parent_head_dim is None:
        mp = os.path.join(d, "arch.json")
        parent_head_dim = (json.load(open(mp)).get("parent_head_dim") if os.path.exists(mp)
                           else None)
    if parent_head_dim is None:
        parent_head_dim = 128
        print(f"[restore_rope] {d}: arch.json has no parent_head_dim; assuming 128 (Qwen3). "
              f"This is WRONG for any SmolLM2-derived reference model.", flush=True)
    P = int(parent_head_dim)
    hd = model.config.head_dim
    idx = qk_gather_index(build_pair_order("hilo", P // 2), hd)
    base = 1.0 / (model.config.rope_theta ** (torch.arange(0, P, 2, dtype=torch.float32) / P))
    want = base[idx[: hd // 2]]
    re = model.model.rotary_emb
    re.inv_freq = want.to(re.inv_freq.device, re.inv_freq.dtype)
    if hasattr(re, "original_inv_freq"):
        re.original_inv_freq = re.inv_freq
    return model
import torch.nn.functional as F

from llmforge.supernet.config import SPECS
# The package loader prefers the frozen copy in assets/heldout, so reference models are scored
# on exactly the documents the supernet probes use.
from llmforge.supernet.data.heldout import heldout_texts
from llmforge.supernet.paths import RUNS


@torch.no_grad()
def heldout_ce(model, tok, n=96, max_len=768, boots=1000):
    """Token-weighted mean NLL, plus a bootstrap SE over evaluation units.

    n defaults to 96 to match `slice_nll.py`. The two arms must be scored on the same documents
    or the comparison inherits a difference in eval set on top of the difference in training,
    and a rank correlation cannot tell the two apart.

    max_len was pinned at 768 with no way to change it, which is the regime
    dominated by local prediction, where attention damage barely shows -- and the pairs this
    ground truth scores differ 2-4x in KV cache. Above 768 the documents are CONCATENATED into
    windows, exactly as `slice_nll.py` does, because truncating per document here instead would
    score the predictor and the thing it predicts on different token groupings: cross-document
    context lowers NLL in one and not the other. Note the unit of the bootstrap changes with it
    -- 96 documents at 768, but 58 windows at 1024 and 14 at 4096 over the same 60k tokens.

    The SE resamples those units, not tokens: tokens inside one are strongly dependent, so a
    token-level interval would be several times too narrow and would mark pairs as resolvable
    that are not. It is what decides which architecture pairs the ground truth can separate.
    """
    import numpy as np
    texts = heldout_texts(n)
    if max_len > 768:
        flat = []
        for t in texts:
            flat += tok(t, add_special_tokens=False).input_ids
        n_win = max(1, len(flat) // max_len)
        ids = [torch.tensor(flat[i * max_len:(i + 1) * max_len], device="cuda").unsqueeze(0)
               for i in range(n_win)]
    else:
        ids = [tok(t, return_tensors="pt", truncation=True, max_length=max_len).input_ids.to("cuda")
               for t in texts]
    ids = [i for i in ids if i.shape[1] >= 8]
    per = []
    for i in ids:
        lg = model(input_ids=i).logits[:, :-1].float()
        t = i[:, 1:]
        per.append((F.cross_entropy(lg.reshape(-1, lg.shape[-1]), t.reshape(-1),
                                    reduction="sum").item(), float(t.numel())))
    tot = sum(s for s, _ in per)
    cnt = sum(c for _, c in per)
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(boots):
        k = rng.integers(0, len(per), len(per))
        bs.append(sum(per[j][0] for j in k) / sum(per[j][1] for j in k))
    # Per-unit sums/counts, so a DIFFERENCE between two architectures can be bootstrapped paired.
    # Every architecture is scored on the same windows in the same order; the unpaired
    # hypot(se_i, se_j) overstates a difference's uncertainty by 5-22x here, which is enough to
    # mark resolvable pairs as ties. slice_nll.py already dumps its side; this is the other half.
    return tot / cnt, float(np.std(bs)), len(per), [(float(a_), float(b_)) for a_, b_ in per]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=f"{RUNS}/groundtruth")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--batch-size", default="8")
    ap.add_argument("--tasks", default="hellaswag,minerva_math500")
    ap.add_argument("--n-texts", type=int, default=96, help="held-out docs; must match slice_nll.py")
    ap.add_argument("--ctx", type=int, default=768,
                    help="token cap per held-out document. MUST match the slice side "
                         "(slice_nll.py --ctx). 768 is the historical default and is the "
                         "regime where attention damage barely shows; use 1024 and 4096.")
    ap.add_argument("--nll-only", action="store_true",
                    help="skip the benchmark suite. The predictor side is scored as NLL, and both "
                         "arms must use the SAME metric, so this is the fast path that actually "
                         "feeds rank_fidelity.py --nll")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    out_path = a.out or (RUNS / ("groundtruth_nll.json" if a.nll_only
                                                else "groundtruth_eval.json"))
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    up = str(out_path).replace('.json', '_units.json')
    units = json.load(open(up)) if os.path.exists(up) else {}
    for d in sorted(glob.glob(os.path.join(a.dir, "*"))):
        arch = os.path.basename(d)
        if not os.path.exists(os.path.join(d, "arch.json")):
            continue
        if arch in res:
            print(f"  {arch}: already evaluated, skip", flush=True)
            continue
        meta = json.load(open(os.path.join(d, "arch.json")))
        model = AutoModelForCausalLM.from_pretrained(
            d, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
        tok = AutoTokenizer.from_pretrained(d)
        restore_rope(model, d, meta.get("parent_head_dim"))
        model.config.use_cache = False
        ce, se, ndoc, per_unit = heldout_ce(model, tok, a.n_texts, max_len=a.ctx)
        if a.nll_only:
            res[arch] = {"nll": ce, "se": se, "n_docs": ndoc, **meta}
            units[arch] = per_unit
            print(f"  {arch:6} W={meta['w']:.3f} nll={ce:.4f} +-{se:.4f} ({ndoc} docs)", flush=True)
            json.dump(res, open(out_path, "w"), indent=1)
            json.dump(units, open(str(out_path).replace(".json", "_units.json"), "w"))
            del model
            torch.cuda.empty_cache()
            continue
        model.config.use_cache = True
        lm = HFLM(pretrained=model, tokenizer=tok, batch_size=a.batch_size)
        rr = {}
        for t in a.tasks.split(","):
            lim = a.limit if t == 'minerva_math500' else None   # HellaSwag needs the full set
            g = simple_evaluate(model=lm, tasks=[t], limit=lim, bootstrap_iters=1000)
            rr.update(g["results"])
        m = rr.get('minerva_math500', {})
        em = next((v for k, v in m.items() if k.startswith("exact_match") and "stderr" not in k), None)
        mv = next((v for k, v in m.items() if k.startswith("math_verify") and "stderr" not in k), None)
        hs = rr.get("hellaswag", {})
        hv = next((v for k, v in hs.items() if k.split(",")[0] in ("acc_norm", "acc")
                   and "stderr" not in k and isinstance(v, float)), None)
        res[arch] = {**meta, "heldout_ce": ce, "exact_match": em, "math_verify": mv,
                     "hellaswag": None if hv is None else hv * 100}
        print(f"  {arch:6} W={meta['w']:.3f} ce={ce:.3f} hella={res[arch]['hellaswag']} "
              f"exact={em} verify={mv}", flush=True)
        json.dump(res, open(out_path, "w"), indent=1)
        del model
        torch.cuda.empty_cache()
    print(f"\nwrote {out_path}  ({len(res)} architectures)")


if __name__ == "__main__":
    main()
