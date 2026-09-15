"""Ground truth for rank fidelity: train each architecture on its OWN weights.

The supernet is only useful as a NAS predictor if slicing ORDERS architectures the way dedicated
training would. That has never been measured; every conclusion about the supernet as an evaluator
rests on it. This builds the reference: extract an architecture to a standalone dense model and
fine-tune it with its own weights, so the only difference from the supernet slice is weight
SHARING.

Same recipe as the supernet (CE + KD from the frozen base) minus elasticity, so the comparison
isolates sharing rather than confounding it with a different objective.

Architectures are chosen in COST LEVELS with several shapes per level. Ranking architectures of
very different size is trivial (bigger wins); the question that matters for search is whether the
supernet can order architectures that cost the SAME.

  python experiments/supernet_fidelity/groundtruth.py --list                  # show the selection
  python experiments/supernet_fidelity/groundtruth.py --arch L2_a --steps 750 # train one
"""
import argparse, glob, json, os, time
import torch
import torch.nn.functional as F

from llmforge.supernet.config import SPECS
from llmforge.supernet.data.datamix import DataMix
from llmforge.supernet.elastic import build_pair_order, pair_order_for
from llmforge.supernet.elastic.sampler import head_grid
from llmforge.supernet.extract import extract_dense
from llmforge.supernet.paths import RUNS


# arch_set lives in the package so sibling scripts import it without path hacks.
from llmforge.supernet.eval.archsets import arch_set  # noqa: F401

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-4b")
    ap.add_argument("--arch", default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--shape", default=None,
                    help="d_qk,n_h,d_mlp -- train this exact uniform shape instead of an arch_set "
                         "entry. arch_set deliberately spreads shapes apart within a level, which "
                         "is why the Qwen3 set contains zero pairs with the parameter count tied; "
                         "the pairs that actually separate a predictor from a parameter count have "
                         "to be named. See docs/supernet.md.")
    ap.add_argument("--steps", type=int, default=750)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--kd", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--mix", default="fineweb10bt=1.0")
    ap.add_argument("--data-seed", type=int, default=0,
                    help="DataMix seed. Fixed at 0 for every reference so far, which means NO shape "
                         "has been trained twice and training-seed variance is unmeasured -- so no "
                         "RMSE in this study can be compared to a floor. Vary it to get replicates.")
    ap.add_argument("--out", default=f"{RUNS}/groundtruth")
    ap.add_argument("--init", default="base",
                    help="'base' = extract from the published checkpoint; a path = extract from a "
                         "trained supernet. The supernet is the deployment path -- after a search "
                         "you would start from it, not go back to a naive truncation -- and it "
                         "starts far higher, which is what gives 2000 steps enough signal to "
                         "separate architectures at all.")
    a = ap.parse_args()
    spec = SPECS[a.model]
    archs = arch_set(spec)

    if a.shape:
        from llmforge.supernet.space import ElasticConfig as _EC
        q, h, m = (int(x) for x in a.shape.split(","))
        _c = _EC.uniform(spec, q, q, n_h=h, d_mlp=m)
        _W0 = _EC.full(spec).weight_params(include_embed=False)
        a.arch = a.arch or f"q{q}h{h}m{m}"
        archs = {a.arch: dict(w=_c.weight_params(include_embed=False) / _W0, kv=_c.kv_frac(),
                              d_qk=q, n_h=h, d_mlp=m)}

    if a.list or not a.arch:
        print(f"{'name':7} {'W':>6} {'KV':>6} {'d_qk':>5} {'n_h':>4} {'d_mlp':>6}")
        for k, v in archs.items():
            print(f"{k:7} {v['w']:6.3f} {v['kv']:6.3f} {v['d_qk']:5} {v['n_h']:4} {v['d_mlp']:6}")
        return
    if a.arch not in archs:
        # Levels can hold more than the default 4 shapes. Widening k only APPENDS -- the selection
        # loop is deterministic and fills in order, so L*_a..L*_d are byte-identical at any k and
        # architectures already trained stay valid.
        archs = arch_set(spec, k=8)
    A = archs[a.arch]
    from llmforge.supernet.space import ElasticConfig
    cfg = ElasticConfig.uniform(spec, A["d_qk"], A["d_qk"], n_h=A["n_h"], d_mlp=A["d_mlp"])
    order = pair_order_for(spec)
    os.makedirs(a.out, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[{a.arch}] W={A['w']:.3f} KV={A['kv']:.3f} d_qk={A['d_qk']} n_h={A['n_h']} "
          f"d_mlp={A['d_mlp']}", flush=True)
    base = AutoModelForCausalLM.from_pretrained(spec.repo, torch_dtype=torch.bfloat16).to("cuda")
    teacher = base.eval()                       # teacher is ALWAYS the published base
    if a.init == "base":
        sd = {k: v.clone() for k, v in base.state_dict().items()}
    else:
        import glob as _g
        from safetensors.torch import load_file as _lf
        sd = {}
        for f in sorted(_g.glob(os.path.join(a.init, "*.safetensors"))):
            sd.update(_lf(f))
        sd = {k: v.to("cuda") for k, v in sd.items()}
        print(f"[{a.arch}] init from supernet {a.init} ({len(sd)} tensors)", flush=True)
    for p in teacher.parameters():
        p.requires_grad_(False)

    student, _ = extract_dense(spec, sd, cfg, order)
    del sd
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    student.config.use_cache = False
    teacher.config.use_cache = False
    np_ = sum(p.numel() for p in student.parameters())
    print(f"[{a.arch}] dense student {np_/1e9:.3f}B params", flush=True)

    ratios = {k: float(v) for k, v in (kv.split("=") for kv in a.mix.split(","))}
    dm = DataMix(ratios=ratios, seed=a.data_seed)
    opt = torch.optim.AdamW(student.parameters(), lr=a.lr, weight_decay=0.1, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps,
                                                pct_start=0.08, anneal_strategy="cos")
    t0 = time.time()
    for step in range(a.steps):
        opt.zero_grad(set_to_none=True)
        for _ in range(a.accum):
            toks, _segs = dm.batch(a.batch)
            toks = toks.to("cuda")
            with torch.no_grad():
                tl = teacher(input_ids=toks).logits
            sl = student(input_ids=toks).logits
            V = sl.shape[-1]
            s2 = sl[:, :-1].reshape(-1, V).float()      # flatten to [B*(S-1), V] FIRST:
            t2 = tl[:, :-1].reshape(-1, V).float()      # batchmean divides by dim 0 only, so on a
                                                        # [B, S-1, V] tensor it would leave the KL
                                                        # S times too large (4096x here) and blow
                                                        # the gradient norm up to ~1e6.
            ce = F.cross_entropy(s2, toks[:, 1:].reshape(-1))
            kd = F.kl_div(F.log_softmax(s2 / a.tau, -1), F.log_softmax(t2 / a.tau, -1),
                          log_target=True, reduction="batchmean") * (a.tau ** 2)
            ((ce + a.kd * kd) / a.accum).backward()
        gn = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 25 == 0 or step == a.steps - 1:
            print(f"  step {step:4d}/{a.steps} ce {ce.item():.3f} kd {kd.item():.3f} "
                  f"gn {gn:.1f} {(time.time()-t0)/(step+1):.1f}s/step "
                  f"peak {torch.cuda.max_memory_allocated()/2**30:.1f}GiB", flush=True)
    d = os.path.join(a.out, a.arch)
    student.save_pretrained(d)
    AutoTokenizer.from_pretrained(spec.repo).save_pretrained(d)
    # model/parent_head_dim let eval_groundtruth rebuild the RoPE frequency subset. Without them
    # it has to assume the Qwen3 parent (head_dim 128) and silently installs the wrong schedule.
    # The recipe fields are recorded because a reference model IS the measuring instrument, and an
    # instrument with no calibration label cannot be trusted later. Until 2026-09-14 only the shape
    # was stored, so two sets trained months apart at different learning rates were
    # indistinguishable on disk -- and that mattered: at 135M the lr 3e-3 and lr 3e-4 references
    # invert 2 of 3 parameter-tied pairs (docs/supernet.md, reference recipe), so a set's learning rate
    # decides which architecture it calls better.
    json.dump({**A, "model": spec.key, "parent_head_dim": spec.head_dim,
               "steps": a.steps, "params": np_, "hours": (time.time()-t0)/3600,
               "recipe": {"lr": a.lr, "mix": a.mix, "kd": getattr(a, "kd", None),
                          "tau": getattr(a, "tau", None), "init": a.init,
                          "batch": a.batch, "accum": a.accum,
                          "seed": getattr(a, "seed", None), "ctx": getattr(a, "ctx", None)}},
              open(os.path.join(d, "arch.json"), "w"), indent=1)
    print(f"[{a.arch}] saved {d} in {(time.time()-t0)/3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
