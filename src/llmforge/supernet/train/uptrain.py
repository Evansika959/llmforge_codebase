"""Uptrain a published Qwen3 checkpoint into an elastic supernet (stage A: d_qk / d_v).

One trainer for every scale. The M2/M4 code had a separate `uptrain_1p7b.py` fork whose only
real difference was a hardcoded model id; geometry now comes from llmforge.supernet.config.ModelSpec,
so 0.6B / 1.7B / 4B run from the same file and 4B's 36 layers and 32 query heads are handled
rather than silently assumed away.

Per optimizer step (sandwich rule + in-place distillation), over `accum` micro-batches:
  * a frozen teacher provides soft targets;
  * the student runs K+2 configs = [FULL, MIN, K random], coverage-annealed global -> per-layer;
  * FULL trains on hard CE only; every other config on CE + kd * KD(teacher).

Two corrections over the M2/M4 trainer, both from the pre-code review:

  --teacher {base,init}  DEFAULT `base`. The old trainer silently defaulted the teacher to
      --init-ckpt, so a warm-started run distilled toward the *previous supernet* rather than the
      published checkpoint. The 0.6B and 1.7B supernets were in fact trained against different
      teachers because of this. `base` keeps every stage and every scale distilling toward the
      same fixed target, which is what makes a warm-started run comparable to a cold one.

  --grad-ckpt            Backbone activation checkpointing, which did not exist before. The 1.7B
      smoke test peaked at 54.7 GB with only 17.2 GB of resident state, i.e. ~37 GB was
      activations; scaling that to 4B (1.25x hidden, 1.29x layers) overflows an 80 GB card.
      Required at 4B, optional below it.

  python -m llmforge.supernet.train.uptrain --model qwen3-4b --steps 12000 --batch 1 --accum 16 --chunked --grad-ckpt
"""
import argparse
import json
import os
import random
import time

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

from ..config import SPECS
from ..data.collate import build_docaware
from ..data.datamix import DataMix
from ..elastic import (add_elastic_temperature, add_kv_alignment, build_pair_order, pair_order_for, enable_elastic,
                                 set_elastic_backend, set_elastic_config)
from ..elastic.sampler import (head_grid, nkv_options, sample_ab, sample_config, sandwich,
                               sandwich_ab)
from ..paths import RUNS
from ..space import ElasticConfig
from .losses import chunked_ce_kd, distill_kl, lm_ce
from .schedule import cosine_warmup


def eval_configs_for(spec, rng, ab=False, with_nkv=False):
    """Probe configs. Each entry is (name, qk, v, n_h, d_mlp, n_kv); None means "keep all".

    with_nkv appends probes that actually exercise KV-group pooling, and changes `min` to include
    it. Without them the probe set is blind to the knob: every entry leaves n_kv at full width, so
    a `min` that falls from 14.98 to 5.27 says the four WIDTH knobs are training and says nothing
    at all about whether pooling damage is repairable -- which was the entire question the run was
    launched to answer. This was live for one smoke run before it was caught.
    """
    g, n = spec.qk_grid, spec.n_layers
    lo, hi = g[0], g[-1]
    F = lambda x: [x] * n
    if not ab:
        uni = [("full", F(hi), F(hi), None, None), ("qk-hi", F(hi), F(lo), None, None),
               ("v-hi", F(lo), F(hi), None, None), ("mid", F(g[1]), F(g[1]), None, None),
               ("min", F(lo), F(lo), None, None)]
        h1 = sample_config(n=n, grid=g, per_layer=True, rng=rng)
        h2 = sample_config(n=n, grid=g, per_layer=True, rng=rng)
        return [c + (None,) for c in
                uni + [("het1", h1[0], h1[1], None, None), ("het2", h2[0], h2[1], None, None)]]
    # A+B: isolate each knob so a regression can be attributed, then the corner and two mixtures
    hg, mg = head_grid(spec), spec.mlp_grid
    kf = spec.n_kv
    probes = [("full", F(hi), F(hi), F(spec.n_q), F(spec.d_mlp), F(kf)),
              ("qk-lo", F(lo), F(hi), F(spec.n_q), F(spec.d_mlp), F(kf)),
              ("v-lo", F(hi), F(lo), F(spec.n_q), F(spec.d_mlp), F(kf)),
              ("h-lo", F(hi), F(hi), F(hg[0]), F(spec.d_mlp), F(kf)),
              ("mlp-lo", F(hi), F(hi), F(spec.n_q), F(mg[0]), F(kf)),
              ("min", F(lo), F(lo), F(hg[0]), F(mg[0]), F(kf))]
    if with_nkv:
        # One probe per rung below full, everything else at FULL width, so the number is
        # attributable to pooling alone. Then a corner that includes it.
        kg = [k for k in nkv_options(spec, spec.n_q) if k < kf]
        for k in sorted(kg, reverse=True):
            probes.append((f"kv{k}", F(hi), F(hi), F(spec.n_q), F(spec.d_mlp), F(k)))
        probes.append(("min+kv", F(lo), F(lo), F(hg[0]), F(mg[0]),
                       F(nkv_options(spec, hg[0])[0])))
    for i in (1, 2):
        c = sample_ab(spec, per_layer=True, rng=rng, with_nkv=with_nkv)
        probes.append((f"het{i}", *c) if with_nkv else (f"het{i}", *c, F(kf)))
    return probes


def load_one(dev, spec, ckpt, grad_ckpt=False, with_nkv=False):
    m = AutoModelForCausalLM.from_pretrained(spec.repo, torch_dtype=torch.bfloat16).to(dev)
    enable_elastic(m)
    set_elastic_backend("sdpa")
    add_elastic_temperature(m)
    if with_nkv:
        # Zero-initialised, so attaching it is a no-op until the optimiser moves it, and the full
        # configuration never reads it at all. Must come BEFORE the warm start so the parameters
        # exist when a checkpoint that has them is loaded.
        add_kv_alignment(m)
    if ckpt:
        miss, unexp = m.load_state_dict(
            load_file(os.path.join(ckpt, "model.safetensors")), strict=False)
        # missing=1 is expected and healthy: embeddings are tied, so lm_head.weight is not
        # serialized. Anything else means the warm start did not land.
        print(f"[warm-start] {ckpt}: missing={len(miss)} unexpected={len(unexp)} "
              f"{'(tied lm_head, expected)' if len(miss) == 1 else '<-- CHECK THIS'}", flush=True)
    m.config.use_cache = False
    if grad_ckpt:
        m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        print("[mem] backbone activation checkpointing ON", flush=True)
    return m


def build_models(dev, spec, order, init_ckpt, teacher_mode, grad_ckpt, with_nkv=False):
    student = load_one(dev, spec, init_ckpt, grad_ckpt=grad_ckpt, with_nkv=with_nkv)
    student.train()
    teacher_ckpt = None if teacher_mode == "base" else init_ckpt
    print(f"[teacher] {teacher_mode}: {teacher_ckpt or spec.repo}", flush=True)
    teacher = load_one(dev, spec, teacher_ckpt)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    set_elastic_config(teacher, spec.head_dim, spec.head_dim, order)
    return teacher, student


@torch.no_grad()
def evaluate(student, toks, mask, pos, order, cfgs):
    student.eval()
    out = {}
    for name, qk, v, nh, dm, nkv in cfgs:
        set_elastic_config(student, qk, v, order, n_h=nh, d_mlp=dm, n_kv=nkv)
        logits = student(input_ids=toks, attention_mask=mask, position_ids=pos).logits
        out[name] = lm_ce(logits, toks).item()
        del logits
    student.train()
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-4b", choices=sorted(SPECS))
    ap.add_argument("--knobs", choices=["qkv", "ab"], default="qkv",
                    help="'qkv' = d_qk/d_v only (stage A); 'ab' = + n_h and d_mlp (combined run)")
    ap.add_argument("--blocks", action="store_true",
                    help="sample one value per MEASURED layer block instead of per layer, so "
                         "training covers the space the search actually explores")
    ap.add_argument("--min-weight", type=float, default=1.0,
                    help="loss weight on the MIN corner; lower it only if MIN destabilises training")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--chunked", action="store_true", help="chunked CE/KD; avoids [B,S,vocab] logits")
    ap.add_argument("--grad-ckpt", action="store_true", help="backbone activation checkpointing")
    ap.add_argument("--kd", type=float, default=1.0)
    ap.add_argument("--ang-lr", type=float, default=0.05,
                    help="Learning rate for the KV alignment angles only. Must be far above the "
                         "backbone LR: Adam moves a parameter ~lr per step, angles live on an "
                         "O(1) radian scale, and at the backbone LR they finish at ~0.4 degrees "
                         "and contribute 0.0023 nats -- below the replication floor.")
    ap.add_argument("--with-nkv", action="store_true",
                    help="Include n_kv (KV-group count) as a fifth elastic knob.\n"
                         "KV cache is the cost that usually binds on edge hardware, so this is the\n"
                         "axis a hardware-aware search most wants -- but unlike the four width\n"
                         "knobs it is not a pure selection: pooling MIXES heads, so it needs the\n"
                         "learned post-norm alignment (add_kv_alignment) and its damage does not\n"
                         "have to behave like theirs. Untrained, n_kv 8->4 costs ~9.5 nats at 0.6B.\n"
                         "Only meaningful on Qwen3: n_kv must divide the model's own, and\n"
                         "SmolLM2's 3 and 5 are prime, leaving a single value.")
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--const-lr", action="store_true",
                    help="Hold the learning rate constant instead of annealing it to zero.\n"
                         "Every sweep so far ends flat, but that is uninformative: the cosine has "
                         "decayed lr to ~1e-3 of peak by then, so of course nothing moves. This "
                         "project already misread that once -- a 1.7B arm looked converged from "
                         "step 3500 and a larger step later took 2.5 nats off its two hardest "
                         "probes. A constant-lr continuation separates 'the schedule finished' "
                         "from 'the budget was enough'.")
    ap.add_argument("--anneal-frac", type=float, default=0.1)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--weight-decay", type=float, default=0.1,
                    help="AdamW decoupled decay. NOT independent of --lr: the update is "
                         "p -= lr*wd*p, so total weight shrinkage scales with the LR integral -- "
                         "2.5%% of weight magnitude at lr 2e-4 against 22.1%% at 2e-3 over 2500 "
                         "steps. Shrinking weights raises output entropy, which helps exactly the "
                         "slices that start out confidently wrong, so an LR sweep at fixed wd "
                         "confounds the two. Scale wd inversely with lr to hold shrinkage fixed.")
    ap.add_argument("--hi-weights", default=None,
                    help="Comma-separated rung sampling weights, e.g. 1,1,2,4. Default (1,1,2,2) "
                         "includes the first quarter of every matrix with probability 1.00 and the "
                         "last quarter with 0.33; tilting toward wide rungs narrows that gap. See "
                         "elastic.sampler.set_hi_weights.")
    ap.add_argument("--full-only", action="store_true",
                    help="Train ONLY the FULL config -- no sandwich, no MIN, no random configs. "
                         "This is the control for attributing full-width degradation: same lr, "
                         "tokens, mix and decay, but no weight sharing. If FULL still degrades, "
                         "the cause is distribution shift or decay; if it does not, the cause is "
                         "weight-sharing interference.")
    ap.add_argument("--init-ckpt", default=None)
    ap.add_argument("--teacher", choices=["base", "init"], default="base",
                    help="'base' = published checkpoint (default, comparable across stages); "
                         "'init' = --init-ckpt, reproducing the old implicit behaviour")
    ap.add_argument("--mix", default="fineweb10bt=1.0",
                    help="comma list bucket=weight, e.g. 'fineweb10bt=0.85,code=0.15'")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-batch", type=int, default=2)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    spec = SPECS[a.model]
    ckpt_dir = a.ckpt_dir or str(RUNS / a.model)
    ratios = {k: float(v) for k, v in (kv.split("=") for kv in a.mix.split(","))}

    if a.hi_weights:
        from ..elastic.sampler import set_hi_weights
        set_hi_weights([int(x) for x in a.hi_weights.split(",")])
        print(f"[sampler] rung weights -> {a.hi_weights}", flush=True)

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    dev = "cuda"
    order = pair_order_for(spec)

    print(f"[spec] {spec.key} {spec.n_params()/1e9:.3f}B | {spec.n_layers}L "
          f"{spec.n_q}Q/{spec.n_kv}KV hd{spec.head_dim} mlp{spec.d_mlp} | grid {spec.qk_grid}",
          flush=True)
    if a.knobs == "ab":
        print(f"[knobs] A+B: d_qk/d_v {spec.qk_grid} | n_h {head_grid(spec)} | "
              f"d_mlp {spec.mlp_grid} | MIN weight {a.min_weight}", flush=True)
    teacher, student = build_models(dev, spec, order, a.init_ckpt, a.teacher, a.grad_ckpt,
                                    with_nkv=a.with_nkv)
    # The KV alignment angles get their own group, and the reason is measured rather than a
    # preference. Adam's per-parameter step is ~lr regardless of gradient scale, so an angle can
    # travel at most the LR integral over the run: at lr 5e-4 over 250 steps that is 0.0925 rad,
    # and the largest angle in the model came out at 0.0914 -- it had moved exactly as far as the
    # schedule physically permitted. Angles have a natural scale of O(1) radians, so the backbone
    # LR leaves them at ~0.4 degrees, and the whole alignment then contributes 0.0023 nats, below
    # the replication floor. Fitting them properly with their own optimiser is worth 0.25-0.32
    # nats. Weight decay is also wrong for them: decay pulls a rotation toward the identity, which
    # is precisely the configuration the mechanism exists to leave.
    ang = [q for nm, q in student.named_parameters() if "kv_ang" in nm]
    rest = [q for nm, q in student.named_parameters() if "kv_ang" not in nm]
    groups = [{"params": rest, "lr": a.lr, "weight_decay": a.weight_decay}]
    if ang:
        groups.append({"params": ang, "lr": a.ang_lr, "weight_decay": 0.0})
        print(f"[nkv] {sum(q.numel() for q in ang):,} alignment angles at lr {a.ang_lr}, no decay",
              flush=True)
    opt = torch.optim.AdamW(groups, lr=a.lr, betas=(0.9, 0.95), weight_decay=a.weight_decay)
    _ang_base = [g.get("lr", a.lr) for g in opt.param_groups]
    dm = DataMix(ratios=ratios, seed=a.seed)
    S = dm.seqlen

    cfgs = eval_configs_for(spec, random.Random(1234), ab=(a.knobs == "ab"),
                            with_nkv=a.with_nkv)
    ev = DataMix(ratios=ratios, seed=a.seed + 9973)
    etoks, esegs = ev.batch(a.eval_batch)
    etoks = etoks.to(dev)
    emask, epos = build_docaware(esegs, S, device=dev, dtype=torch.bfloat16, neg=-1e9)

    def log_eval(step):
        e = evaluate(student, etoks, emask, epos, order, cfgs)
        print("  [eval %5d] " % step + " ".join(f"{k} {e[k]:.3f}" for k, *_ in cfgs), flush=True)

    os.makedirs(ckpt_dir, exist_ok=True)
    tok_per_step = a.batch * a.accum * S
    print(f"[budget] {a.steps} steps x {tok_per_step} tok = {a.steps*tok_per_step/1e9:.2f}B tokens",
          flush=True)
    t0 = time.time()
    log_eval(0)
    for step in range(a.steps):
        lr = a.lr if a.const_lr else cosine_warmup(step, a.steps, a.warmup, a.lr)
        for gi, g in enumerate(opt.param_groups):
            # Each group follows the same schedule SHAPE from its own base, so the angle group
            # keeps its 100x head start instead of being overwritten by the backbone's rate.
            g["lr"] = lr * (_ang_base[gi] / a.lr)
        opt.zero_grad(set_to_none=True)
        losses = None
        for _ in range(a.accum):
            toks, segs = dm.batch(a.batch)
            toks = toks.to(dev)
            mask, pos = build_docaware(segs, S, device=dev, dtype=torch.bfloat16, neg=-1e9)
            if a.full_only:
                f = ElasticConfig.full(spec)
                configs = [(f.d_qk, f.d_v, f.n_h, f.d_mlp)]
            elif a.knobs == "ab":
                configs = sandwich_ab(step, a.steps, spec, K=a.K, with_nkv=a.with_nkv,
                                      anneal_frac=a.anneal_frac, rng=rng, blocks=a.blocks)
            else:
                configs = [(qk, v, None, None) for qk, v in
                           sandwich(step, a.steps, n=spec.n_layers, grid=spec.qk_grid,
                                    K=a.K, anneal_frac=a.anneal_frac, rng=rng)]
            scale = 1.0 / (a.accum * len(configs))
            with torch.no_grad():
                if a.chunked:
                    teach = teacher.model(input_ids=toks, attention_mask=mask, position_ids=pos)[0]
                else:
                    teach = teacher(input_ids=toks, attention_mask=mask, position_ids=pos).logits
            ml = []
            for ci, cfg in enumerate(configs):
                qk, v, nh, dmlp = cfg[:4]
                nkv = cfg[4] if len(cfg) > 4 else None
                set_elastic_config(student, qk, v, order, n_h=nh, d_mlp=dmlp, n_kv=nkv)
                kdw = 0.0 if ci == 0 else a.kd
                w = a.min_weight if ci == 1 else 1.0     # ci==1 is MIN
                if a.chunked:
                    s_hidden = student.model(input_ids=toks, attention_mask=mask, position_ids=pos)[0]
                    loss = chunked_ce_kd(s_hidden, student.lm_head, toks, teach, teacher.lm_head,
                                         kd_weight=kdw, tau=a.tau, chunk=a.chunk)
                else:
                    s_logits = student(input_ids=toks, attention_mask=mask, position_ids=pos).logits
                    loss = lm_ce(s_logits, toks)
                    if kdw:
                        loss = loss + kdw * distill_kl(s_logits, teach, T=a.tau)
                (loss * scale * w).backward()
                ml.append(loss.item())
                del loss
            del teach
            losses = ml if losses is None else [x + y for x, y in zip(losses, ml)]
        losses = [x / a.accum for x in losses]
        gn = torch.nn.utils.clip_grad_norm_(student.parameters(), a.clip)
        opt.step()

        if step % a.log_every == 0 or step == a.steps - 1:
            dt = (time.time() - t0) / (step + 1)
            # --full-only leaves a single config, so MIN and the randoms are absent.
            tail = (f"min {losses[1]:.3f} rand [{' '.join(f'{c:.2f}' for c in losses[2:])}] "
                    if len(losses) > 1 else "(full-only) ")
            print(f"step {step:5d} lr {lr:.1e} gn {gn:5.1f} | full {losses[0]:.3f} "
                  f"{tail}| {dt:.1f}s/step "
                  f"{a.batch*a.accum*S/dt/1000:.1f}k tok/s "
                  f"| peak {torch.cuda.max_memory_allocated()/2**30:.1f}GiB", flush=True)
        if a.eval_every and (step + 1) % a.eval_every == 0:
            log_eval(step + 1)
        if a.ckpt_every and (step + 1) % a.ckpt_every == 0:
            d = os.path.join(ckpt_dir, f"step{step+1}")
            student.save_pretrained(d)
            torch.save(opt.state_dict(), os.path.join(d, "optim.pt"))
            # Every argument, so a checkpoint can be traced to the run that made it. This file is
            # also the marker that separates a post-fix elastic-KV checkpoint from a pre-fix one:
            # the supernet project treats an n_kv checkpoint without run.json as unquotable.
            json.dump(vars(a), open(os.path.join(d, "run.json"), "w"), indent=1, default=str)

    log_eval(a.steps)
    student.save_pretrained(os.path.join(ckpt_dir, "final"))
    print(f"done {a.steps} steps in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
