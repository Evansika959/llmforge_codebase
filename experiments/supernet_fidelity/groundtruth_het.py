#!/usr/bin/env python
"""Dedicated training of PER-LAYER HETEROGENEOUS architectures -- the measurement that decides
whether the supernet's flattened per-layer sensitivity is a defect or a faithful reflection.

The 48-shape uniform grid cannot answer that. Its architectures all apply one width everywhere, so
they say nothing about whether trained models actually differ when the width varies BY LAYER. The
supernet says they barely do (CV 1.18 -> 0.18 on d_mlp after training); the untrained base says
they differ by up to 58x. Only real training settles it -- and the answer flips the recommendation:
if dedicated training also flattens, the supernet is right, per-layer search has nothing to find,
and changing the sampling distribution would make the predictor WORSE by inventing differences.

Why this cannot reuse groundtruth.py: `extract_dense` asserts uniform widths, because a dense HF
model has one head_dim and one intermediate_size. A heterogeneous architecture has no dense form.

So it is trained IN PLACE through the elastic forward with its own config pinned. Only the active
weights receive gradient -- index_select and prefix slicing route zeros to the rest -- which is
exactly what dedicated training of that architecture means. The forward is bit-equivalent to the
dense extraction where one exists (verified 1e-5 max|dlogit| at three configs), so the two arms are
measured on the same footing. Inactive weights still decay under AdamW; they are never read.

  python experiments/supernet_fidelity/groundtruth_het.py --list                       # what would be trained
  python experiments/supernet_fidelity/groundtruth_het.py --arch H03                   # train one
"""
import argparse, json, os, sys, time

import numpy as np
import torch
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer

from llmforge.supernet.config import SPECS
from llmforge.supernet.data.collate import build_docaware
from llmforge.supernet.data.datamix import DataMix
from llmforge.supernet.elastic import (add_elastic_temperature, enable_elastic, pair_order_for,
                                set_elastic_backend, set_elastic_config)
from llmforge.supernet.space import ElasticConfig
from llmforge.supernet.train.losses import chunked_ce_kd
from llmforge.supernet.train.schedule import cosine_warmup
from llmforge.supernet.paths import RUNS

SET = f"{RUNS}/gt135_het/ARCHS_HET.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="smollm2-135m")
    ap.add_argument("--arch", default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--set", default=SET)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-3)      # same as the uniform arm
    ap.add_argument("--kd", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--mix", default="sl_fineweb=0.40,sl_code=0.25,sl_math=0.20,sl_rag=0.15")
    ap.add_argument("--out", default=f"{RUNS}/gt135_het")
    ap.add_argument("--grad-ckpt", action="store_true",
                    help="non-reentrant gradient checkpointing, for models above SmolLM2 scale")
    a = ap.parse_args()

    spec = SPECS[a.model]
    archs = {r["name"]: r for r in json.load(open(a.set))}
    if a.list or not a.arch:
        print(f"{'name':6} {'w':>6} {'kv':>6} {'het':>6}  spread(d_qk / n_h / d_mlp)")
        for k, r in archs.items():
            print(f"{k:6} {r['w']:6.3f} {r['kv']:6.3f} {r.get('het', 0):6.3f}  "
                  f"{len(set(r['d_qk'])):>2}/{len(set(r['n_h'])):>2}/{len(set(r['d_mlp'])):>2} levels")
        return

    A = archs[a.arch]
    cfg = ElasticConfig(spec, A["d_qk"], A["d_v"], [spec.n_kv] * spec.n_layers,
                        A["n_h"], A["d_mlp"])
    order = pair_order_for(spec)
    os.makedirs(a.out, exist_ok=True)
    dev, S = "cuda", 4096
    print(f"[{a.arch}] w={A['w']:.3f} kv={A['kv']:.3f} het={A.get('het',0):.3f} "
          f"steps={a.steps} lr={a.lr}", flush=True)

    teacher = AutoModelForCausalLM.from_pretrained(
        spec.repo, torch_dtype=torch.bfloat16).to(dev).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student = AutoModelForCausalLM.from_pretrained(
        spec.repo, torch_dtype=torch.bfloat16).to(dev)
    enable_elastic(student); set_elastic_backend("sdpa")
    # enable_elastic patches the CLASS, so the teacher -- same family -- now runs the elastic
    # forward too and needs a config of its own. Full width is bit-equivalent to the stock forward
    # (4.0e-05 max|dlogit|, 100% argmax), so this leaves the teacher exactly as it was.
    full = ElasticConfig.full(spec)
    set_elastic_config(teacher, full.d_qk, full.d_v, order, n_h=full.n_h, d_mlp=full.d_mlp)
    teacher.config.use_cache = False
    student.config.use_cache = False
    set_elastic_config(student, cfg.d_qk, cfg.d_v, order, n_h=cfg.n_h, d_mlp=cfg.d_mlp)
    if a.grad_ckpt:
        # Checkpointing only activates in train mode. These models have no dropout, so train
        # mode changes nothing else.
        student.train()
        student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        student.gradient_checkpointing_disable()

    ratios = {k: float(v) for k, v in (kv.split("=") for kv in a.mix.split(","))}
    dm = DataMix(ratios=ratios, seed=a.data_seed)
    opt = torch.optim.AdamW(student.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1)
    t0 = time.time()
    for step in range(a.steps):
        lr = cosine_warmup(step, a.steps, 200, a.lr)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        for _ in range(a.accum):
            toks, segs = dm.batch(a.batch)
            toks = toks.to(dev)
            mask, pos = build_docaware(segs, S, device=dev, dtype=torch.bfloat16, neg=-1e9)
            with torch.no_grad():
                teach = teacher.model(input_ids=toks, attention_mask=mask, position_ids=pos)[0]
            s_hidden = student.model(input_ids=toks, attention_mask=mask, position_ids=pos)[0]
            loss = chunked_ce_kd(s_hidden, student.lm_head, toks, teach, teacher.lm_head,
                                 kd_weight=a.kd, tau=a.tau, chunk=1024)
            (loss / a.accum).backward()
            last = loss.item()
        gn = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0).item()
        opt.step()
        if step % 25 == 0 or step == a.steps - 1:
            print(f"  step {step:4d}/{a.steps} loss {last:.3f} gn {gn:.1f} "
                  f"{(time.time()-t0)/(step+1):.1f}s/step "
                  f"peak {torch.cuda.max_memory_allocated()/2**30:.1f}GiB", flush=True)

    d = os.path.join(a.out, a.arch)
    os.makedirs(d, exist_ok=True)
    from safetensors.torch import save_file
    # lm_head is tied to embed_tokens and shares storage; safetensors refuses aliased tensors.
    # Dropping it is safe -- from_pretrained re-ties on load, which is how every other checkpoint
    # in this project is stored.
    sd = {k: v.contiguous() for k, v in student.state_dict().items() if k != "lm_head.weight"}
    save_file(sd, os.path.join(d, "model.safetensors"))
    json.dump({**A, "steps": a.steps, "lr": a.lr, "model": spec.key,
               "parent_head_dim": spec.head_dim, "heterogeneous": True,
               "hours": (time.time() - t0) / 3600},
              open(os.path.join(d, "arch.json"), "w"), indent=1)
    print(f"[{a.arch}] saved {d} in {(time.time()-t0)/3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
