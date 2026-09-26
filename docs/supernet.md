# Supernet software evaluator

LLMForge scores the quality of a candidate architecture with an elastic supernet. The supernet
starts from a published pretrained checkpoint. Continued training makes many of its sub-networks
usable at once, and every candidate in the search space is one of those sub-networks. Scoring a
candidate therefore means selecting its slice of the trained weights and measuring held-out loss.
No candidate is trained on its own during the search.

This is post-training architecture search, or post-NAS. It only reaches shrink-slices of the base
checkpoint. Hidden size, depth and vocabulary stay at their base values.

The code lives in `src/llmforge/supernet`. The co-search reaches it through the software evaluator
backend in `llmforge.evaluators`, which pairs each loss with the costs reported by a hardware
backend.

## Base models

`ModelSpec.from_pretrained` checks every field below against the published `config.json`, so a
registry typo cannot survive a run. All five checkpoints tie input and output embeddings.
`llmforge.supernet.config.get()` falls back to `$LLMFORGE_SUPERNET_MODEL` when no key is given.

| key | checkpoint | family | layers | hidden | n_q | n_kv | head_dim | d_mlp | vocab |
|---|---|---|---|---|---|---|---|---|---|
| `smollm2-135m` | HuggingFaceTB/SmolLM2-135M | llama | 30 | 576 | 9 | 3 | 64 | 1536 | 49,152 |
| `smollm2-360m` | HuggingFaceTB/SmolLM2-360M | llama | 32 | 960 | 15 | 5 | 64 | 2560 | 49,152 |
| `qwen3-0.6b` | Qwen/Qwen3-0.6B-Base | qwen3 | 28 | 1024 | 16 | 8 | 128 | 3072 | 151,936 |
| `qwen3-1.7b` | Qwen/Qwen3-1.7B-Base | qwen3 | 28 | 2048 | 16 | 8 | 128 | 6144 | 151,936 |
| `qwen3-4b` | Qwen/Qwen3-4B-Base | qwen3 | 36 | 2560 | 32 | 8 | 128 | 9728 | 151,936 |

## Search space

Five per-layer knobs are searched. Every grid is shrink-only, so the base value is always the
largest entry. `n_kv` takes every divisor of the base `n_kv`, and `n_h` takes the entries of its grid
that the active `n_kv` divides, so a draw picks `n_kv` first and `n_h` second.

| key | d_qk and d_v | n_kv | n_h | (n_h, n_kv) pairs | d_mlp | uniform shapes |
|---|---|---|---|---|---|---|
| `smollm2-135m` | 16, 32, 48, 64 | 1, 3 | 3, 6, 9 | 6 | 384, 768, 1152, 1536 | 384 |
| `smollm2-360m` | 16, 32, 48, 64 | 1, 5 | 5, 10, 15 | 6 | 640, 1280, 1920, 2560 | 384 |
| `qwen3-0.6b` | 32, 64, 96, 128 | 1, 2, 4, 8 | 2, 4, 8, 16 | 13 | 768, 1536, 2304, 3072 | 832 |
| `qwen3-1.7b` | 32, 64, 96, 128 | 1, 2, 4, 8 | 2, 4, 8, 16 | 13 | 1536, 3072, 4608, 6144 | 832 |
| `qwen3-4b` | 32, 64, 96, 128 | 1, 2, 4, 8 | 4, 8, 16, 32 | 15 | 2432, 4864, 7296, 9728 | 960 |

A supernet trained without `--with-nkv` serves only the base `n_kv`, and its `n_h` grid is the
multiples of the base `n_kv`. Every checkpoint the hardware searches in `experiments/hw_nas` have used
so far is of that kind, and `llmforge.search.elastic_space` does not expose `n_kv` as a gene yet.

The cost range between the smallest corner and the full model, from `ElasticConfig`. Weight
parameters include the tied embedding and exclude norm gains. KV bytes assume a 16-bit cache.

| key | weight parameters | KV bytes per token, width knobs only | KV bytes per token, with `n_kv` |
|---|---|---|---|
| `smollm2-135m` | 50.4M to 134.5M | 5,760 to 23,040 | 1,920 to 23,040 |
| `smollm2-360m` | 112.1M to 361.8M | 10,240 to 40,960 | 2,048 to 40,960 |
| `qwen3-0.6b` | 227.1M to 596.0M | 28,672 to 114,688 | 3,584 to 114,688 |
| `qwen3-1.7b` | 586.4M to 1.72B | 28,672 to 114,688 | 3,584 to 114,688 |
| `qwen3-4b` | 1.09B to 4.02B | 36,864 to 147,456 | 4,608 to 147,456 |

The width knobs alone cannot take the KV cache below a quarter of the base, because `d_qk` and `d_v`
bottom out at a quarter of `head_dim`. `n_kv` takes it to 8.3% at SmolLM2-135M, 5.0% at SmolLM2-360M
and 3.1% at Qwen3.

A heterogeneous genome assigns one value per layer block, from
`llmforge.supernet.elastic.sampler.blocks_for`. The four attention knobs, `n_kv` included, share one
partition and `d_mlp` uses another. Qwen3-4B uses blocks fitted to measured per-layer top-1 sensitivity. Every
other model uses five contiguous blocks of near-equal size.

| key | attention blocks | MLP blocks |
|---|---|---|
| `smollm2-135m` | 0-5, 6-11, 12-17, 18-23, 24-29 | same as attention |
| `smollm2-360m` | 0-5, 6-11, 12-18, 19-24, 25-31 | same as attention |
| `qwen3-0.6b` | 0-4, 5-10, 11-15, 16-21, 22-27 | same as attention |
| `qwen3-1.7b` | 0-4, 5-10, 11-15, 16-21, 22-27 | same as attention |
| `qwen3-4b` | 0, 1-11, 12-21, 22-24, 25-35 | 0, 1-3, 4-33, 34-35 |

## How slicing works

- **d_qk.** Rotary dimensions come in pairs, and a valid slice keeps whole pairs. The pairs are
  ordered once by the `hilo` rule, which interleaves the highest and the lowest frequencies so any
  prefix keeps both local and long-range pairs. A slice keeps the first `d_qk / 2` pairs of that
  order for queries and keys. On Qwen3 the QK-Norm is recomputed over the active dimensions with
  the gathered gain. Attention logits scale by `1 / sqrt(d_qk)`.
- **d_v.** A contiguous prefix of every value head. The output is zero-padded back to `head_dim`
  before `o_proj`, which equals slicing the columns of `o_proj`.
- **n_h.** Group-balanced selection keeps `n_h / n_kv` query heads from every KV group, grouped by the
  active `n_kv` rather than the base one. A plain prefix of heads would leave whole KV groups without
  a reader. `n_kv` must divide `n_h`.
- **n_kv.** Fewer KV groups mean-pool adjacent groups. Before pooling, q, k and v are rotated pairwise
  by learned angles after QK-Norm and before RoPE, and the attention output is rotated back before
  `o_proj`. Pooling mixes weights where the other knobs only select them, which is why it needs the
  alignment. The angles start at zero, one set per `n_kv` rung, and v carries a separate set per `d_v`
  rung because its rotation plane moves with `d_v`. `add_kv_alignment` attaches them, and
  `--with-nkv` trains them in their own optimizer group at `--ang-lr`, default 0.05, without weight
  decay.
- **d_mlp.** A nested prefix: the first `d_mlp` rows of `gate_proj` and `up_proj` and the first
  `d_mlp` columns of `down_proj`.
- **Temperature.** Every attention module carries a learnable `logit_scale` with one entry per
  `d_qk` rung. It starts at zero, so an untrained supernet computes exactly the base model. Narrow
  slices have structurally softer attention logits, and the temperature lets training correct that.
- **Not searched.** The attention and MLP gates exist only in the cost model of `ElasticConfig`.
- **Ablation switches.** `elastic.attention.ABLATION` swaps balanced head selection for a plain head
  prefix, and KV pooling for keeping the first group of every bundle. Training and search never set
  it. `experiments/supernet_fidelity/conversion_ablation.py` uses it to measure what each rule is worth
  on the untrained checkpoint.

At full width every gather is a permutation, so the elastic forward matches stock Hugging Face
attention. `tests/supernet/test_parity.py` checks this on a GPU.

## Training

`python -m llmforge.supernet.train.uptrain` trains a supernet with the sandwich rule and in-place
distillation.

- Every micro-batch runs K + 2 configurations through the student: FULL, MIN and K random draws,
  with K = 2.
- FULL trains on next-token cross-entropy alone. MIN and the random draws train on cross-entropy
  plus a KL term toward the frozen published checkpoint, with temperature 2 and weight 1. The
  teacher is always the published checkpoint unless `--teacher init` is passed.
- MIN is the true corner of all four knobs. `--min-weight` scales its loss and defaults to 1.
- Coverage anneal: for the first 10% of steps a random draw uses one value for every layer. After
  that each layer draws independently, or each block does under `--blocks`.
- `d_qk` and `d_v` rungs are drawn with weights 1, 1, 2 and 2 from narrowest to widest, which
  `--hi-weights` overrides. `n_h` and `d_mlp` are drawn uniformly.
- AdamW with betas 0.9 and 0.95, weight decay 0.1, gradient clipping at 1.0, 200 warmup steps and a
  cosine decay to zero. `--const-lr` holds the rate constant instead.
- Cross-entropy and KL are computed in sequence chunks from hidden states, so full-vocabulary logits
  never materialize. `--grad-ckpt` enables activation checkpointing, which Qwen3-4B needs on an
  80 GB GPU.
- Sequences are packed to 4096 tokens with document-aware block-diagonal causal masks and positions
  that restart at every document. Every recipe below uses 16 sequences per step, or 65,536 tokens.
- The `[eval]` line printed during training scores two packed sequences drawn from the training pool
  with the run seed. It is not held out, so use it only for the shape of a trajectory.

### Data

```bash
python -m llmforge.supernet.data.download_raw
python -m llmforge.supernet.data.download
python -m llmforge.supernet.data.pack_stream --parquet-dir <sample-10BT parquet dir> --out fineweb10bt
scripts/supernet/pack_smollm.sh
scripts/supernet/pack_qwen.sh
```

`download_raw` writes the four raw buckets to `$LLMFORGE_DATA/raw` as jsonl:

| bucket | source | filter and format | cap |
|---|---|---|---|
| `fineweb_edu` | `HuggingFaceFW/fineweb-edu`, sample-10BT | document text as is | 3 GB |
| `code` | `bigcode/the-stack-smol` | source files of at least 2,000 characters | 3 GB |
| `openmath` | `nvidia/OpenMathInstruct-2` | problem followed by its generated solution | 2 GB |
| `rag_hotpot` | `hotpotqa/hotpot_qa`, distractor, train | titled context paragraphs, question, answer | 3 GB |

Each source is streamed in its published order and cut at the cap, so a rerun reproduces the same
documents while the upstream datasets are unchanged. HotpotQA ends before its cap.
`bigcode/the-stack-smol` is gated, so accept its terms of use on the Hugging Face Hub and log in
before the code bucket is fetched. `download` fetches the complete FineWeb-Edu sample-10BT parquet,
which the Qwen3 recipes use as their web bucket.

Packed buckets are token-id arrays under `$LLMFORGE_DATA/packed`. Token ids do not transfer between
tokenizers, so SmolLM2 trains on its own `sl_` buckets, which `pack_smollm.sh` packs from all four
raw buckets. `pack_qwen.sh` packs the `code`, `math` and `rag` buckets with the Qwen3 tokenizer, and
`fineweb10bt` comes from the parquet.

The SmolLM2 recipes and the Qwen3-0.6B recipe mix web, code, math and retrieval at 40, 25, 20 and 15
percent. The retrieval bucket stands in for a long-document bucket that was never packed. The
Qwen3-1.7B and Qwen3-4B recipes train on FineWeb-Edu sample-10BT alone, apart from one math and
code variant at 4B.

### Recipes

Every script runs from the repository root, writes checkpoints to `runs/supernet/<name>` and logs
to `logs/supernet`. All of them pass `--teacher base --warmup 200 --anneal-frac 0.1 --chunked` with
K = 2 and seed 0. Environment variables such as `LR`, `STEPS` and `TAG` override the defaults.

| script | model | knobs | lr | steps | batch x accum | tokens | mix | other flags |
|---|---|---|---|---|---|---|---|---|
| `run_uptrain_smollm135m.sh` | smollm2-135m | all four | 5e-4 | 4500 | 8 x 2 | 295M | sl 40/25/20/15 | none |
| `run_uptrain_smollm360m.sh` | smollm2-360m | all four | 1e-3 | 2500 | 4 x 4 | 164M | sl 40/25/20/15 | none |
| `run_uptrain_qwen06b.sh` | qwen3-0.6b | all four | 5e-4 | 2500 | 2 x 8 | 164M | 40/25/20/15 | none |
| `run_uptrain_1p7b.sh` | qwen3-1.7b | all four | 3e-5 | 5000 | 2 x 8 | 328M | fineweb10bt | none |
| `run_ablation_1p7b.sh` | qwen3-1.7b | all four | 3e-5 | 3500 | 2 x 8 | 229M | fineweb10bt | one arm |
| `run_uptrain_ab.sh` | qwen3-4b | all four | 3e-5 | 8000 | 2 x 8 | 524M | fineweb10bt | `--grad-ckpt` |
| `run_uptrain_4b_blocks.sh` | qwen3-4b | all four | 3e-5 | 2500 | 2 x 8 | 164M | fineweb10bt | `--blocks --grad-ckpt` |
| `run_uptrain_4b_mathmix.sh` | qwen3-4b | all four | 3e-5 | 2500 | 2 x 8 | 164M | fineweb10bt/math/code 50/30/20 | `--grad-ckpt` |
| `run_uptrain_4b.sh` | qwen3-4b | d_qk, d_v | 3e-5 | 8000 | 2 x 8 | 524M | fineweb10bt | `--grad-ckpt` |

`run_ablation_1p7b.sh` changes one setting per arm: K of 1 or 4, a MIN weight of 0.3, a learning
rate of 6e-5, 1e-4, 2e-4 or 4e-4, or gradient clipping at 10 or 100 with the rate held at 2e-4.

The supernet learning rate is swept for each scale and never inherited from another scale. The
selection rule is the lowest mean held-out loss over six named probes at the final step, namely
`full`, `qk-lo`, `v-lo`, `h-lo`, `mlp-lo` and `min`, with the full-width loss reported beside it as a
guard. Check that the chosen rate is bracketed by the sweep. The sweep arms trained so far ran
2500 steps each:

- SmolLM2-135M at 2e-4, 5e-4, 1e-3, 2e-3, 4e-3 and 8e-3, a full-width-only control trained with
  `--full-only`, and two further controls named `tilt` and `wdctl` whose flags the scripts do not
  record.
- SmolLM2-360M at 5e-4, 1e-3, 2e-3 and 4e-3.
- Qwen3-0.6B at 2e-4, 5e-4, 1e-3 and 2e-3, plus a 1200-step continuation warm-started from the 5e-4
  arm.

## Scoring slices

`llmforge.supernet.search.nsga` holds the two functions a search needs.

- `load_supernet(ckpt, spec)` loads the base checkpoint, installs the elastic forward with the eager
  backend, registers the temperature parameters and loads every safetensors shard in `ckpt`. It
  refuses a checkpoint that carries no trained temperature.
- `make_evaluator(model, tok, order, spec, texts, max_len=1024)` tokenizes every document once,
  truncates at 1024 tokens and drops documents shorter than 8 tokens. The returned `loss_of(cfg)`
  pins a configuration and returns the token-weighted mean next-token cross-entropy over all
  documents.

A search scores documents 0 to 31 of the frozen held-out set. The final front is then re-scored on
documents 32 to 95, so the validation score never shares a document with the search.
`assets/heldout/README.md` documents the provenance of those documents.

```python
from llmforge.supernet.config import SPECS
from llmforge.supernet.data.heldout import heldout_texts
from llmforge.supernet.search.nsga import load_supernet, make_evaluator
from llmforge.supernet.space import ElasticConfig

spec = SPECS["smollm2-135m"]
model, tok, order = load_supernet("runs/supernet/<run>/<step>", spec)
search_loss, _ = make_evaluator(model, tok, order, spec, heldout_texts(32))
val_loss, _ = make_evaluator(model, tok, order, spec, heldout_texts(64, skip=32))

cfg = ElasticConfig.uniform(spec, d_qk=48, d_v=32, n_h=6, d_mlp=1152).validate()
print(search_loss(cfg), val_loss(cfg), cfg.cost(ctx=1024))
```

Fidelity studies use `experiments/supernet_fidelity/slice_nll.py`, which concatenates documents
into fixed windows. Pass `--texts assets/heldout/heldout_ab.json --n 192 --ctx 1024` and confirm the
printed window count. Cross-entropy does not transfer across tokenizers, so a SmolLM2 score and a
Qwen3 score never share an axis. Compare rankings within one scale instead.

## Exact dense extraction

`llmforge.supernet.extract.extract_dense(spec, state_dict, cfg, pair_order)` builds a standalone
Hugging Face model for a uniform configuration with `d_qk` equal to `d_v`.

- The slice keeps a subset of the parent's rotary frequencies, so the dense model's `inv_freq` is
  replaced with that subset. Hugging Face does not save `inv_freq`, so
  `experiments/supernet_fidelity/eval_groundtruth.py` restores it from the `parent_head_dim` field
  of `arch.json` whenever a saved model is reloaded.
- The learned temperature folds into the `q_norm` gain on Qwen3 and into `q_proj` on Llama, which
  has no QK-Norm.

`experiments/supernet_fidelity/validate_slicing.py` checks the extraction against the slice. The
recorded result is a maximum absolute fp32 logit difference of 3.4e-5 with full argmax agreement.
Heterogeneous configurations have no dense Hugging Face form. They are trained and scored in place
through the elastic forward with the configuration pinned, so only active weights receive gradient.

## Ground truth

The supernet is a predictor, so claims about it need reference models trained on their own weights.
All scripts below live in `experiments/supernet_fidelity`.

- `groundtruth.py` extracts one uniform architecture and fine-tunes it with the supernet objective
  minus weight sharing: cross-entropy plus KL toward the frozen published checkpoint at temperature
  2. It uses AdamW with betas 0.9 and 0.95, weight decay 0.1, a one-cycle cosine schedule with 8%
  warmup, gradient clipping at 1.0 and activation checkpointing. `--init base` extracts from the
  published checkpoint and `--init <ckpt>` from a trained supernet. Every model directory records
  its shape, `parent_head_dim` and full recipe in `arch.json`.
- `groundtruth_het.py` trains a per-layer heterogeneous architecture in place.
- `gt_grid.py` trains every shape in an architecture list and skips shapes already on disk. It
  shards across machines with `--stride`, `--offset` and `--take`. `gt135_set.py --freeze` writes
  the full 48-shape SmolLM2-135M grid with `d_qk` equal to `d_v`.
- `eval_groundtruth.py` and `eval_het.py` score held-out cross-entropy on the same windows the
  slices use and store per-window values, so differences between architectures can be bootstrapped
  paired.
- `rank_fidelity.py`, `gt135_calibration.py` and `tau_vs_tokens.py` report Kendall tau_b,
  within-band concordance with an exact permutation null, top-1 regret against free rules in both
  directions, and leave-one-out calibration error against a free additive knob table.

Reference recipes used so far. Every set trained 2000 steps at batch 2 and accumulation 8, on the
SmolLM2 40/25/20/15 mix, initialized from the published checkpoint.

| set | model | reference lr |
|---|---|---|
| `gt135` | smollm2-135m | 3e-3 |
| `gt135_hiband` | smollm2-135m | 3e-4 |
| `gt135_lr1e-4` | smollm2-135m | 1e-4 |
| `gt360` | smollm2-360m | 1e-3 |

### Identity check

At full width a reference architecture is the base model, so fine-tuning should leave its held-out
loss where it started. `identity_check.py` sweeps the reference learning rate at full width. A rate
fails when the run ends above the untouched base by more than the replication floor of about
0.005 nats. Measured on 58 windows of 1024 tokens, against base losses of 2.7101 for SmolLM2-135M
and 2.4960 for SmolLM2-360M:

| lr | change at 135M | change at 360M |
|---|---|---|
| 1e-5 | not run | -0.0016 |
| 3e-5 | -0.0018 | -0.0041 |
| 1e-4 | -0.0036 | -0.0075 |
| 3e-4 | -0.0030 | -0.0059 |
| 1e-3 | not run | +0.0205, fails |
| 3e-3 | +0.1280, fails | not run |

The rate each ground-truth set used is the rate that fails. A single lower rate is not a fix,
because narrow architectures trained at 1e-4 end 2.5 to 2.8 nats worse than at 3e-3. At 135M the
recipe's shape-dependent noise is 0.034 to 0.036 nats while parameter-tied pairs differ by only
0.011 to 0.015 nats, so within-band pair comparisons at that scale measure the recipe rather than
the architecture. Run the identity check on every new scale before its references are used, and
never inherit a reference learning rate across scales.

## Measurement rules

- Report Kendall tau_b or tau_a. The ratio `(c - d) / (c + d)` is Goodman-Kruskal gamma, which drops
  tied pairs and inflates free rules that tie often.
- Report top-1 selection regret for every free rule in both directions.
- The same configuration trained from different data seeds varies with a standard deviation of
  0.0054 nats. Treat that as a lower bound on the noise of a supernet arm.
- Save per-architecture and per-window scores, not only summary statistics.
- Pass `--texts` explicitly and check the printed window count, because both sides of a comparison
  must score the same documents with the same tokenizer.

## Checking a supernet before a search

A supernet run is only a usable evaluator if its slice scores respect nesting. A configuration that
is at least as wide as another in every knob should never score worse. Violations cluster at the
MIN corner, which the sandwich rule trains on every step. In the SmolLM2-135M sweep the violation
count fell as the supernet learning rate rose. Count violations on the uniform grid, then check
selection regret under size and KV budgets against reference models, before a run serves a search.

## Reproducing

```bash
pip install -e .    # or export PYTHONPATH=src

# supernet
python -m llmforge.supernet.data.download
scripts/supernet/pack_smollm.sh
LR=8e-3 STEPS=2500 TAG=lr8e-3 scripts/supernet/run_uptrain_smollm135m.sh
python experiments/supernet_fidelity/slice_nll.py --model smollm2-135m \
    --ckpt runs/supernet/sl135_lr8e-3/step2500 --probes \
    --texts assets/heldout/heldout_ab.json --n 192 --ctx 1024

# ground truth and rank fidelity
python experiments/supernet_fidelity/identity_check.py --model smollm2-135m --shape 64,9,1536 \
    --lrs 3e-5,1e-4,3e-4 --mix sl_fineweb=0.40,sl_code=0.25,sl_math=0.20,sl_rag=0.15 \
    --out runs/supernet/ident135
python experiments/supernet_fidelity/gt135_set.py --freeze
python experiments/supernet_fidelity/gt_grid.py --archs runs/supernet/gt135/ARCHS.json \
    --model smollm2-135m --out runs/supernet/gt135 --lr 3e-3 \
    --mix sl_fineweb=0.40,sl_code=0.25,sl_math=0.20,sl_rag=0.15
python experiments/supernet_fidelity/eval_groundtruth.py --dir runs/supernet/gt135 --nll-only --ctx 1024
python experiments/supernet_fidelity/slice_nll.py --model smollm2-135m \
    --ckpt runs/supernet/sl135_lr8e-3/step2500 --archs runs/supernet/gt135/ARCHS.json --ctx 1024
python experiments/supernet_fidelity/rank_fidelity.py --nll \
    --slices runs/supernet/slice_nll_smollm2-135m.json --truth runs/supernet/groundtruth_nll.json
```

Use the same `--ctx` and the same document file on the slice side and the reference side of every
comparison.
