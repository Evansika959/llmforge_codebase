# Heterogeneous / infinite-attention support (`.rlm` format + `run_llmforge.c`)

Goal: run LLMForge NSGA models that use **infinite-head attention** in the pure-C engine so
they can eventually run on Android. Target model: `nsga_best3_rotary_periln_105M`.

## Why the existing llama2.c path cannot run these models

The llama2.c `.bin` formats (legacy v0 read by `run.c`, v2 Q8_0 read by `runq.c`) encode a single
**global** `Config` (`dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab, seq_len`) and assume:

- every layer has the same `n_head`, `n_kv`, `head_dim = dim/n_head`, and `hidden_dim`;
- q/k/v all share one `head_dim`;
- attention output is `concat(heads) == dim`, projected by a **square** `wo: dim→dim`.

`nsga_best3_rotary_periln_105M` violates all of these (ground truth from checkpoint tensor shapes):

| layer | attn      | n_head | n_kv | qk_dim | v_dim | mlp_hidden | c_proj (out×in) |
|-------|-----------|--------|------|--------|-------|------------|-----------------|
| 0–2   | infinite  | 8      | 2    | 320    | 512   | 768        | 768 × 4096      |
| 3–5   | infinite  | 8      | 4    | 256    | 224   | 3584       | 768 × 1792      |
| 6     | identity  | –      | –    | –      | –     | 1280       | (no attn)       |

- **infinite = concat path**: each head emits `v_dim`, heads are concatenated to `n_head·v_dim`, then
  a **non-square** `c_proj: (n_head·v_dim → n_embd)`. `qk_dim ≠ v_dim`.
- per-layer dims all differ; one layer has **no attention** (`nn.Identity`).

So we need a **self-describing, per-layer format** and a dedicated runner. First cut: **fp32 only**
(parity focus). Quantization + Android come after bit-exact parity is proven.

## Confirmed architecture (from LLMForge source + checkpoint)

Source: `variations/attention_variations.py` (`InfiniteHeadAttention` L981, `AttnIdentity` L973,
`_compute_kv_group_distribution` L18), `variations/block_variations.py` (`attn_then_mlp_forward`
L114, `_resolve_unit_norm_flags` L267), `variations/mlp_variations.py` (`Swiglu` L~527/595),
`variations/position_encoding_variations.py` (`RotaryEmbedding` L21), `variations/norm_variations.py`
(`RMSNorm` L53). Model args: `model_args` in `ckpt.pt` carries the `*_layerlist` arrays.

**Block (per layer), variant `attn_then_mlp`, with `use_pre_ln=use_peri_ln=True`, `use_post_ln=False`:**
```
x_attn_in = pre_ln_attn(x)                 # RMSNorm
attn_out  = attn(x_attn_in)                # infinite OR identity(=x_attn_in)
attn_out  = peri_ln_attn(attn_out)         # RMSNorm
x         = x + attn_out                   # residual add

x_mlp_in  = pre_ln_mlp(x)                  # RMSNorm
mlp_out   = mlp(x_mlp_in)                  # GeGLU
mlp_out   = peri_ln_mlp(mlp_out)           # RMSNorm
x         = x + mlp_out                    # residual add
```
Identity layer 6 keeps all four norms; attention is a pass-through, so it contributes
`x += peri_ln_attn(pre_ln_attn(x))` (NOT a no-op).

**Infinite attention forward** (no qk-norm, no v-norm, no flash-lobo, `use_concat_heads=True`, `bias=False`):
```
q = c_attn_q(x)   # (n_head·qk, n_embd) @ x  -> n_head·qk
k = c_attn_k(x)   # (n_kv·qk,  n_embd)       -> n_kv·qk
v = c_attn_v(x)   # (n_kv·vd,  n_embd)       -> n_kv·vd
reshape q->(n_head,qk), k->(n_kv,qk), v->(n_kv,vd)
RoPE(q) over qk ; RoPE(k) over qk           # interleaved, base 10000
per head h: g = head_to_kv[h]
    att[t] = (q_h · k_g[t]) / sqrt(qk)       # scale = 1/sqrt(qk_dim)
    softmax(att[:pos+1]); causal
    out_h = Σ_t att[t] · v_g[t]              # (vd,)
concat heads -> (n_head·vd)                   # head-major order
y = c_proj(concat)                            # (n_embd, n_head·vd) @ concat -> n_embd
```
GQA group map = `_compute_kv_group_distribution(n_head, n_kv)`: contiguous groups, sizes
`base = n_head//n_kv` with the first `n_head%n_kv` groups getting one extra; `head_to_kv[h]` =
that layout. For this model all layers divide evenly, but the runner implements the general rule.

**GeGLU MLP** (`mlp_variant=swiglu`, `activation=gelu`, offsets 0, no l2/cproj scaling):
```
mlp_out = c_fc_out( gelu(c_fc_in1(x)) * c_fc_in2(x) )   # w2( gelu(w1 x) * w3 x )
```

**RMSNorm (IMPORTANT — no epsilon):** `rms = ‖x‖₂ / √d ; y = x / rms · gain`. The existing
`run.c`/`runq.c` use `eps=1e-5`; the new runner must use **no eps** for faithful parity.

**RoPE:** interleaved adjacent pairs (`x[0::2],x[1::2]`), base `10000`, over the full `qk_dim`
(rope_length=None). `inv_freq[d] = 10000^(-2d/qk)`, angle `= pos · inv_freq[d]`. Same convention
already validated for smollm2; only the per-layer `head_size = qk_dim` changes.

**Head/vocab:** `n_embd=768`, `vocab=50257`, `block_size=1024`, `wte` tied to `lm_head`
(`wte_weight_tying=True`), no abs pos emb, tokenizer = tiktoken gpt2 (reuse `bpe.h`).

## `.rlm` file format (little-endian, fp32)

```
Global header (magic + fixed fields, then zero-pad to 256 bytes):
  u32   magic          = 0x726c6d31   ("rlm1")
  i32   version        = 1 (fp32) | 2 (Q8_0)
  i32   n_layer
  i32   n_embd
  i32   vocab_size
  i32   seq_len        (block_size)
  i32   shared_classifier   (1 = lm_head tied to wte)
  i32   act             (0 = gelu, 1 = silu)   [global]
  f32   rope_theta      (10000.0)
  i32   group_size      (Q8_0 group size; 0 for fp32)
  <zero pad to 256 bytes>

Per-layer descriptor table (n_layer × 8 × i32):
  i32   attn_type   (0 = infinite_concat, 1 = identity)
  i32   n_head
  i32   n_kv
  i32   qk_dim
  i32   v_dim
  i32   mlp_hidden
  i32   reserved0 (0)
  i32   reserved1 (0)

Weights (fp32), per layer in order:
  pre_ln_attn.gain    (n_embd)
  peri_ln_attn.gain   (n_embd)
  if attn_type == infinite_concat:
      c_attn_q  (n_head·qk_dim × n_embd)
      c_attn_k  (n_kv·qk_dim   × n_embd)
      c_attn_v  (n_kv·v_dim    × n_embd)
      c_proj    (n_embd × n_head·v_dim)
  pre_ln_mlp.gain     (n_embd)
  peri_ln_mlp.gain    (n_embd)
  w1 / c_fc_in1 (mlp_hidden × n_embd)     # gate
  w3 / c_fc_in2 (mlp_hidden × n_embd)     # value
  w2 / c_fc_out (n_embd × mlp_hidden)     # down

Final:
  ln_f.gain   (n_embd)
  wte         (vocab × n_embd)
  lm_head     (vocab × n_embd)   only if shared_classifier == 0
```
All matrices stored row-major `(out_features, in_features)` exactly as PyTorch — `matmul(out,x,w,n,d)`
computes `out[o] = Σ_i w[o·n+i]·x[i]`.

**Version 2 (Q8_0):** identical layout, but every *matmul* weight (c_attn_q/k/v, c_proj, w1/w3/w2,
wte, lm_head) is stored group-quantized — `int8[numel]` immediately followed by `f32[numel/group_size]`
scales (symmetric, `scale = max|w|/127` per group of `group_size` consecutive row-major elements).
Norm gains stay fp32. `group_size` (default 64) is chosen so it divides every matmul in-dim (`n_embd`,
each `n_head·v_dim`, each `mlp_hidden`); it backs off by halving if needed. Because every in-dim is a
multiple of `group_size`, each row's groups align exactly with the flat quantization groups, and every
fp32 sub-block stays 4-byte aligned (int8 blocks have `numel % 4 == 0`) — safe for ARM.

## `run_llmforge.c` (fp32 runner)

- `mmap` the file; parse header + descriptor table; set up per-layer weight pointers by walking the
  weight region using each descriptor's sizes.
- Scratch buffers sized to the **per-layer max**: `q(max n_head·qk)`, `att(max n_head × seq_len)`,
  `headcat(max n_head·vd)`, `mlp hidden(max mlp_hidden)`, plus `x, xb, xb2 (n_embd)`.
- **KV cache**: per-layer arrays `key_cache[l] = seq_len·n_kv·qk`, `value_cache[l] = seq_len·n_kv·vd`
  (identity layers allocate none).
- Per token at position `pos`, per layer: run the block above; identity layers skip q/k/v/c_proj and
  set `attn_out = x_attn_in`.
- Reuse `bpe.h` for the gpt2 tokenizer; reuse the `-I ids_file` and `-g tokenizer.bin` CLI paths and
  greedy (`-t 0`) generation already added to `run.c`.

## Validation (goldfish-elephant) — RESULTS

1. **Design gated** by a fresh-context reviewer that independently read the LLMForge source +
   checkpoint shapes: all 10 architectural claims CONFIRMED (concat-not-sum, `1/sqrt(qk_dim)` scale,
   identity residual, no-eps RMSNorm, interleaved RoPE, GQA distribution, tied classifier, no active
   softcap/embedding-scale/residual-alpha). One actionable note — MLP uses **exact erf GELU** — which
   the runner honors.
2. **Golden reference:** `llmforge_bridge/ref_dump.py` runs the real PyTorch `GPT` (fp32, CPU, greedy),
   fully independent of the `.rlm` file and the C runner (shares only checkpoint weights).
3. **Token-exact parity achieved** (`-t 0` greedy, `-I` token-ids in / token-ids out):

   | model | layers | shape notes | tokens | result |
   |-------|--------|-------------|--------|--------|
   | nsga_best3_rotary_periln_105M | 7 | infinite×6 + identity, 3 prompts | 32/48/64/80 | ✅ identical |
   | nsga_af200_b_155M | 10 | pure infinite (no identity) | 40 | ✅ identical |
   | nsga_g30_best1_365M | 32 | n_embd=1088, infinite+identity, qk 32–96 | 40 | ✅ identical |

   Build: `make runllmforge`. Reproduce, e.g.:
   ```bash
   python llmforge_bridge/export_llmforge_hetero.py <ckpt_dir> model.rlm
   python llmforge_bridge/ref_dump.py <ckpt_dir> ids.txt 40 > ref.txt
   ./run_llmforge model.rlm -I ids.txt -n 40 -t 0 > c.txt
   diff c.txt ref.txt        # identical
   ```
   Text generation: `./run_llmforge model.rlm -g tokenizer_gpt2.bin -i "Once upon a time" -t 0.8 -p 0.9`.
4. **26 NSGA infinite-attention checkpoints** in the backup are now exportable/runnable (were 0 before).

Parity is greedy/token-level (argmax), not bitwise-identical logits; that is the meaningful
correctness bar for inference and matches how the earlier Tier-1/Tier-2 models were validated.

## Q8_0 quantization (`runq_llmforge.c`) — RESULTS

`runq_llmforge.c` (`make runqllmforge`) is the int8 variant: same forward math, but the projection
matmuls quantize the activation to int8 and do integer dot-products against int8 weights with fp32
group scales (primitives copied verbatim from the repo's proven `runq.c`). Norms and the KV cache stay
fp32; the token embedding is dequantized once at load for the row lookup.

- **~4× smaller**: nsga_best3 fp32 `.rlm` 416 MB → Q8 `.rlm` 110 MB (max quant err 0.0068).
- **Tracks fp32 then diverges** (expected int8 cascade at logit ties; matches the earlier smollm2
  25/32 finding). If the Q8 matmul/format walk were wrong it would garble from token 1 instead:

  | model | Q8 vs fp32 (greedy) | text |
  |-------|---------------------|------|
  | nsga_best3 (repetitive prompt) | 64/64 identical | coherent |
  | nsga_best3 (diverse prompt) | tracks 14 tok, then diverges | coherent |
  | nsga_af200_b_155M | 40/40 identical | coherent |
  | nsga_g30_best1_365M | 40/40 identical | coherent |

  ```bash
  make runqllmforge
  python llmforge_bridge/export_llmforge_hetero.py <ckpt_dir> model.q8.rlm --version 2
  ./runq_llmforge model.q8.rlm -g tokenizer_gpt2.bin -i "Once upon a time" -t 0.8 -p 0.9
  ```
- Version-guarded: `run_llmforge` rejects v2 files and `runq_llmforge` rejects v1, each with a clear message.

## Out of scope

Android build (next), and other infinite-attention sub-options (`use_concat_heads=False` sum path,
`n_cproj>1`, qk/v-norm, flash-lobo, MoE, non-`add` residual). The exporter's `validate()` rejects any
checkpoint using an unsupported option, so it never silently emits a wrong model. The descriptor table
+ `attn_type` field leave room to add these later without a format break.
