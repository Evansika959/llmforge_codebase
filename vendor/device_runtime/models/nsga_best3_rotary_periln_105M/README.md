# Example model: nsga_best3_rotary_periln_105M

A ready-to-run **NSGA-searched LLMForge model** that uses **infinite-head attention**, exported to
the `.rlm` format so you have a heterogeneous-architecture model to run immediately after cloning.

Unlike the uniform Tier-1 models (e.g. `../smollm2_135M`), this one has **per-layer heterogeneous
dims**, `infinite` attention (concat path, non-square `c_proj`) in layers 0–5, an `identity` attention
layer (6), GQA, peri-LN, GeGLU + erf-GELU, and RoPE. It needs the dedicated `run_llmforge` / `runq_llmforge`
engines (not `run.c`/`runq.c`). Design + validation: [`../../doc/llmforge_hetero_infinite.md`](../../doc/llmforge_hetero_infinite.md).

| layer | attn      | n_head | n_kv | qk_dim | v_dim | mlp_hidden |
|-------|-----------|--------|------|--------|-------|------------|
| 0–2   | infinite  | 8      | 2    | 320    | 512   | 768        |
| 3–5   | infinite  | 8      | 4    | 256    | 224   | 3584       |
| 6     | identity  | –      | –    | –      | –     | 1280       |

dim 768, vocab 50257 (tiktoken-gpt2), block_size 1024, tied embeddings.

## Files
- `tokenizer_gpt2.bin` — GPT-2 byte-level BPE tokenizer (~510 KB). **Committed in git.**
- `nsga_best3_rotary_periln_105M.q8.rlm` — Q8_0 quantized weights (~106 MB). **Git LFS** (see
  `.gitattributes`); run `git lfs pull` after cloning to fetch it.

## Run (desktop)
```bash
# from the repo root
make runqllmforge          # builds runq_llmforge (Q8 engine); use `make runllmforge` for the fp32 engine
./runq_llmforge models/nsga_best3_rotary_periln_105M/nsga_best3_rotary_periln_105M.q8.rlm \
              -g models/nsga_best3_rotary_periln_105M/tokenizer_gpt2.bin \
              -i "Once upon a time" -t 0.8 -p 0.9 -n 128
```
- `-t 0` = deterministic greedy (repetitive on a small model); `-t 0.8 -p 0.9` = varied sampling.
- Multithreaded build (bigger speedup): `make runllmforgeomp` (needs `libomp` on macOS).

## Notes on Q8
Q8_0 int8 weights are ~4× smaller than fp32 but lossy: greedy Q8 tracks the fp32/PyTorch output for a
number of tokens, then diverges at logit ties (expected int8 behavior) while staying coherent. For
bit-exact-vs-PyTorch behavior use the fp32 build instead:
```bash
python llmforge_bridge/export_llmforge_hetero.py <ckpt_dir> model.rlm --version 1   # fp32 .rlm
make runllmforge && ./run_llmforge model.rlm -g .../tokenizer_gpt2.bin -i "..."
```

## Regenerate (optional)
```bash
# from the repo root, from a LLMForge checkpoint directory (containing ckpt.pt)
python llmforge_bridge/export_llmforge_hetero.py <ckpt_dir> \
    models/nsga_best3_rotary_periln_105M/nsga_best3_rotary_periln_105M.q8.rlm --version 2
python llmforge_bridge/export_gpt2_tokenizer.py models/nsga_best3_rotary_periln_105M/tokenizer_gpt2.bin
```
