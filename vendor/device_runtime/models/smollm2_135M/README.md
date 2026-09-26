# Example model: smollm2_135M

A ready-to-run LLMForge model exported to llama2.c format, so you have something to run
immediately after cloning. This is a **Tier-1** model (RoPE + GeGLU + RMSNorm + GQA, tiktoken-gpt2
tokenizer); its output is bit-exact vs the original PyTorch model (see
[`../../doc/llmforge2c.md`](../../doc/llmforge2c.md)).

## Files
- `tokenizer_gpt2.bin` — GPT-2 byte-level BPE tokenizer table (~510 KB). **Committed in git.**
- `smollm2_135M.q8.bin` — 8-bit quantized weights (~138 MB). **NOT in git** (GitHub blocks files
  >100 MB and this public fork disallows Git LFS). Get it one of two ways:
  ```bash
  # 1) download from the GitHub Release (see the repo's Releases page for the asset URL)
  MODEL_URL="https://github.com/<you>/nanollm.c/releases/download/<tag>/smollm2_135M.q8.bin" \
    llmforge_bridge/fetch_model.sh
  # 2) or regenerate from a LLMForge checkpoint
  python llmforge_bridge/export_llmforge.py <ckpt_dir> models/smollm2_135M/smollm2_135M.q8.bin --version 2
  ```

## Run (desktop)
```bash
# from the repo root
make rungelu          # builds run_gelu (fp32) and runq_gelu (Q8)
./runq_gelu models/smollm2_135M/smollm2_135M.q8.bin \
            -g models/smollm2_135M/tokenizer_gpt2.bin \
            -i "Once upon a time" -t 0.8 -p 0.9 -n 128
```
- `-t 0` = deterministic greedy (repetitive on a small model); `-t 0.8 -p 0.9` = varied sampling.
- `-g` selects the GPT-2 BPE tokenizer path (required for this model).

## Run on Android
Build the arm64 engine and push the two files — see the "Android deploy" section of the design doc.

## Regenerate (optional)
Run from the repo root; write directly into this directory so the files land under the LFS pattern
(`models/**/*.q8.bin`):
```bash
python llmforge_bridge/export_llmforge.py <ckpt_dir> models/smollm2_135M/smollm2_135M.q8.bin --version 2
python llmforge_bridge/export_gpt2_tokenizer.py models/smollm2_135M/tokenizer_gpt2.bin
```
