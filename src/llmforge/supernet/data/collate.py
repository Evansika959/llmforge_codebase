"""Doc-aware attention mask + per-doc position_ids from packed segment lengths.

For a packed sequence with segment lengths [L0, L1, ...] (summing to SEQLEN), a token may
attend only to earlier-or-equal tokens *within its own segment* (block-diagonal causal), and
RoPE position_ids restart at 0 for each segment. This is what makes intra-doc long-range the
only learnable signal and blocks spurious cross-doc attention (review #G).

Returns an additive float mask [B,1,S,S] (0 = attend, -inf = block) consumable by the
elastic attention forward (`attn = attn + mask`), plus position_ids [B,S] (long).
"""
import torch

NEG = float("-inf")


def build_docaware(seglens_batch, seqlen, device="cpu", dtype=torch.float32, neg=NEG):
    # neg: masked-position value. -inf for the eager softmax path; a large finite negative
    # (e.g. -1e9) for the SDPA training path, which can NaN on -inf with some backends.
    B = len(seglens_batch)
    mask = torch.full((B, 1, seqlen, seqlen), neg, device=device, dtype=dtype)
    pos = torch.zeros((B, seqlen), dtype=torch.long, device=device)
    ar = torch.arange(seqlen, device=device)
    for b, seglens in enumerate(seglens_batch):
        off = 0
        for L in seglens:
            if L <= 0:
                continue
            causal = torch.triu(torch.full((L, L), neg, device=device, dtype=dtype), diagonal=1)
            mask[b, 0, off:off + L, off:off + L] = causal
            pos[b, off:off + L] = ar[:L]
            off += L
    return mask, pos
