"""Chunked next-token CE and in-place-distillation KL, upcast to fp32 in sequence chunks so
the 151,936-vocab logits don't blow up memory (review #5 / compute). Both operate on shifted
positions (predict token t+1 from t).

lm_ce / distill_kl take pre-computed full logits (used by eval + the v1 loop).
chunked_ce_kd (v2) takes *hidden states* and applies lm_head inside gradient-checkpointed
sequence chunks, so the [B,S,152k] logits are never materialized -- the teacher only holds its
hidden states (~16MB vs ~2.5GB), letting the micro-batch grow for higher throughput."""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def lm_ce(logits, input_ids, chunk=1024):
    logits = logits[:, :-1]
    targets = input_ids[:, 1:]
    B, T, V = logits.shape
    logits = logits.reshape(B * T, V)
    targets = targets.reshape(B * T)
    total, n = 0.0, 0
    step = chunk * B
    for i in range(0, B * T, step):
        total = total + F.cross_entropy(logits[i:i + step].float(), targets[i:i + step], reduction="sum")
        n += targets[i:i + step].numel()
    return total / n


def distill_kl(student_logits, teacher_logits, T=1.0, chunk=1024):
    """KL(teacher || student) with temperature T, scaled by T^2 (standard KD)."""
    s = student_logits[:, :-1]
    t = teacher_logits[:, :-1]
    B, Tt, V = s.shape
    s = s.reshape(B * Tt, V)
    t = t.reshape(B * Tt, V)
    total, n = 0.0, 0
    step = chunk * B
    for i in range(0, B * Tt, step):
        logp_s = F.log_softmax(s[i:i + step].float() / T, dim=-1)
        p_t = F.softmax(t[i:i + step].float() / T, dim=-1)
        total = total + (p_t * (p_t.clamp_min(1e-9).log() - logp_s)).sum()
        n += logp_s.shape[0]
    return (total / n) * (T * T)


def chunked_ce_kd(hidden_s, lm_head_s, input_ids, hidden_t=None, lm_head_t=None,
                  kd_weight=0.0, tau=2.0, chunk=1024):
    """Per-token mean of (CE + kd_weight * KD), computed in gradient-checkpointed sequence
    chunks so no full [B,S,V] logits tensor exists. hidden_s requires grad (student backbone);
    hidden_t is the frozen teacher's hidden states (no grad). KD uses KL(teacher||student)*tau^2.
    Returns a scalar equal (up to fp32 reassociation) to lm_ce + kd_weight*distill_kl on the
    same logits."""
    B, S, H = hidden_s.shape
    V = lm_head_s.weight.shape[0]
    targets = input_ids[:, 1:]
    use_kd = kd_weight > 0 and hidden_t is not None
    total = hidden_s.new_zeros(())
    n = 0
    for start in range(0, S - 1, chunk):
        end = min(start + chunk, S - 1)
        tgt = targets[:, start:end].reshape(-1)

        def compute(h, _start=start, _end=end, _tgt=tgt):
            ls = lm_head_s(h).reshape(-1, V).float()
            ce = F.cross_entropy(ls, _tgt, reduction="sum")
            if use_kd:
                with torch.no_grad():
                    lt = (lm_head_t(hidden_t[:, _start:_end]).reshape(-1, V).float()) / tau
                    pt = F.softmax(lt, dim=-1)
                logp_s = F.log_softmax(ls / tau, dim=-1)
                kd = (pt * (pt.clamp_min(1e-9).log() - logp_s)).sum() * (tau * tau)
                return ce + kd_weight * kd
            return ce

        total = total + checkpoint(compute, hidden_s[:, start:end], use_reentrant=False)
        n += tgt.numel()
    return total / n
