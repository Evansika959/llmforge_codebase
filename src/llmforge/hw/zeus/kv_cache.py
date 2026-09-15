"""Measurement-only KV cache shim for ZEUS HW eval.

Adds proper KV-cached decode to Evo_GPT for power/energy measurement only.
DOES NOT modify Evo_GPT itself. Works by:

  1. Monkey-patching each `InfiniteHeadAttention` layer's `forward()` with a
     mode-aware version that supports "capture" (during prefill) and
     "decode" (single-token, single-query SDPA against cached K, V).
  2. Bypassing `GPT.forward()` for decode steps (since it hardcodes
     position = arange(0, t)) and instead running an explicit per-token
     loop: wte → +wpe(pos_offset) → blocks → ln_f → lm_head.

Why this matters
----------------
Evo_GPT's `model.generate()` recomputes the full prefill+generated context
on every step (no `past_key_values`). For a NSGA-II co-search that uses
ZEUS to measure decode-time power and energy, that means:

  • `tpot` and `energy_per_token` are inflated by an O(N) recompute factor.
  • Decode looks compute-bound (prefill-style GEMM) instead of
    memory-bound (cached GEMV). The reported `power_W` is closer to
    *prefill power* than *steady-state decode power*.
  • Cross-arch ranking is biased: GQA (`n_kv_group < n_head`) and
    attention-variant differences that benefit cached decode are mostly
    invisible without a real cache.

This shim restores production-representative cached decode for the
narrow purpose of HW measurement.

Supported feature set (validated against search_space_200M.yaml)
----------------------------------------------------------------
attention_variant ∈ {"infinite", "identity"}: ✓
  - "identity" is `nn.Identity`-style — no state, nothing to cache.
  - "infinite" (IHA) is patched with mode-aware forward.

The cached path supports the IHA configuration that the search space
actually exposes today. At attach time we hard-assert these flags so
silent miscounting is impossible:

  - use_rotary_embeddings = False
  - use_qk_norm = False / use_v_norm = False / use_qk_norm_scale = False
  - use_flash_lobo = False
  - disable_flash_attention = False (we want SDPA)
  - softmax_variant_attn = "softmax"
  - l2_norm_attn_q / _k / _v / _cproj = False
  - post_act_l2_norm = False, cproj_scale ∈ {None, 1.0}
  - use_concat_heads ∈ {True, False}, n_cproj ≥ 1: all supported

If any active layer fails the guard, the caller falls back to the
non-cached path and surfaces it via the result dict.

API
---
  attach_iha_kv_cache(model)             # may raise UnsupportedKVCache
  detach_iha_kv_cache(model)
  set_iha_mode(model, "off"|"capture"|"decode")
  clear_iha_kv_state(model)
  run_cached_decode(model, prefill_len, decode_len, ...)
  parity_check(model, prefill_len, decode_len, device, dtype, atol, rtol)
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)


class UnsupportedKVCache(Exception):
    """Raised when the model has an IHA configuration the cached path doesn't support."""


# --------------------------------------------------------------------------
# IHA detection (lazy; no import-time dependency on Evo_GPT)
# --------------------------------------------------------------------------

def _is_iha(module: torch.nn.Module) -> bool:
    """Duck-type check: an IHA has c_attn_q/k/v + n_head + n_kv_group +
    n_qk_head_dim + n_v_head_dim + _expand_kv. Avoids importing the
    Evo_GPT class to keep this shim decoupled."""
    needed = ("c_attn_q", "c_attn_k", "c_attn_v",
              "n_head", "n_kv_group", "n_qk_head_dim", "n_v_head_dim",
              "_expand_kv")
    return all(hasattr(module, n) for n in needed)


def _iha_modules(model: torch.nn.Module) -> List[torch.nn.Module]:
    out = []
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        for blk in model.transformer.h:
            attn = getattr(blk, "attn", None)
            if attn is not None and _is_iha(attn):
                out.append(attn)
    return out


# --------------------------------------------------------------------------
# Feature-flag guard at attach time
# --------------------------------------------------------------------------

_FORBIDDEN_TRUE = (
    "use_qk_norm", "use_v_norm", "use_qk_norm_scale",
    "use_flash_lobo", "use_flash_lobo_per_head",
    "l2_norm_attn_q", "l2_norm_attn_k", "l2_norm_attn_v", "l2_norm_attn_cproj",
    "post_act_l2_norm",
)


def _assert_supported_iha(iha: torch.nn.Module, layer_idx: int) -> None:
    for flag in _FORBIDDEN_TRUE:
        if getattr(iha, flag, False):
            raise UnsupportedKVCache(
                f"IHA layer {layer_idx} has {flag}=True; cached path doesn't "
                f"support this feature yet. Disable it or extend zeus_kv_cache.py.")
    # rotary
    if getattr(iha, "rotary_emb_q", None) is not None or getattr(iha, "rotary_emb_k", None) is not None:
        raise UnsupportedKVCache(
            f"IHA layer {layer_idx} has rotary positional embeddings; cached "
            f"path doesn't yet thread the position offset through rotary. "
            f"Set use_rotary_embeddings=False or extend zeus_kv_cache.py.")
    # flash on
    if getattr(iha, "disable_flash_attention", False):
        raise UnsupportedKVCache(
            f"IHA layer {layer_idx} has disable_flash_attention=True; "
            f"cached path is SDPA-only.")
    # plain softmax
    if getattr(iha, "softmax_variant_attn", "softmax") != "softmax":
        raise UnsupportedKVCache(
            f"IHA layer {layer_idx} uses softmax_variant_attn="
            f"{iha.softmax_variant_attn!r}; cached path requires plain softmax.")
    # cproj_scale ≈ 1
    cproj_scale = getattr(iha, "cproj_scale", 1.0)
    if cproj_scale not in (None, 1.0):
        raise UnsupportedKVCache(
            f"IHA layer {layer_idx} has cproj_scale={cproj_scale}; cached path "
            f"only supports cproj_scale=1.0/None.")


# --------------------------------------------------------------------------
# Mode-aware forward for IHA
# --------------------------------------------------------------------------

_GQA_SUPPORTED: Optional[bool] = None


def _sdpa_gqa_supported() -> bool:
    """Whether scaled_dot_product_attention accepts enable_gqa in this PyTorch build."""
    global _GQA_SUPPORTED
    if _GQA_SUPPORTED is None:
        try:
            q, k = torch.zeros(1, 2, 1, 2), torch.zeros(1, 1, 1, 2)
            F.scaled_dot_product_attention(q, k, k, enable_gqa=True)
            _GQA_SUPPORTED = True
        except TypeError:
            _GQA_SUPPORTED = False
    return _GQA_SUPPORTED


def _attend(self: torch.nn.Module, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
            is_causal: bool) -> torch.Tensor:
    """Grouped-query attention without materializing repeated key and value heads.

    With enable_gqa, query head i reads KV group i // (n_head / n_kv_group), the same grouping as
    the model's _expand_kv, so the result matches the training forward. Builds without enable_gqa
    fall back to the explicit expansion.
    """
    if self.n_head != self.n_kv_group and _sdpa_gqa_supported():
        return F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0,
                                              is_causal=is_causal, enable_gqa=True)
    return F.scaled_dot_product_attention(q, self._expand_kv(k), self._expand_kv(v), attn_mask=None,
                                          dropout_p=0.0, is_causal=is_causal)


def _iha_capture_forward(self: torch.nn.Module, x: torch.Tensor, iter_num=None) -> torch.Tensor:
    """Same compute as original IHA.forward (in the supported feature subset),
    and write post-projection K, V into a preallocated cache.

    Inputs:
        x: (B, T, n_embd)
    Output:
        y: (B, T, n_embd)
    Side effect:
        self._kv_state = (K_buf, V_buf) with shapes (B, n_kv, capacity, qk_dim) and
        (B, n_kv, capacity, v_dim), holding T valid positions (self._kv_len = T).

    The cache is allocated once per prefill at the capacity set by set_kv_capacity, as a serving
    engine would, so decode steps write one position in place instead of copying the whole cache.
    """
    B, T, _ = x.size()
    q = self.c_attn_q(x)
    k = self.c_attn_k(x)
    v = self.c_attn_v(x)
    q = q.view(B, T, self.n_head, self.n_qk_head_dim).transpose(1, 2)
    k = k.view(B, T, self.n_kv_group, self.n_qk_head_dim).transpose(1, 2)
    v = v.view(B, T, self.n_kv_group, self.n_v_head_dim).transpose(1, 2)

    capacity = max(T, int(getattr(self, "_kv_capacity", 0) or 0))
    state = getattr(self, "_kv_state", None)
    if (state is not None and state[0].shape == (B, self.n_kv_group, capacity, self.n_qk_head_dim)
            and state[1].shape == (B, self.n_kv_group, capacity, self.n_v_head_dim)
            and state[0].dtype == k.dtype):
        # Reuse the buffers in place. Captured CUDA graphs hold pointers into them.
        k_buf, v_buf = state
    else:
        k_buf = k.new_empty(B, self.n_kv_group, capacity, self.n_qk_head_dim)
        v_buf = v.new_empty(B, self.n_kv_group, capacity, self.n_v_head_dim)
    k_buf[:, :, :T].copy_(k.detach())
    v_buf[:, :, :T].copy_(v.detach())
    self._kv_state = (k_buf, v_buf)
    self._kv_len = T

    y = _attend(self, q, k, v, is_causal=True)
    return _iha_project_heads(self, y, B, T)


def _iha_decode_forward(self: torch.nn.Module, x: torch.Tensor, iter_num=None) -> torch.Tensor:
    """Single-token cached decode step.

    Inputs:
        x: (B, 1, n_embd)
    Output:
        y: (B, 1, n_embd)
    Side effect:
        writes K, V of the new token at position self._kv_len of the preallocated cache and
        advances self._kv_len by one.
    """
    state = getattr(self, "_kv_state", None)
    if state is None:
        raise RuntimeError(
            "IHA._kv_state is empty in decode mode. Run a capture-mode "
            "forward over the prefill before invoking decode steps.")
    B, T_new, _ = x.size()
    if T_new != 1:
        raise RuntimeError(f"decode mode expects T=1, got T={T_new}")

    k_buf, v_buf = state                                # (B, n_kv, capacity, *)
    L = int(self._kv_len)
    if L >= k_buf.size(2):
        # Capacity exhausted: grow once by doubling. Measurements size the cache up front, so
        # this path only serves callers that did not.
        k_buf = torch.cat([k_buf, torch.empty_like(k_buf)], dim=2)
        v_buf = torch.cat([v_buf, torch.empty_like(v_buf)], dim=2)
        self._kv_state = (k_buf, v_buf)
    q = self.c_attn_q(x)
    k_new = self.c_attn_k(x)
    v_new = self.c_attn_v(x)
    q = q.view(B, 1, self.n_head, self.n_qk_head_dim).transpose(1, 2)
    k_new = k_new.view(B, 1, self.n_kv_group, self.n_qk_head_dim).transpose(1, 2)
    v_new = v_new.view(B, 1, self.n_kv_group, self.n_v_head_dim).transpose(1, 2)

    k_buf[:, :, L:L + 1].copy_(k_new)
    v_buf[:, :, L:L + 1].copy_(v_new)
    self._kv_len = L + 1

    # Single query: causal mask is degenerate (one row, all keys allowed),
    # so is_causal=False is the correct setting here.
    y = _attend(self, q, k_buf[:, :, :L + 1], v_buf[:, :, :L + 1], is_causal=False)
    return _iha_project_heads(self, y, B, 1)


def _iha_project_heads(self: torch.nn.Module, y: torch.Tensor, B: int, T: int) -> torch.Tensor:
    """c_proj branch matching the three IHA layouts; mirrors lines 1281-1313
    of attention_variations.py."""
    if self.use_concat_heads:
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.n_v_head_dim)
        y = self.c_proj(y)
    elif self.n_cproj == 1:
        y = y.sum(dim=1)
        y = self.c_proj(y)
    else:
        y_sum = y.sum(dim=1)
        proj_outputs = [proj(y_sum) for proj in self.c_proj_list]
        y = torch.stack(proj_outputs, dim=0).sum(dim=0)
    return self.resid_dropout(y)


def _make_dispatcher(iha: torch.nn.Module):
    """Build a forward function bound to this IHA that dispatches by
    self._kv_cache_mode. Closes over the original forward for "off"."""
    orig_forward = iha.forward

    def dispatched(x: torch.Tensor, iter_num=None) -> torch.Tensor:
        mode = getattr(iha, "_kv_cache_mode", "off")
        if mode == "off":
            return orig_forward(x, iter_num)
        if mode == "capture":
            return _iha_capture_forward(iha, x, iter_num)
        if mode == "decode":
            return _iha_decode_forward(iha, x, iter_num)
        raise ValueError(f"Unknown _kv_cache_mode={mode!r}")

    return dispatched, orig_forward


# --------------------------------------------------------------------------
# Public attach / detach / mode / clear
# --------------------------------------------------------------------------

def attach_iha_kv_cache(model: torch.nn.Module) -> int:
    """Patch every IHA layer in `model` to support cached forward.

    Returns the number of patched layers. Raises UnsupportedKVCache if any
    layer's feature flags fall outside the supported subset.
    Idempotent — re-attaching is a no-op."""
    ihas = _iha_modules(model)
    n_patched = 0
    for i, iha in enumerate(ihas):
        if getattr(iha, "_kv_cache_attached", False):
            n_patched += 1
            continue
        _assert_supported_iha(iha, i)
        dispatched, orig = _make_dispatcher(iha)
        iha._orig_forward_pre_kv_cache = orig
        iha.forward = dispatched
        iha._kv_cache_mode = "off"
        iha._kv_state = None
        iha._kv_cache_attached = True
        n_patched += 1
    return n_patched


def detach_iha_kv_cache(model: torch.nn.Module) -> None:
    """Restore the original IHA.forward on every patched layer and drop state."""
    for iha in _iha_modules(model):
        if not getattr(iha, "_kv_cache_attached", False):
            continue
        iha.forward = iha._orig_forward_pre_kv_cache
        del iha._orig_forward_pre_kv_cache
        iha._kv_cache_mode = "off"
        iha._kv_state = None
        iha._kv_cache_attached = False


def set_iha_mode(model: torch.nn.Module, mode: str) -> None:
    if mode not in ("off", "capture", "decode"):
        raise ValueError(f"mode must be off|capture|decode, got {mode!r}")
    for iha in _iha_modules(model):
        if getattr(iha, "_kv_cache_attached", False):
            iha._kv_cache_mode = mode


def clear_iha_kv_state(model: torch.nn.Module) -> None:
    for iha in _iha_modules(model):
        if getattr(iha, "_kv_cache_attached", False):
            iha._kv_state = None
            iha._kv_len = 0


def set_kv_capacity(model: torch.nn.Module, capacity: int) -> None:
    """Token capacity of the cache each capture-mode prefill allocates, prefill plus decode."""
    for iha in _iha_modules(model):
        if getattr(iha, "_kv_cache_attached", False):
            iha._kv_capacity = int(capacity)


def set_iha_kv_len(model: torch.nn.Module, length: int) -> None:
    """Set the next write position of every layer's cache, keeping its buffers."""
    for iha in _iha_modules(model):
        if getattr(iha, "_kv_cache_attached", False):
            iha._kv_len = int(length)


# --------------------------------------------------------------------------
# Per-token embedding (replaces GPT.forward's prelude for decode steps)
# --------------------------------------------------------------------------

def _embed_one_token(model: torch.nn.Module, tok_id: torch.Tensor,
                     position_offset: int) -> torch.Tensor:
    """Replicate GPT.forward's pre-block embedding pipeline for a single
    new token at absolute position `position_offset`.

    Mirrors model.py:520-549 (the non-token_dict path) for the supported
    feature subset. Only the features actually exercised by the search
    space are implemented; anything else trips an assert at attach time
    or a clear runtime error here.
    """
    cfg = model.config
    if getattr(cfg, "multidataset_wte", False) or getattr(cfg, "multicontext", False):
        raise UnsupportedKVCache(
            "multidataset_wte/multicontext models are not supported by the cached "
            "decode path; disable for HW measurement runs.")
    if getattr(model, "use_lsv", False):
        raise UnsupportedKVCache("use_lsv=True not supported in cached decode.")
    if getattr(model, "use_ln_f_input_mixer", False):
        raise UnsupportedKVCache("use_ln_f_input_mixer=True not supported in cached decode.")

    tok_emb = model.transformer.wte(tok_id)              # (B, 1, n_embd_wte_or_n_embd)
    if getattr(model, "n_embd_wte", None):
        tok_emb = model.transformer.scale_up(tok_emb)
    if getattr(cfg, "use_embedding_scale", False):
        tok_emb = tok_emb * model.embedding_scale
    if getattr(cfg, "norm_variant_wte", None) is not None:
        tok_emb = model.transformer.post_embedding_norm(tok_emb)

    if getattr(cfg, "use_abs_pos_embeddings", False):
        device = tok_id.device
        pos = torch.tensor([position_offset], dtype=torch.long, device=device)
        pos_emb = model.transformer.wpe(pos)             # (1, n_embd)
        x = tok_emb + pos_emb                            # broadcast over batch dim
        if getattr(cfg, "norm_variant_abs", None) is not None:
            x = model.transformer.post_abs_norm(x)
        x = model.transformer.drop(x)
    else:
        x = model.transformer.drop(tok_emb)
    return x


# --------------------------------------------------------------------------
# Cached decode loop
# --------------------------------------------------------------------------

def run_cached_decode(model: torch.nn.Module,
                      prefill_len: int,
                      decode_len: int,
                      batch_size: int,
                      device: torch.device,
                      vocab_size: Optional[int] = None,
                      run_lm_head: bool = True,
                      decode_token_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Run `decode_len` cached decode steps. Returns the last step's hidden
    state post-ln_f (B, 1, n_embd) — useful for parity checks. Logits from
    `lm_head` are computed if `run_lm_head=True` (matches production
    workload but result is discarded).

    Preconditions:
      - `attach_iha_kv_cache(model)` has been called.
      - A capture-mode prefill has populated each IHA._kv_state for length
        `prefill_len`.
      - `set_iha_mode(model, "decode")` was called.
    """
    cfg = model.config
    if vocab_size is None:
        vocab_size = int(getattr(cfg, "vocab_size", 50304))

    if decode_token_ids is None:
        decode_token_ids = torch.randint(
            0, vocab_size, (batch_size, decode_len), device=device, dtype=torch.long)
    elif decode_token_ids.shape != (batch_size, decode_len):
        raise ValueError(f"decode_token_ids shape {tuple(decode_token_ids.shape)} "
                         f"!= ({batch_size}, {decode_len})")

    last_hidden = None
    for step in range(decode_len):
        last_hidden = decode_step(model, decode_token_ids[:, step:step + 1], prefill_len + step,
                                  run_lm_head=run_lm_head)
    return last_hidden


def decode_step(model: torch.nn.Module, tok_id: torch.Tensor, position: int,
                run_lm_head: bool = True) -> torch.Tensor:
    """One cached decode step for tokens `tok_id` (B, 1) at absolute `position`.

    Returns the hidden state after ln_f. The step launches a fixed kernel sequence for a given
    position, so it can be captured as a CUDA graph.
    """
    x = _embed_one_token(model, tok_id, position_offset=position)
    for blk in model.transformer.h:
        x = blk(x, None)                                # routes through cached IHA
    x = model.transformer.ln_f(x)
    if getattr(model, "n_embd_wte", None):
        x = F.linear(x, model.transformer.scale_down.weight.t())
    if run_lm_head:
        _ = model.lm_head(x)
    return x


def capture_decode_graphs(model: torch.nn.Module, prefill_len: int, tokens: torch.Tensor):
    """Capture one CUDA graph per decode position after a capture-mode prefill.

    `tokens` (B, decode_len) holds the decode inputs. Graph i writes position prefill_len + i of the
    preallocated cache and attends over positions up to it, so replaying the graphs in order after a
    prefill reproduces cached decoding without per-kernel launch overhead, as serving engines do.
    Returns the graphs and each graph's static output hidden state.
    """
    device = tokens.device
    set_iha_mode(model, "decode")
    # The warmup runs on a side stream and reads what the current stream produced: the cache filled by
    # the prefill and the drawn `tokens`. Wait for that work first. Without the wait, a long prefill let
    # the side stream read token ids that were not yet written, which failed with a device-side assert.
    torch.cuda.current_stream(device).synchronize()
    side = torch.cuda.Stream(device=device)
    with torch.cuda.stream(side):
        for _ in range(3):
            set_iha_kv_len(model, prefill_len)
            decode_step(model, tokens[:, :1], prefill_len)
    torch.cuda.current_stream(device).wait_stream(side)
    pool = torch.cuda.graph_pool_handle()
    graphs, outputs = [], []
    for step in range(tokens.size(1)):
        set_iha_kv_len(model, prefill_len + step)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=pool):
            out = decode_step(model, tokens[:, step:step + 1], prefill_len + step)
        graphs.append(g)
        outputs.append(out)
    set_iha_mode(model, "off")
    return graphs, outputs


@torch.no_grad()
def graph_parity_check(model: torch.nn.Module, prefill_len: int = 24, decode_len: int = 8,
                       batch_size: int = 2) -> float:
    """Max absolute difference between eager cached decoding and CUDA-graph replay."""
    device = next(model.parameters()).device
    vocab = int(getattr(model.config, "vocab_size", 50304))
    prompt = torch.randint(0, vocab, (batch_size, prefill_len), device=device)
    decode_ids = torch.randint(0, vocab, (batch_size, decode_len), device=device)
    set_kv_capacity(model, prefill_len + decode_len)

    def prefill():
        set_iha_kv_len(model, 0)
        set_iha_mode(model, "capture")
        model(prompt)

    prefill()
    set_iha_mode(model, "decode")
    eager = run_cached_decode(model, prefill_len, decode_len, batch_size, device,
                              run_lm_head=False, decode_token_ids=decode_ids).clone()
    prefill()
    graphs, outputs = capture_decode_graphs(model, prefill_len, decode_ids)
    prefill()
    for g in graphs:
        g.replay()
    torch.cuda.synchronize()
    diff = float((outputs[-1].float() - eager.float()).abs().max())
    set_iha_mode(model, "off")
    del graphs, outputs
    clear_iha_kv_state(model)
    return diff


# --------------------------------------------------------------------------
# Parity self-test
# --------------------------------------------------------------------------

@torch.no_grad()
def parity_check(model: torch.nn.Module,
                 prefill_len: int = 16,
                 decode_len: int = 4,
                 device: Optional[torch.device] = None,
                 atol: float = 1e-1,
                 rtol: float = 1e-1,
                 verbose: bool = True) -> Tuple[bool, float, float]:
    """Run a single-arch parity check: cached vs uncached final logits.

    The cached path's last logit (after `decode_len` decode steps) should
    match the uncached path's logit at position `prefill_len + decode_len - 1`
    (when the same token sequence is fed in one shot via `model(full_seq)`).

    Tolerance: bf16 round-off compounds roughly with √n_layers. For shallow
    archs (≤12 layers) max_abs lands ~1-3e-2; for SwiGLU + 30-layer SmolLM2-
    style stacks it lands ~9e-2. Default atol=rtol=1e-1 is set to pass any
    structurally-correct stack in this search space's depth range while
    still catching real cache bugs (which produce ≥10x larger drift). For
    a stricter test on shallower archs, lower atol explicitly.

    Returns (passed, max_abs_err, max_rel_err).
    """
    device = device or next(model.parameters()).device
    vocab = int(getattr(model.config, "vocab_size", 50304))
    full_len = prefill_len + decode_len
    full_seq = torch.randint(0, vocab, (1, full_len), device=device, dtype=torch.long)

    # Path A: uncached single-shot forward
    set_iha_mode(model, "off")
    out_a = model(full_seq)
    logits_a = out_a[0] if isinstance(out_a, tuple) else out_a
    # GPT.forward returns logits sliced to the last position when targets is None,
    # so logits_a is (B, 1, vocab). Also guard the (B, T, vocab) case.
    last_a = logits_a[:, -1, :].float()                  # (1, vocab)

    # Path B: cached prefill + cached decode
    clear_iha_kv_state(model)
    set_kv_capacity(model, full_len)
    set_iha_mode(model, "capture")
    _ = model(full_seq[:, :prefill_len])                 # populates K/V caches
    set_iha_mode(model, "decode")
    decode_ids = full_seq[:, prefill_len:]               # (1, decode_len)
    last_hidden = run_cached_decode(
        model, prefill_len=prefill_len, decode_len=decode_len,
        batch_size=1, device=device,
        run_lm_head=False, decode_token_ids=decode_ids,
    )
    # Compute the final-step logits manually so we always get (B, 1, vocab),
    # mirroring the slice GPT.forward applies in inference mode.
    last_b = model.lm_head(last_hidden)[:, -1, :].float()
    if getattr(model.config, "final_logit_softcapping", None):
        cap = model.config.final_logit_softcapping
        last_b = torch.tanh(last_b / cap) * cap

    set_iha_mode(model, "off")
    clear_iha_kv_state(model)

    diff = (last_a - last_b).abs()
    max_abs = float(diff.max())
    denom = last_a.abs().clamp_min(1e-6)
    max_rel = float((diff / denom).max())
    passed = max_abs <= atol or max_rel <= rtol
    if verbose:
        log.info(f"[kv-cache parity] prefill={prefill_len} decode={decode_len} "
                 f"max_abs={max_abs:.3e}  max_rel={max_rel:.3e}  "
                 f"{'PASS' if passed else 'FAIL'}")
    return passed, max_abs, max_rel


# --------------------------------------------------------------------------
# Convenience: attach-or-skip wrapper for callers that want soft fallback
# --------------------------------------------------------------------------

@contextmanager
def kv_cache_session(model: torch.nn.Module):
    """Context manager: attach on enter, detach on exit. Re-raises
    UnsupportedKVCache so the caller can fall back to non-cached eval."""
    n = attach_iha_kv_cache(model)
    try:
        yield n
    finally:
        detach_iha_kv_cache(model)
