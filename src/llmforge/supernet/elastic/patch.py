"""Model-level API: install the elastic forwards and select one sub-network.

Lives apart from attention.py and mlp.py because it spans both sublayers -- `set_elastic_config`
sets attention knobs (d_qk, d_v, n_h) and the MLP knob (d_mlp) in one call.
"""
import torch

from .attention import ABLATION, elastic_attention_forward, head_index, qk_gather_index
from .families import all_attn_classes, family
from .mlp import elastic_mlp_forward

# Stock forwards are captured per family the first time that family is patched, so a run touching
# only one family never imports the other's modeling module.
_ORIG: dict[str, tuple] = {}


def enable_elastic(model):
    """Install the elastic forwards for whichever family `model` belongs to.

    Patching is class-level, as the original Qwen3-only version was: one process cannot hold an
    elastic model and a stock model of the SAME family at once. Different families are
    independent, so patching Llama leaves Qwen3 alone.
    """
    f = family(model)
    # Qwen3's stock forward passes sliding_window to the attention interface (upstream marks it
    # "diff with Llama"); the elastic forward does not. Every published Qwen3 base ships
    # use_sliding_window: false, so this has never mattered -- but that is a property of the
    # checkpoint, not of the architecture, so check it instead of assuming it.
    if getattr(model.config, "use_sliding_window", False):
        raise NotImplementedError(
            f"{getattr(model.config, 'model_type', '?')} has use_sliding_window=True; the elastic "
            f"forward ignores sliding_window and would attend over the full context.")
    if f.name not in _ORIG:
        _ORIG[f.name] = (f.attn_cls.forward, f.mlp_cls.forward)
    f.attn_cls.forward = elastic_attention_forward
    f.mlp_cls.forward = elastic_mlp_forward
    return model


def disable_elastic(model=None):
    """Restore the stock forwards, and with a model, the state set_elastic_config overwrote.

    num_key_value_groups is a real attribute of the module, not something the elastic forward
    owns: set_elastic_config lowers it to drive repeat_kv when n_h shrinks. Leaving it lowered
    makes the *stock* forward fail on a head-count mismatch, so restore it here.
    """
    for n in ([family(model).name] if model is not None else list(_ORIG)):
        if n in _ORIG:
            f = family(n)
            f.attn_cls.forward, f.mlp_cls.forward = _ORIG[n]
    if model is not None:
        for m in model.modules():
            if isinstance(m, all_attn_classes()):
                m.num_key_value_groups = m.config.num_attention_heads // m.config.num_key_value_heads
                for attr in ("_q_idx", "_qk_idx", "qk_active", "v_active", "n_h_active",
                             "_family"):
                    if hasattr(m, attr):
                        delattr(m, attr)
        for L in model.model.layers:
            # Inert under the stock MLP forward, but a later enable_elastic without a matching
            # set_elastic_config would run a silently narrowed MLP.
            if hasattr(L.mlp, "mlp_active"):
                del L.mlp.mlp_active
        # logit_scale is deliberately NOT removed: add_elastic_temperature is a separate call and
        # the parameter carries trained state. disable_elastic is the inverse of enable_elastic.


def _per_layer(x, n, name):
    if x is None:
        return [None] * n
    out = list(x) if isinstance(x, (list, tuple)) else [x] * n
    if len(out) != n:
        raise ValueError(f"{name} has {len(out)} entries, expected {n}")
    return out



def add_kv_alignment(model, rungs=None, spec=None):
    """Attach the learnable post-norm alignment that makes KV-group pooling usable.

    One rotation angle per (rung, kv head, rotary pair) for k and for v, per layer. Initialised to
    zero, which is the identity, so attaching this changes nothing until training moves it -- and
    the full-width configuration never touches it at all.

    Why rotations rather than the full orthogonal U_v the offline study fitted: a block-diagonal
    per-pair rotation is already orthogonal, which is the only property the un-rotation before
    o_proj needs, and it costs head_dim/2 parameters per head instead of head_dim^2. At 1.7B that
    is 86k parameters against 3.7M for the same job.

    The angles are trained, not fitted. The offline version solved for them on a data sample with
    the weights frozen, which is the right thing when you are measuring whether the axis can work
    at all; inside the supernet the weights are moving anyway, so letting the optimiser place the
    angles is both simpler and strictly more expressive.

    NOTE on the grid: n_kv can only take divisors of the model's own n_kv, so SmolLM2 (n_kv 3 and
    5, both prime) gets a binary knob -- full, or a single shared KV head, a 3x or 5x jump in one
    step. Qwen3's n_kv 8 gives {1,2,4,8}. The axis is only a real search dimension on Qwen3.
    """
    from torch import nn
    from ..config import SPECS
    attns = sorted((m for m in model.modules() if isinstance(m, all_attn_classes())),
                   key=lambda m: m.layer_idx)
    if spec is None:
        hd = attns[0].head_dim
        spec = next((v for v in SPECS.values() if v.head_dim == hd), None)
    spec_qk_grid = list(spec.qk_grid) if spec is not None else [attns[0].head_dim]
    for m in attns:
        n_kv_full = m.config.num_key_value_heads
        r = rungs if rungs is not None else [d for d in range(1, n_kv_full) if n_kv_full % d == 0]
        r = [x for x in r if x < n_kv_full]
        m._kv_rungs = list(r)
        if not r:
            continue
        half = m.head_dim // 2
        dev = next(m.parameters()).device
        dt = next(m.parameters()).dtype
        # k is indexed [rung, kv_head, pair]; v needs an extra axis for the d_v rung, because v
        # is a contiguous prefix so its rotation PLANE moves with d_v while k's does not (the
        # rotary gather keeps k's pairs fixed). Sharing one v angle set across d_v was measured to
        # turn a -0.253 nat gain at full width into a +0.039 nat regression at d_v=64.
        vr = sorted(spec_qk_grid)
        m._v_rungs = list(vr)
        m.kv_ang_k = nn.Parameter(torch.zeros(len(r), n_kv_full, half, device=dev, dtype=dt))
        m.kv_ang_v = nn.Parameter(torch.zeros(len(r), len(vr), n_kv_full, half,
                                              device=dev, dtype=dt))
    return model


def set_elastic_config(model, qk, v, pair_order, n_h=None, d_mlp=None, n_kv=None):
    """Select one sub-network. Each argument is an int (all layers) or a per-layer list.

    qk, v    per-head query/key and value dims   (nested RoPE-pair-ordered / prefix slices)
    n_h      query heads, balanced across KV groups; None = keep all
    d_mlp    MLP width, nested prefix;            None = keep all
    n_kv     KV groups, mean-pooled after a learned post-norm rotation; None = keep all.
             Must divide the model's own n_kv AND the active n_h. Unlike the other four knobs
             this one is not a pure selection -- pooling mixes heads, so it needs the alignment
             in attention.kv_rotate to stay invariance-preserving.

    Nothing is copied or overwritten: every knob selects a subset of the live weights, so one
    weight set serves every configuration and gradients flow back to the full parameters.
    """
    attns = sorted((m for m in model.modules() if isinstance(m, all_attn_classes())),
                   key=lambda m: m.layer_idx)
    mlps = [L.mlp for L in model.model.layers]
    n = len(attns)
    qk_list = _per_layer(qk, n, "qk")
    v_list = _per_layer(v, n, "v")
    h_list = _per_layer(n_h, n, "n_h")
    m_list = _per_layer(d_mlp, n, "d_mlp")

    # A pair_order built for another family would silently gather the wrong columns (the second
    # half of each RoPE pair sits at +head_dim/2), so check it rather than trust the caller.
    hd = attns[0].head_dim
    if 2 * len(pair_order) != hd:
        raise ValueError(
            f"pair_order has {len(pair_order)} pairs but head_dim is {hd}; "
            f"build it with build_pair_order(mode, {hd // 2}) or pair_order_for(model).")

    # n_kv has to be resolved BEFORE n_h, because head selection groups query heads by KV group
    # and the grouping is against the ACTIVE n_kv, not the base one. Doing it the other way round
    # rejects every head count a smaller n_kv unlocks -- n_h=4 under n_kv=2 is legal, but checked
    # against the base n_kv=8 it raises. That killed a 1.7B run 26 minutes in.
    kv_list = _per_layer(n_kv, len(attns), "n_kv")
    kv_active = []
    for m, ka in zip(attns, kv_list):
        nf = m.config.num_key_value_heads
        ka = nf if ka is None else int(ka)
        if nf % ka:
            raise ValueError(f"n_kv={ka} must divide the model's n_kv={nf}")
        if ka != nf and not hasattr(m, "kv_ang_k"):
            raise RuntimeError(
                "n_kv below full width needs the alignment parameters; call "
                "add_kv_alignment(model) after enable_elastic(). Pooling without them costs "
                "~6.4 nats at 8->4 and the axis is unusable.")
        kv_active.append(ka)

    idx_cache, head_cache = {}, {}
    for m, qka, va, ha, ka in zip(attns, qk_list, v_list, h_list, kv_active):
        if qka not in idx_cache:
            idx_cache[qka] = qk_gather_index(pair_order, qka)
        m.qk_active = int(qka)
        m.v_active = int(va)
        m._qk_idx = idx_cache[qka]

        n_q, n_kv_full = m.config.num_attention_heads, m.config.num_key_value_heads
        ha = n_q if ha is None else int(ha)
        if ha % ka:
            raise ValueError(f"n_kv={ka} must divide n_h={ha}")
        if ha == n_q:
            m._q_idx = None
        else:
            # Grouped by the ACTIVE n_kv: keep ha/ka heads from each of the ka surviving groups.
            if (ha, ka) not in head_cache:
                head_cache[(ha, ka)] = head_index(n_q, ka, ha)
            m._q_idx = head_cache[(ha, ka)]
            if ABLATION["heads"] == "prefix":
                if ka != n_kv_full:
                    raise ValueError("the head-prefix ablation keeps every KV group")
                m._q_idx = torch.arange(ha, dtype=torch.long)
        # repeat_kv is recomputed from the tensors after pooling, so this is only the un-merged
        # bookkeeping value; see the note at the repeat in elastic_attention_forward.
        m.num_key_value_groups = max(1, ha // n_kv_full)
        m.n_h_active = ha
        m.kv_active = ka

    for mlp, ma in zip(mlps, m_list):
        mlp.mlp_active = None if ma is None else int(ma)
    return model
