"""Which transformer family a model belongs to, and the two things that differ between them.

The elastic forward was written against Qwen3 and hard-codes `mq.Qwen3Attention` in four places.
Porting it to Llama turns out to be a net simplification rather than a second implementation:
the two families agree on the attention forward signature, on every projection name, on
`head_dim` / `num_key_value_groups` / `scaling` / `layer_idx`, and on `apply_rotary_pos_emb` and
`repeat_kv`. Exactly one thing differs -- Qwen3 applies a learnable per-dim RMSNorm to q and k
between the projection and RoPE, and Llama does not.

So a family is a class pair plus a boolean, and the forward branches once.

Why this matters beyond tidiness: Qwen3's smallest published checkpoint cannot reach the 50-150M
range at all. Its vocabulary is 151,936, so at hidden 1024 the embeddings alone are 156M -- above
the top of that range, and slicing cannot touch them. SmolLM2-135M carries a 49,152 vocabulary and
28M of embeddings, which puts its whole span (52-134M) inside the target. The port is what makes
that scale reachable.

Note the two families also imply different grids. SmolLM2-135M is 9 query heads over 3 KV groups,
so `n_h` must be a multiple of 3 rather than of 8, and its head_dim is 64 rather than 128.
`ModelSpec` already carries those per model; nothing here needs to know about them.
"""
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable


@dataclass(frozen=True)
class Family:
    name: str
    attn_cls: type
    mlp_cls: type
    apply_rope: Callable
    repeat_kv: Callable
    qk_norm: bool          # is there a learnable RMSNorm on q/k between projection and RoPE?


def _qwen3() -> Family:
    import transformers.models.qwen3.modeling_qwen3 as m
    return Family("qwen3", m.Qwen3Attention, m.Qwen3MLP,
                  m.apply_rotary_pos_emb, m.repeat_kv, qk_norm=True)


def _llama() -> Family:
    import transformers.models.llama.modeling_llama as m
    return Family("llama", m.LlamaAttention, m.LlamaMLP,
                  m.apply_rotary_pos_emb, m.repeat_kv, qk_norm=False)


_BUILDERS = {"qwen3": _qwen3, "llama": _llama}
_CACHE: dict[str, Family] = {}


def family(name_or_model) -> Family:
    """Resolve a family from `config.model_type`, a model, or the family name itself.

    Built lazily and cached, because importing a modeling module has a real cost and a run only
    ever touches one family.
    """
    key = getattr(getattr(name_or_model, "config", None), "model_type", None) \
        or getattr(name_or_model, "model_type", None) \
        or name_or_model
    key = str(key)
    if key not in _BUILDERS:
        raise ValueError(
            f"unsupported family {key!r}; add a builder in elastic/families.py. The two things a "
            f"new family needs are its attention/MLP classes and whether it applies a norm to q/k "
            f"before RoPE -- everything else in the elastic forward is shared.")
    if key not in _CACHE:
        _CACHE[key] = _BUILDERS[key]()
    return _CACHE[key]


@lru_cache(maxsize=1)
def all_attn_classes() -> tuple:
    """Every attention class this build can patch, for isinstance checks that must span families.

    Cached because this sits inside per-module loops (`set_elastic_config` walks every module) and
    an uncached version rebuilt BOTH families on every call: 2.07 ms per set_elastic_config on a
    397-module model against 0.60 ms cached, times 8 calls per training step. Before the port this
    was a module-level class reference costing nothing.

    Note this does import both modeling modules, so the laziness claim above holds only for code
    paths that never take an isinstance check spanning families.
    """
    return tuple(b().attn_cls for b in _BUILDERS.values())
