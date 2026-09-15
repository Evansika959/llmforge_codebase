"""Single source of truth for model geometry and the elastic search space.

Why this exists: the M2/M4 code hardcoded 28 layers, 16 query heads, and a module-level
`FULLKV = kv_bytes_per_token(128, 128)` in nine files. Qwen3-4B has 36 layers and 32 query
heads, so those constants silently produce wrong budgets rather than errors -- a "50% KV"
search on 4B was really a ~39% search. Everything geometric now derives from a ModelSpec.

Geometry below is verified against each model's published config.json; `ModelSpec.from_pretrained`
re-reads it at load time and asserts agreement, so a registry typo cannot survive a run.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    """Geometry of a base checkpoint plus the elastic grids derived from it."""

    key: str
    repo: str
    hidden: int
    n_layers: int
    n_q: int
    n_kv: int
    head_dim: int
    d_mlp: int
    vocab: int
    tied: bool = True
    # Explicit query-head grid, for models the halving rule below cannot describe. See nh_grid.
    nh_grid_override: tuple[int, ...] | None = None
    # HF model_type, which is also the elastic family key (see elastic/families.py). It decides
    # whether the model has QK-Norm, which changes both the parameter count and where the elastic
    # temperature folds on extraction. from_pretrained() checks it against config.json.
    family: str = "qwen3"

    # ---- elastic grids -------------------------------------------------------
    # All grids are SHRINK-ONLY: the base checkpoint's value is always the maximum.

    @property
    def qk_grid(self) -> list[int]:
        """Per-head qk/v dims: quarter-steps of head_dim. Must stay even (RoPE pairs)."""
        q = self.head_dim // 4
        assert q % 2 == 0, f"head_dim {self.head_dim} gives odd quarter-step {q}"
        return [q * i for i in (1, 2, 3, 4)]

    @property
    def nkv_grid(self) -> list[int]:
        """KV groups reachable by mean-pooling: ALL divisors of n_kv.

        The rule used to be power-of-two divisors only, which halts on the first odd value. That is
        harmless on Qwen3, whose n_kv is 8 and whose divisors are all powers of two anyway, but it
        deleted the axis entirely on SmolLM2: n_kv 3 and 5 are prime, so the halving rule returned
        {3} and {5} -- a single value, i.e. no knob. Pooling needs only that the target divide n_kv
        (the forward reshapes to [.., n_kv_active, group, head_dim] and means over `group`), so
        every divisor is reachable and the restriction bought nothing.

        The consequence for SmolLM2 is a binary knob with a large step -- 3 groups or 1, a 3x cut
        in one move -- rather than the graded {1,2,4,8} Qwen3 gets. That is still worth having: the
        width knobs alone cannot take KV cache below 25% at any scale (d_qk=d_v at their minimum is
        a quarter of head_dim), and n_kv=1 takes SmolLM2 to 8.3%.
        """
        return sorted(d for d in range(1, self.n_kv + 1) if self.n_kv % d == 0)

    @property
    def nh_grid(self) -> list[int]:
        """Query heads: powers of two spanning an 8x shrink range below n_q.

        Stating the rule as a range (rather than 'powers of two') resolves the asymmetry a
        reviewer flagged: 16Q gives {2,4,8,16} and 32Q gives {4,8,16,32}, both 8x, rather
        than 32Q silently gaining or losing a rung.

        The rule needs n_q to be a power of two, which the Qwen3 family is and the SmolLM2
        family is not: 9Q halves to 4.5, so the rule would leave {9} and delete the head knob
        entirely at the scale the port exists to reach. Those models carry an explicit grid
        instead -- multiples of n_kv, which is what head_index() actually requires. The
        override is deliberately not a new general rule, because a general rule would also
        change the Qwen3 grids that every result recorded so far is defined against.
        """
        if self.nh_grid_override is not None:
            return list(self.nh_grid_override)
        out, h = [], self.n_q
        while h >= max(1, self.n_q // 8):
            out.append(h)
            if h % 2:
                break
            h //= 2
        return sorted(out)

    @property
    def mlp_grid(self) -> list[int]:
        """MLP width: nested prefixes at quarter-steps of the full intermediate size."""
        q = self.d_mlp // 4
        return [q * i for i in (1, 2, 3, 4)]

    def head_pairs(self) -> list[tuple[int, int]]:
        """Valid (n_h, n_kv) under the GQA grouping condition n_kv | n_h and n_kv <= n_h."""
        return [(h, k) for h in self.nh_grid for k in self.nkv_grid if k <= h and h % k == 0]

    # ---- search-space size ---------------------------------------------------

    def states_per_layer(self) -> int:
        """1 (attention gated off) + |pairs| x |qk| x |v|, times 1 (mlp off) + |mlp|."""
        attn = 1 + len(self.head_pairs()) * len(self.qk_grid) ** 2
        return attn * (1 + len(self.mlp_grid))

    def groups(self, layers_per_group: int = 4) -> int:
        assert self.n_layers % layers_per_group == 0, (
            f"{self.n_layers} layers not divisible by {layers_per_group}")
        return self.n_layers // layers_per_group

    # ---- parameter accounting ------------------------------------------------

    def layer_params(self) -> dict[str, int]:
        e, hd = self.hidden, self.head_dim
        return {
            "q_proj": e * self.n_q * hd,
            "k_proj": e * self.n_kv * hd,
            "v_proj": e * self.n_kv * hd,
            "o_proj": self.n_q * hd * e,
            "mlp": 3 * e * self.d_mlp,
        }

    def n_params(self) -> int:
        """Total parameters, embeddings included once when tied."""
        per = sum(self.layer_params().values())
        emb = self.vocab * self.hidden * (1 if self.tied else 2)
        # RMSNorm gains: 2 per layer (input/post-attn) + final norm, plus q_norm + k_norm on
        # families that have QK-Norm. Counting those unconditionally overstated smollm2-135m by
        # 30 x 2 x 64 = 3,840 parameters.
        from .elastic.families import family as _family
        qkn = 2 * self.head_dim if _family(self.family).qk_norm else 0
        norms = self.n_layers * (2 * self.hidden + qkn) + self.hidden
        return per * self.n_layers + emb + norms

    def kv_bytes_per_token(self, bytes_per: int = 2) -> int:
        """Full-config KV cache cost -- the denominator for every budget fraction."""
        return self.n_layers * self.n_kv * (self.head_dim * 2) * bytes_per

    # ---- construction --------------------------------------------------------

    @classmethod
    def from_pretrained(cls, key: str) -> "ModelSpec":
        """Return the registry spec after asserting it matches the published config.json."""
        from transformers import AutoConfig

        spec = SPECS[key]
        c = AutoConfig.from_pretrained(spec.repo)
        head_dim = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
        actual = {
            "hidden": c.hidden_size,
            "n_layers": c.num_hidden_layers,
            "n_q": c.num_attention_heads,
            "n_kv": c.num_key_value_heads,
            "head_dim": head_dim,
            "d_mlp": c.intermediate_size,
            "vocab": c.vocab_size,
            "tied": bool(getattr(c, "tie_word_embeddings", True)),
            "family": c.model_type,
        }
        bad = {k: (v, getattr(spec, k)) for k, v in actual.items() if getattr(spec, k) != v}
        if bad:
            raise ValueError(f"{key}: registry disagrees with {spec.repo}/config.json: "
                             f"{ {k: f'config={a} registry={b}' for k, (a, b) in bad.items()} }")
        return spec


# Verified against each published config.json (2026-08-12).
SPECS: dict[str, ModelSpec] = {
    "qwen3-0.6b": ModelSpec("qwen3-0.6b", "Qwen/Qwen3-0.6B-Base",
                            hidden=1024, n_layers=28, n_q=16, n_kv=8,
                            head_dim=128, d_mlp=3072, vocab=151936),
    "qwen3-1.7b": ModelSpec("qwen3-1.7b", "Qwen/Qwen3-1.7B-Base",
                            hidden=2048, n_layers=28, n_q=16, n_kv=8,
                            head_dim=128, d_mlp=6144, vocab=151936),
    "qwen3-4b": ModelSpec("qwen3-4b", "Qwen/Qwen3-4B-Base",
                          hidden=2560, n_layers=36, n_q=32, n_kv=8,
                          head_dim=128, d_mlp=9728, vocab=151936),
    # Llama-family, for the 50-150M band. Qwen3 cannot reach it: its 151,936-token vocabulary is
    # 156M of embeddings at hidden 1024, above the top of the band and untouched by slicing.
    # SmolLM2-135M's 49,152 vocabulary is 28M, so its whole elastic span (52-134M) lands inside.
    "smollm2-135m": ModelSpec("smollm2-135m", "HuggingFaceTB/SmolLM2-135M",
                              hidden=576, n_layers=30, n_q=9, n_kv=3,
                              head_dim=64, d_mlp=1536, vocab=49152,
                              nh_grid_override=(3, 6, 9), family="llama"),
    "smollm2-360m": ModelSpec("smollm2-360m", "HuggingFaceTB/SmolLM2-360M",
                              hidden=960, n_layers=32, n_q=15, n_kv=5,
                              head_dim=64, d_mlp=2560, vocab=49152,
                              nh_grid_override=(5, 10, 15), family="llama"),
}

DEFAULT = "qwen3-4b"


def get(key: str | None = None) -> ModelSpec:
    """Registry lookup; `key` defaults to $LLMFORGE_SUPERNET_MODEL, then DEFAULT."""
    import os

    return SPECS[key or os.environ.get("LLMFORGE_SUPERNET_MODEL", DEFAULT)]
