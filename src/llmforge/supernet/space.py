"""The elastic configuration object and its cost model.

This is the type that flows through training, evaluation, search and export. The M2/M4 code
passed a bare `(qk_list, v_list)` 2-tuple, which cannot express the five knobs M5 adds; a
fresh reviewer correctly called the absence of this object the single blocking gap in the
design doc.

Seven per-layer knobs:
    d_qk, d_v    per-head query/key and value dims   (nested prefixes of head_dim)
    n_kv         KV groups                           (nested mean-pooling)
    n_h          query heads                         (group-balanced selection)
    d_mlp        MLP intermediate width              (nested prefix)
    attn_on      attention sublayer active           (0 => zero KV for that layer)
    mlp_on       MLP sublayer active
`attn_on=0 and mlp_on=0` is the whole-block mask.

Cost model corrections over search/hw_eval.py + eval/kv_cost.py:
  * the full-config denominator comes from the spec's own geometry, not a hardcoded 28 layers
    (the old code reported a 36-layer all-128 config as 1.2857 of full);
  * n_kv is per-layer, not a scalar;
  * gates zero their sublayer's contribution;
  * decode_macs includes the tied LM head, which is config-invariant but 26% of decode MACs
    at 0.6B -- omitting it inflates every relative efficiency claim.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import ModelSpec

BYTES_KV = 2  # fp16/bf16 KV cache


@dataclass
class ElasticConfig:
    spec: ModelSpec
    d_qk: list[int]
    d_v: list[int]
    n_kv: list[int]
    n_h: list[int]
    d_mlp: list[int]
    attn_on: list[bool] = field(default=None)
    mlp_on: list[bool] = field(default=None)

    def __post_init__(self):
        L = self.spec.n_layers
        if self.attn_on is None:
            self.attn_on = [True] * L
        if self.mlp_on is None:
            self.mlp_on = [True] * L
        for name in ("d_qk", "d_v", "n_kv", "n_h", "d_mlp", "attn_on", "mlp_on"):
            v = getattr(self, name)
            if len(v) != L:
                raise ValueError(f"{name} has {len(v)} entries, expected {L}")

    # ---- constructors --------------------------------------------------------

    @classmethod
    def full(cls, spec: ModelSpec) -> "ElasticConfig":
        L = spec.n_layers
        return cls(spec, [spec.head_dim] * L, [spec.head_dim] * L, [spec.n_kv] * L,
                   [spec.n_q] * L, [spec.d_mlp] * L)

    @classmethod
    def uniform(cls, spec: ModelSpec, d_qk: int, d_v: int, n_kv: int = None,
                n_h: int = None, d_mlp: int = None) -> "ElasticConfig":
        L = spec.n_layers
        return cls(spec, [d_qk] * L, [d_v] * L, [n_kv or spec.n_kv] * L,
                   [n_h or spec.n_q] * L, [d_mlp or spec.d_mlp] * L)

    @classmethod
    def from_groups(cls, spec: ModelSpec, groups: dict[str, list], per: int = None) -> "ElasticConfig":
        """Expand a grouped genome (one value per group of `per` layers) to per-layer lists."""
        per = per or spec.n_layers // len(groups["d_qk"])
        exp = {k: [x for x in v for _ in range(per)] for k, v in groups.items()}
        return cls(spec, **exp)

    # ---- validation ----------------------------------------------------------

    def validate(self) -> "ElasticConfig":
        s = self.spec
        for i in range(s.n_layers):
            if not self.attn_on[i]:
                continue
            if self.d_qk[i] not in s.qk_grid or self.d_v[i] not in s.qk_grid:
                raise ValueError(f"layer {i}: d_qk/d_v off grid {s.qk_grid}")
            if self.n_kv[i] not in s.nkv_grid:
                raise ValueError(f"layer {i}: n_kv {self.n_kv[i]} not in {s.nkv_grid}")
            if self.n_h[i] not in s.nh_grid:
                raise ValueError(f"layer {i}: n_h {self.n_h[i]} not in {s.nh_grid}")
            if self.n_h[i] % self.n_kv[i]:
                raise ValueError(f"layer {i}: GQA condition violated, "
                                 f"n_kv={self.n_kv[i]} does not divide n_h={self.n_h[i]}")
        for i in range(s.n_layers):
            if self.mlp_on[i] and self.d_mlp[i] not in s.mlp_grid:
                raise ValueError(f"layer {i}: d_mlp {self.d_mlp[i]} not in {s.mlp_grid}")
        return self

    # ---- cost primitives -----------------------------------------------------

    def kv_bytes_per_token(self, bytes_per: int = BYTES_KV) -> int:
        return sum(self.n_kv[i] * (self.d_qk[i] + self.d_v[i]) * bytes_per
                   for i in range(self.spec.n_layers) if self.attn_on[i])

    def kv_frac(self) -> float:
        """Fraction of the FULL config's KV cache, denominator from this spec's geometry."""
        return self.kv_bytes_per_token() / self.spec.kv_bytes_per_token()

    def attn_params(self) -> int:
        e = self.spec.hidden
        return sum(e * (self.n_h[i] * self.d_qk[i] + self.n_kv[i] * self.d_qk[i]
                        + self.n_kv[i] * self.d_v[i] + self.n_h[i] * self.d_v[i])
                   for i in range(self.spec.n_layers) if self.attn_on[i])

    def mlp_params(self) -> int:
        return sum(3 * self.spec.hidden * self.d_mlp[i]
                   for i in range(self.spec.n_layers) if self.mlp_on[i])

    def embed_params(self) -> int:
        s = self.spec
        return s.vocab * s.hidden * (1 if s.tied else 2)

    def weight_params(self, include_embed: bool = True) -> int:
        return self.attn_params() + self.mlp_params() + (self.embed_params() if include_embed else 0)

    def decode_macs(self, ctx: int) -> int:
        """MACs per generated token at context length `ctx`, LM head included.

        Projections are GEMVs so their MAC count equals their parameter count; the attention
        term is QK_attn + PV_attn against the cache and does not overlap with it.
        """
        attn_ctx = sum(self.n_h[i] * ctx * (self.d_qk[i] + self.d_v[i])
                       for i in range(self.spec.n_layers) if self.attn_on[i])
        lm_head = self.spec.vocab * self.spec.hidden
        return self.attn_params() + self.mlp_params() + attn_ctx + lm_head

    def cost(self, ctx: int = 4096) -> dict:
        """The handoff record consumed by an external hardware model."""
        return {
            "kv_bytes_per_token": self.kv_bytes_per_token(),
            "kv_frac": round(self.kv_frac(), 6),
            "attn_params": self.attn_params(),
            "mlp_params": self.mlp_params(),
            "embed_params": self.embed_params(),
            "weight_params": self.weight_params(),
            "decode_macs": self.decode_macs(ctx),
            "ctx": ctx,
            "gated_attn": int(sum(not a for a in self.attn_on)),
            "gated_mlp": int(sum(not m for m in self.mlp_on)),
        }

    # ---- serialization -------------------------------------------------------

    def to_dict(self) -> dict:
        return {"model": self.spec.key,
                **{k: list(getattr(self, k)) for k in
                   ("d_qk", "d_v", "n_kv", "n_h", "d_mlp", "attn_on", "mlp_on")}}

    @classmethod
    def from_dict(cls, d: dict) -> "ElasticConfig":
        from .config import SPECS

        spec = SPECS[d["model"]]
        return cls(spec, **{k: d[k] for k in
                            ("d_qk", "d_v", "n_kv", "n_h", "d_mlp", "attn_on", "mlp_on")})

    def __repr__(self) -> str:
        return (f"ElasticConfig({self.spec.key}, kv={self.kv_frac():.1%}, "
                f"W={self.weight_params()/1e9:.2f}B, "
                f"gated={sum(not a for a in self.attn_on)}a/{sum(not m for m in self.mlp_on)}m)")
