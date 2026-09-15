"""One architecture as a dict, with analytic cost estimates.

Schema
    globals
        n_embd            model width
        block_size        context length used by an estimate when no seq_len is passed
        layer_mask        which layer slots are active
        vocab_size        vocabulary size (default 50257)
        tie_embeddings    input and output embeddings share one matrix (default True)
        mlp_variant       "swiglu" with 3 matrices (default) or "mlp" with 2
        use_concat_heads  heads are concatenated before the output projection (default False)
        base_model        supernet key, present on individuals from an elastic search space
    layers, one dict per slot
        n_head, n_kv_group, n_qk_head_dim, n_v_head_dim, mlp_size, n_cproj,
        attention_variant ("infinite", "causal", "mha" or "identity")

Layers whose layer_mask entry is False are inactive and cost nothing. Estimates count weight
matrices and embeddings and ignore norm gains and biases.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional

DEFAULT_VOCAB = 50257


def _mlp_matrix_count(variant: str = "swiglu") -> int:
    """SwiGLU has gate, up and down projections. A plain MLP has up and down."""
    return 3 if variant == "swiglu" else 2


class Individual(dict):
    """Runtime Individual that behaves like a dict and carries cost helpers."""

    def __init__(self, globals: Dict[str, Any] = None, layers: List[Dict[str, Any]] = None):
        super().__init__()
        self["globals"] = globals or {}
        self["layers"] = layers or []

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Individual":
        return Individual(dict(d.get("globals", {})), [dict(li) for li in d.get("layers", [])])

    # ---- structure -----------------------------------------------------------------------

    def active_layers(self) -> List[Dict[str, Any]]:
        layers = self["layers"]
        mask = self["globals"].get("layer_mask", [True] * len(layers))
        return [li for i, li in enumerate(layers) if i < len(mask) and mask[i]]

    def d_model(self) -> int:
        g = self["globals"]
        return int(g.get("n_embd", g.get("d_model", 768)))

    def vocab_size(self) -> int:
        return int(self["globals"].get("vocab_size", DEFAULT_VOCAB))

    def _dims(self, li: Dict[str, Any]) -> Dict[str, Any]:
        g = self["globals"]
        d = self.d_model()
        h = int(li.get("n_head", 8))
        return {
            "d": d,
            "h": h,
            "m": int(li.get("mlp_size", 4 * d)),
            "qk": int(li.get("n_qk_head_dim", d // max(1, h))),
            "v": int(li.get("n_v_head_dim", d // max(1, h))),
            "n_cproj": int(li.get("n_cproj", 1)),
            "kv": int(li.get("n_kv_group", h)),
            "concat": bool(li.get("use_concat_heads", g.get("use_concat_heads", False))),
            "attn": li.get("attention_variant", g.get("attention_variant", "mha")),
            "mlp_variant": li.get("mlp_variant", g.get("mlp_variant", "swiglu")),
        }

    def _seq(self, seq_len: Optional[int]) -> int:
        return int(seq_len if seq_len is not None else self["globals"].get("block_size", 512))

    # ---- parameters ----------------------------------------------------------------------

    def embedding_params(self) -> int:
        tied = bool(self["globals"].get("tie_embeddings", True))
        return self.vocab_size() * self.d_model() * (1 if tied else 2)

    def layer_params(self, li: Dict[str, Any]) -> int:
        x = self._dims(li)
        d, h, qk, v, kv = x["d"], x["h"], x["qk"], x["v"], x["kv"]
        if x["attn"] == "infinite":
            cproj = (h * v) * d if x["concat"] else x["n_cproj"] * (v * d)
            attn = d * (h * qk) + d * (kv * qk) + d * (kv * v) + cproj
        elif x["attn"] in ("causal", "mha"):
            out = (h * v) * d if x["concat"] else x["n_cproj"] * (v * d)
            attn = d * (h * (qk + qk + v)) + out
        elif x["attn"] == "identity":
            attn = 0
        else:
            raise ValueError(f"Unknown attention_variant: {x['attn']}")
        return int(attn + _mlp_matrix_count(x["mlp_variant"]) * d * x["m"])

    def estimate_params(self) -> int:
        """Weight matrices of active layers plus embeddings, counted once when tied."""
        return int(self.embedding_params() + sum(self.layer_params(li) for li in self.active_layers()))

    # ---- compute and memory --------------------------------------------------------------

    def _layer_flops(self, li: Dict[str, Any], seq: int) -> float:
        x = self._dims(li)
        d, h, qk, v, kv, m = x["d"], x["h"], x["qk"], x["v"], x["kv"], x["m"]
        if x["attn"] == "infinite":
            proj = 2.0 * seq * d * (h * qk) + 2.0 * seq * d * (kv * qk) + 2.0 * seq * d * (kv * v)
            core = 2.0 * h * seq * (seq / kv) * qk + 2.0 * h * seq * (seq / kv)
            outp = 2.0 * seq * (h * v) * d if x["concat"] else x["n_cproj"] * (2.0 * seq * v * d)
            attn = proj + core + outp
        elif x["attn"] in ("causal", "mha"):
            proj = 2.0 * seq * d * (h * (qk + qk + v))
            core = 2.0 * h * seq * seq * qk + 2.0 * h * seq * seq
            outp = 2.0 * seq * (h * v) * d if x["concat"] else x["n_cproj"] * (2.0 * seq * v * d)
            attn = proj + core + outp
        elif x["attn"] == "identity":
            attn = 0.0
        else:
            raise ValueError(f"Unknown attention_variant: {x['attn']}")
        return attn + 2.0 * _mlp_matrix_count(x["mlp_variant"]) * seq * d * m

    def estimate_flops(self, seq_len: Optional[int] = None) -> float:
        """Forward and backward-style 2x matmul FLOPs over `seq_len` tokens, embeddings excluded."""
        seq = self._seq(seq_len)
        return float(sum(self._layer_flops(li, seq) for li in self.active_layers()))

    def estimate_mem_access(self, seq_len: Optional[int] = None) -> float:
        """Weight and KV-cache access proxy, same accounting as estimate_flops."""
        return self.estimate_flops(seq_len)

    def estimate_kv_cache_size(self, seq_len: Optional[int] = None) -> int:
        """KV-cache scalars for `seq_len` cached tokens. Multiply by bytes per scalar for bytes."""
        seq = self._seq(seq_len)
        d = self.d_model()
        total = 0
        for li in self.active_layers():
            variant = li.get("attention_variant", self["globals"].get("attention_variant", "mha"))
            if variant == "infinite":
                total += seq * int(li["n_kv_group"]) * (int(li["n_qk_head_dim"]) + int(li["n_v_head_dim"]))
            elif variant == "causal":
                n_head = int(li.get("n_head", 2 ** li.get("n_head_exp", 1)))
                n_kv = int(li.get("n_kv_group", 2 ** li.get("n_kv_group_exp", 1)))
                total += 2 * seq * n_kv * (d // n_head)
        return int(total)

    def decode_macs(self, ctx: int) -> int:
        """Multiply-accumulates per generated token at context `ctx`, LM head included."""
        total = self.vocab_size() * self.d_model()
        for li in self.active_layers():
            total += self.layer_params(li)
            x = self._dims(li)
            if x["attn"] in ("infinite", "causal", "mha"):
                total += x["h"] * int(ctx) * (x["qk"] + x["v"])
        return int(total)

    # ---- identity and display ------------------------------------------------------------

    def arch_key(self) -> str:
        """Stable hash of the active architecture, independent of layer_mask padding."""
        g = self["globals"]
        record = {
            "globals": {k: g.get(k) for k in ("base_model", "n_embd", "vocab_size", "tie_embeddings",
                                              "mlp_variant", "use_concat_heads")},
            "layers": [{k: li[k] for k in sorted(li)} for li in self.active_layers()],
        }
        return hashlib.sha1(json.dumps(record, sort_keys=True).encode()).hexdigest()[:16]

    def print_individual(self, include_inactive: bool = False, include_params: bool = True,
                         max_layers: int = None) -> None:
        g = self.get("globals", {})
        layers = self.get("layers", [])
        mask = g.get("layer_mask", [True] * len(layers))
        active = sum(1 for i in range(min(len(mask), len(layers))) if mask[i])
        print(f"Globals: {g}")
        print(f"Total layers: {len(layers)}; Active layers: {active}")
        if include_params:
            print(f"Estimated params: {self.estimate_params() / 1e6:.2f}M")
        for i, layer in enumerate(layers):
            if max_layers is not None and i >= max_layers:
                print(f"... (only showing first {max_layers} layers)")
                break
            on = mask[i] if i < len(mask) else True
            if not on and not include_inactive:
                continue
            kv = ", ".join(f"{k}={v}" for k, v in layer.items() if k != "params")
            print(f"  - Layer {i}: {kv}{'' if on else ' [inactive]'}")
