"""Transformer layer as a directed acyclic graph of operations.

Builds the full op sequence for one transformer layer including GEMMs,
fused VRC ops (softmax, RMSNorm), and defines fusion edges.
"""

import dataclasses
from typing import List, Tuple

from .op_model import OpType


@dataclasses.dataclass
class LayerOp:
    """One operation in the layer DAG."""
    name: str
    op_type: OpType
    in_channels: int
    out_channels: int
    seq_length: int
    scale_factor: float = 1.0
    n_heads: int = 1
    n_kv_groups: int = 1
    # VRC-specific
    n_elements: int = 0
    vrc_type: str = ""


class LayerGraph:
    """Builds the operation graph for one transformer layer.

    Infinite attention (decode mode):
      QK_gen → V_gen → QK_attn → Softmax → PV_attn → ATTN_proj →
      RMSNorm_attn → MLP_FC1 → MLP_FC2 → RMSNorm_mlp

    Identity attention (MLP-only):
      MLP_FC1 → MLP_FC2 → RMSNorm
    """

    def __init__(self, layer_spec: dict, n_embd: int, context_length: int,
                 mode: str = "decode"):
        self.layer_spec = layer_spec
        self.n_embd = n_embd
        self.context_length = context_length
        self.mode = mode
        self.ops: List[LayerOp] = []
        self._build()

    def _build(self):
        ls = self.layer_spec
        nh = ls['n_head']
        nkv = ls['n_kv_group']
        qk = ls['n_qk_head_dim']
        vd = ls['n_v_head_dim']
        mlp = ls['mlp_size']
        attn = ls.get('attention_variant', 'infinite')

        proj_seq = 1 if self.mode == 'decode' else self.context_length
        # DXE spatial constraint: D=128 for output dimension.
        # Context must be Timeloop-mappable: round up to nearest multiple of 128
        # for large values, or nearest power of 2 for small values.
        ctx = max(self.context_length, 32)
        if ctx <= 128:
            # Round up to nearest power of 2 (32, 64, 128)
            p = 32
            while p < ctx:
                p *= 2
            ctx = p
        else:
            # Round up to nearest multiple of 128
            ctx = ((ctx + 127) // 128) * 128

        if attn == 'infinite':
            # Projection GEMMs
            self.ops.append(LayerOp(
                name='QK_gen', op_type=OpType.GEMV_DECODE,
                in_channels=self.n_embd,
                out_channels=qk * (nh + nkv),
                seq_length=proj_seq,
                n_heads=nh, n_kv_groups=nkv))

            self.ops.append(LayerOp(
                name='V_gen', op_type=OpType.GEMV_DECODE,
                in_channels=self.n_embd,
                out_channels=vd * nkv,
                seq_length=proj_seq,
                n_heads=nh, n_kv_groups=nkv))

            # Attention GEMMs (with KV cache correction)
            self.ops.append(LayerOp(
                name='QK_attn', op_type=OpType.QK_ATTN,
                in_channels=qk, out_channels=ctx,
                seq_length=nh // nkv,
                scale_factor=nkv,
                n_heads=nh, n_kv_groups=nkv))

            # Fused softmax (VRC)
            softmax_elements = ctx * nh  # per-head softmax over context
            self.ops.append(LayerOp(
                name='softmax', op_type=OpType.VRC_SOFTMAX,
                in_channels=0, out_channels=0, seq_length=0,
                n_elements=softmax_elements, vrc_type='softmax'))

            self.ops.append(LayerOp(
                name='PV_attn', op_type=OpType.PV_ATTN,
                in_channels=ctx, out_channels=vd,
                seq_length=nh // nkv,
                scale_factor=nkv,
                n_heads=nh, n_kv_groups=nkv))

            self.ops.append(LayerOp(
                name='ATTN_proj', op_type=OpType.GEMV_DECODE,
                in_channels=vd * nh,
                out_channels=self.n_embd,
                seq_length=proj_seq))

            # Post-attention RMSNorm (VRC)
            self.ops.append(LayerOp(
                name='rmsnorm_attn', op_type=OpType.VRC_RMSNORM,
                in_channels=0, out_channels=0, seq_length=0,
                n_elements=self.n_embd * proj_seq, vrc_type='rmsnorm'))

            # MLP
            self.ops.append(LayerOp(
                name='MLP_FC1', op_type=OpType.GEMV_DECODE,
                in_channels=self.n_embd, out_channels=mlp,
                seq_length=proj_seq))

            self.ops.append(LayerOp(
                name='MLP_FC2', op_type=OpType.GEMV_DECODE,
                in_channels=mlp, out_channels=self.n_embd,
                seq_length=proj_seq))

            # Post-MLP RMSNorm (VRC)
            self.ops.append(LayerOp(
                name='rmsnorm_mlp', op_type=OpType.VRC_RMSNORM,
                in_channels=0, out_channels=0, seq_length=0,
                n_elements=self.n_embd * proj_seq, vrc_type='rmsnorm'))

        else:
            # Identity or causal: MLP only
            self.ops.append(LayerOp(
                name='MLP_FC1', op_type=OpType.GEMV_DECODE,
                in_channels=self.n_embd, out_channels=mlp,
                seq_length=proj_seq))

            self.ops.append(LayerOp(
                name='MLP_FC2', op_type=OpType.GEMV_DECODE,
                in_channels=mlp, out_channels=self.n_embd,
                seq_length=proj_seq))

            self.ops.append(LayerOp(
                name='rmsnorm', op_type=OpType.VRC_RMSNORM,
                in_channels=0, out_channels=0, seq_length=0,
                n_elements=self.n_embd * proj_seq, vrc_type='rmsnorm'))

    def get_gemm_ops(self) -> List[LayerOp]:
        """Return only GEMM/GEMV ops (for Timeloop evaluation)."""
        return [op for op in self.ops
                if op.op_type not in (OpType.VRC_SOFTMAX, OpType.VRC_RMSNORM)]

    def get_vrc_ops(self) -> List[LayerOp]:
        """Return only VRC ops."""
        return [op for op in self.ops
                if op.op_type in (OpType.VRC_SOFTMAX, OpType.VRC_RMSNORM)]

    def get_fusion_edges(self) -> List[Tuple[int, int]]:
        """Return (producer_idx, consumer_idx) pairs for GEMM fusion savings.

        Only between adjacent GEMM ops (VRC ops are fused in hardware,
        not relevant to DRAM fusion savings).
        """
        gemm_ops = self.get_gemm_ops()
        gemm_indices = [i for i, op in enumerate(self.ops)
                        if op.op_type not in (OpType.VRC_SOFTMAX,
                                              OpType.VRC_RMSNORM)]

        attn = self.layer_spec.get('attention_variant', 'infinite')
        if attn == 'infinite':
            # GEMM-only indices (skipping VRC):
            # 0:QK_gen, 1:V_gen, 2:QK_attn, 3:PV_attn, 4:ATTN_proj,
            # 5:MLP_FC1, 6:MLP_FC2
            return [(0, 2), (1, 3), (2, 3), (3, 4), (4, 5), (5, 6)]
        else:
            # 0:MLP_FC1, 1:MLP_FC2
            return [(0, 1)]

    def weight_bytes(self, n_embd: int) -> int:
        """Total weight bytes for this layer (INT8 = 1 byte/param)."""
        ls = self.layer_spec
        nh, nkv = ls['n_head'], ls['n_kv_group']
        qk, vd, mlp = ls['n_qk_head_dim'], ls['n_v_head_dim'], ls['mlp_size']
        attn = ls.get('attention_variant', 'infinite')

        if attn == 'infinite':
            return (n_embd * qk * (nh + nkv) +      # QK_gen
                    n_embd * vd * nkv +               # V_gen
                    vd * nh * n_embd +                 # ATTN_proj
                    n_embd * mlp +                     # MLP_FC1
                    mlp * n_embd)                      # MLP_FC2
        else:
            return n_embd * mlp + mlp * n_embd
