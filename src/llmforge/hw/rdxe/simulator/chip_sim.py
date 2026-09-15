"""Layer 2: Single-chip inference simulator.

Models the full autoregressive decode loop on one DXE chip:
  - Prefill phase (batch GEMM)
  - Decode phase (GEMV + growing KV cache)
  - Per-token energy/latency traces with all corrections
  - KV cache state tracking
  - WMEM layer-switch reload when weights exceed 3MB
  - Fusion savings between consecutive GEMMs
"""

import dataclasses
from typing import Dict, List, Optional

from ..core.constants import CLOCK_PERIOD_NS, WMEM_TOTAL_B
from ..core.config import DXEConfig
from ..core.op_model import OpModel, OpResult, OpType
from ..core.kv_cache_model import KVCacheState
from ..core.layer_graph import LayerGraph


@dataclasses.dataclass
class TokenTrace:
    """Per-token energy and latency breakdown."""
    token_idx: int
    phase: str  # "prefill" or "decode"
    context_length: int

    # Per-op results (flat list across all layers)
    op_results: List[OpResult] = dataclasses.field(default_factory=list)

    # Aggregated totals
    total_energy_pJ: float = 0.0
    total_cycles: float = 0.0

    # Breakdown by source
    compute_energy_pJ: float = 0.0      # GEMM MAC + SRAM (from Timeloop)
    kv_cache_energy_pJ: float = 0.0     # KV cache reads + writes
    kv_cache_write_energy_pJ: float = 0.0
    dram_energy_pJ: float = 0.0         # remaining DRAM (non-KV)
    wmem_reload_energy_pJ: float = 0.0
    vrc_energy_pJ: float = 0.0
    fusion_savings_pJ: float = 0.0

    # Corrections applied (for comparison)
    kv_correction_pJ: float = 0.0
    vlink_correction_pJ: float = 0.0

    @property
    def energy_uJ(self) -> float:
        return self.total_energy_pJ / 1e6

    @property
    def latency_ns(self) -> float:
        return self.total_cycles * CLOCK_PERIOD_NS

    @property
    def latency_us(self) -> float:
        return self.latency_ns / 1e3


@dataclasses.dataclass
class InferenceResult:
    """Full inference run result."""
    model_name: str
    n_layers: int
    n_embd: int
    prefill_length: int
    decode_length: int
    total_tokens: int

    token_traces: List[TokenTrace] = dataclasses.field(default_factory=list)

    # Aggregated metrics
    total_energy_uJ: float = 0.0
    total_cycles: float = 0.0
    total_latency_ms: float = 0.0
    energy_per_token_uJ: float = 0.0
    latency_per_token_us: float = 0.0
    tokens_per_second: float = 0.0

    # Breakdown
    compute_energy_uJ: float = 0.0
    kv_cache_energy_uJ: float = 0.0
    dram_energy_uJ: float = 0.0
    wmem_reload_energy_uJ: float = 0.0
    vrc_energy_uJ: float = 0.0
    fusion_savings_uJ: float = 0.0

    # Correction totals
    total_kv_correction_uJ: float = 0.0
    total_vlink_correction_uJ: float = 0.0

    def _aggregate(self):
        """Compute aggregates from token traces."""
        self.total_energy_uJ = sum(t.total_energy_pJ for t in self.token_traces) / 1e6
        self.total_cycles = sum(t.total_cycles for t in self.token_traces)
        self.total_latency_ms = self.total_cycles * CLOCK_PERIOD_NS / 1e6

        self.compute_energy_uJ = sum(t.compute_energy_pJ for t in self.token_traces) / 1e6
        self.kv_cache_energy_uJ = sum(
            t.kv_cache_energy_pJ + t.kv_cache_write_energy_pJ
            for t in self.token_traces) / 1e6
        self.dram_energy_uJ = sum(t.dram_energy_pJ for t in self.token_traces) / 1e6
        self.wmem_reload_energy_uJ = sum(t.wmem_reload_energy_pJ for t in self.token_traces) / 1e6
        self.vrc_energy_uJ = sum(t.vrc_energy_pJ for t in self.token_traces) / 1e6
        self.fusion_savings_uJ = sum(t.fusion_savings_pJ for t in self.token_traces) / 1e6

        self.total_kv_correction_uJ = sum(t.kv_correction_pJ for t in self.token_traces) / 1e6
        self.total_vlink_correction_uJ = sum(t.vlink_correction_pJ for t in self.token_traces) / 1e6

        self.total_tokens = self.prefill_length + self.decode_length
        if self.total_tokens > 0:
            self.energy_per_token_uJ = self.total_energy_uJ / self.total_tokens
        if self.decode_length > 0:
            decode_traces = [t for t in self.token_traces if t.phase == "decode"]
            if decode_traces:
                decode_cycles = sum(t.total_cycles for t in decode_traces)
                self.latency_per_token_us = (
                    decode_cycles / len(decode_traces) * CLOCK_PERIOD_NS / 1e3)
        if self.total_latency_ms > 0:
            self.tokens_per_second = self.total_tokens / (self.total_latency_ms / 1e3)


class ChipSimulator:
    """Single-DXE-chip inference simulator."""

    def __init__(self, op_model: OpModel, config: DXEConfig = None):
        self.op_model = op_model
        self.config = config or DXEConfig()
        self.kv_state = KVCacheState()

    def run_inference(self, model_spec: dict,
                      prefill_length: int = None,
                      decode_length: int = None,
                      model_name: str = "") -> InferenceResult:
        """Run full inference simulation.

        Args:
            model_spec: dict with 'n_embd', 'layers' (list of layer dicts)
            prefill_length: prompt tokens (batch GEMM)
            decode_length: tokens to generate autoregressively
            model_name: label for results
        """
        self.kv_state.reset()

        if prefill_length is None:
            prefill_length = self.config.prefill_length
        if decode_length is None:
            decode_length = self.config.decode_length

        n_embd = model_spec['n_embd']
        layers = model_spec['layers']
        n_layers = len(layers)

        # Check if weights fit in WMEM
        # Use chip-specific WMEM capacity if provided, else reference 3MB
        total_weight_bytes = sum(
            LayerGraph(l, n_embd, 1).weight_bytes(n_embd) for l in layers)
        chip_wmem = getattr(self.config, 'chip_wmem_bytes', None) or WMEM_TOTAL_B
        needs_reload = (self.config.enable_wmem_reload and
                        total_weight_bytes > chip_wmem)

        result = InferenceResult(
            model_name=model_name, n_layers=n_layers, n_embd=n_embd,
            prefill_length=prefill_length, decode_length=decode_length,
            total_tokens=prefill_length + decode_length,
        )

        # Prefill phase
        if prefill_length > 0:
            trace = self._simulate_prefill(layers, n_embd, prefill_length,
                                           needs_reload)
            result.token_traces.append(trace)

        # Decode phase
        for t in range(decode_length):
            trace = self._simulate_decode_token(
                layers, n_embd, token_idx=prefill_length + t,
                needs_reload=needs_reload)
            result.token_traces.append(trace)

            # Update KV cache happens inside _simulate_decode_token

        result._aggregate()
        return result

    def _simulate_prefill(self, layers: list, n_embd: int,
                          prefill_length: int, needs_reload: bool
                          ) -> TokenTrace:
        """Simulate prefill phase — all layers process L tokens as batch."""
        trace = TokenTrace(
            token_idx=-1, phase="prefill", context_length=prefill_length)

        for li, layer in enumerate(layers):
            graph = LayerGraph(layer, n_embd, prefill_length, mode="prefill")
            layer_results = self._evaluate_layer_graph(graph, layer)
            trace.op_results.extend(layer_results)

            # Populate KV cache
            self.kv_state.add_tokens(li, prefill_length,
                                     self.config.user_idx)

            # KV cache write energy
            ls = layer
            if ls.get('attention_variant', 'infinite') == 'infinite':
                write_e = KVCacheState.kv_write_energy_pJ(
                    ls['n_kv_group'], ls['n_qk_head_dim'], ls['n_v_head_dim'],
                    self.config.kv_cache_energy_pJ) * prefill_length
                trace.kv_cache_write_energy_pJ += write_e

            # WMEM reload between layers
            if needs_reload and li < len(layers) - 1:
                wb = graph.weight_bytes(n_embd)
                reload = self.op_model.evaluate_wmem_reload(wb)
                trace.op_results.append(reload)
                trace.wmem_reload_energy_pJ += reload.energy_pJ

        self._aggregate_trace(trace)
        return trace

    def _simulate_decode_token(self, layers: list, n_embd: int,
                                token_idx: int, needs_reload: bool
                                ) -> TokenTrace:
        """Simulate one decode token across all layers."""
        # Context = tokens already in KV cache (from prior tokens)
        # Use layer 0's context as representative
        ctx = self.kv_state.get_context(0, self.config.user_idx)
        if ctx == 0:
            ctx = 1  # minimum context for attention

        trace = TokenTrace(
            token_idx=token_idx, phase="decode", context_length=ctx)

        for li, layer in enumerate(layers):
            layer_ctx = self.kv_state.get_context(li, self.config.user_idx)
            if layer_ctx == 0:
                layer_ctx = 1

            graph = LayerGraph(layer, n_embd, layer_ctx, mode="decode")
            layer_results = self._evaluate_layer_graph(graph, layer)
            trace.op_results.extend(layer_results)

            # Update KV cache (add this token)
            if layer.get('attention_variant', 'infinite') == 'infinite':
                self.kv_state.add_token(li, self.config.user_idx)

                # KV cache write energy for this token
                write_e = KVCacheState.kv_write_energy_pJ(
                    layer['n_kv_group'], layer['n_qk_head_dim'],
                    layer['n_v_head_dim'], self.config.kv_cache_energy_pJ)
                trace.kv_cache_write_energy_pJ += write_e

            # WMEM reload
            if needs_reload and li < len(layers) - 1:
                wb = graph.weight_bytes(n_embd)
                reload = self.op_model.evaluate_wmem_reload(wb)
                trace.op_results.append(reload)
                trace.wmem_reload_energy_pJ += reload.energy_pJ

        self._aggregate_trace(trace)
        return trace

    def _evaluate_layer_graph(self, graph: LayerGraph,
                               layer_spec: dict) -> List[OpResult]:
        """Evaluate all ops in a layer graph."""
        results = []

        for op in graph.ops:
            if op.op_type in (OpType.VRC_SOFTMAX, OpType.VRC_RMSNORM):
                if self.config.enable_vrc:
                    r = self.op_model.evaluate_vrc(
                        op.n_elements, op.vrc_type, op_name=op.name)
                    results.append(r)
            else:
                r = self.op_model.evaluate_gemm(
                    op.in_channels, op.out_channels, op.seq_length,
                    op_type=op.op_type, op_name=op.name,
                    n_heads=op.n_heads, n_kv_groups=op.n_kv_groups,
                    scale_factor=op.scale_factor)
                results.append(r)

        return results

    def _aggregate_trace(self, trace: TokenTrace):
        """Aggregate op results into trace totals."""
        for r in trace.op_results:
            trace.total_energy_pJ += r.energy_pJ
            trace.total_cycles += r.cycles

            if r.op_type == OpType.WMEM_RELOAD:
                # Already accounted for in wmem_reload_energy_pJ
                pass
            elif r.op_type in (OpType.VRC_SOFTMAX, OpType.VRC_RMSNORM):
                trace.vrc_energy_pJ += r.energy_pJ
            elif r.op_type in (OpType.QK_ATTN, OpType.PV_ATTN):
                trace.kv_cache_energy_pJ += max(0, r.energy_pJ)
                trace.kv_correction_pJ += r.kv_cache_correction_pJ
                trace.vlink_correction_pJ += r.vlink_correction_pJ
            else:
                trace.compute_energy_pJ += r.energy_pJ

        # Add KV write energy to total
        trace.total_energy_pJ += trace.kv_cache_write_energy_pJ
        trace.total_energy_pJ += trace.wmem_reload_energy_pJ

    def reset(self):
        self.kv_state.reset()
