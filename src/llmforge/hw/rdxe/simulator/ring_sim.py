"""Layer 3: Multi-chip ring pipeline simulator.

Models N DXE chips connected in a token-level pipeline ring:
  chip_0 -> chip_1 -> ... -> chip_{N-1} -> chip_0

Each chip holds a subset of layers. Tokens flow through the ring.
Supports prefill (pipeline ramp-up) and decode (steady-state) phases.
"""

import dataclasses
from typing import Dict, List, Optional, Tuple

from ..core.constants import (
    CLOCK_PERIOD_NS,
    INTER_CHIP_ENERGY_PJ_PER_BIT,
    INTER_CHIP_BW_BYTES_PER_CYCLE,
    INTER_CHIP_LATENCY_CYCLES,
    WMEM_TOTAL_B,
)
from ..core.config import DXEConfig
from ..core.op_model import OpModel
from ..core.layer_graph import LayerGraph
from .chip_sim import ChipSimulator, InferenceResult, TokenTrace


@dataclasses.dataclass
class RingEvent:
    """An event in the pipeline simulation."""
    cycle: float
    chip_id: int
    event_type: str  # "compute_start", "compute_end", "transfer"
    token_idx: int
    energy_pJ: float = 0.0
    detail: str = ""


@dataclasses.dataclass
class RingResult:
    """Multi-chip ring simulation result."""
    n_chips: int
    layers_per_chip: List[List[int]]
    total_tokens: int
    prefill_length: int
    decode_length: int

    # Pipeline metrics
    pipeline_depth: int = 0
    pipeline_bubble_cycles: float = 0.0
    steady_state_throughput_tps: float = 0.0
    total_latency_ms: float = 0.0

    # Serving metrics
    ttft_ms: float = 0.0               # Time to first token (prefill latency)
    tpot_us: float = 0.0               # Time per output token (decode steady-state)
    prefill_latency_ms: float = 0.0
    decode_latency_ms: float = 0.0
    stage_cycles: float = 0.0          # cycles per pipeline stage

    # Per-chip
    per_chip_cycles: List[float] = dataclasses.field(default_factory=list)
    per_chip_energy_uJ: List[float] = dataclasses.field(default_factory=list)
    per_chip_utilization: List[float] = dataclasses.field(default_factory=list)

    # Energy totals
    total_energy_uJ: float = 0.0
    compute_energy_uJ: float = 0.0
    inter_chip_comm_energy_uJ: float = 0.0
    kv_cache_energy_uJ: float = 0.0
    vrc_energy_uJ: float = 0.0

    energy_per_token_uJ: float = 0.0
    tokens_per_second: float = 0.0

    events: List[RingEvent] = dataclasses.field(default_factory=list)


class RingSimulator:
    """Models N DXE chips in a token-level pipeline ring."""

    def __init__(self, n_chips: int, op_model: OpModel,
                 config: DXEConfig = None):
        self.n_chips = n_chips
        self.op_model = op_model
        self.config = config or DXEConfig()

    def assign_layers(self, layers: list, n_embd: int,
                      strategy: str = "round_robin") -> List[List[int]]:
        """Assign layer indices to chips.

        Returns: list of layer index lists, one per chip.
        """
        n = len(layers)

        if strategy == "single_layer":
            # One layer per chip. N chips must >= N layers.
            assignment = [[i] for i in range(min(n, self.n_chips))]
            # Pad remaining chips with empty
            while len(assignment) < self.n_chips:
                assignment.append([])
            return assignment

        elif strategy == "balanced":
            # Balance by weight bytes
            weights = [LayerGraph(l, n_embd, 1).weight_bytes(n_embd)
                       for l in layers]
            assignment = [[] for _ in range(self.n_chips)]
            chip_loads = [0] * self.n_chips
            # Greedy: assign heaviest layer to lightest chip
            for li in sorted(range(n), key=lambda i: -weights[i]):
                lightest = min(range(self.n_chips), key=lambda c: chip_loads[c])
                assignment[lightest].append(li)
                chip_loads[lightest] += weights[li]
            # Sort each chip's layers in order
            for a in assignment:
                a.sort()
            return assignment

        else:  # round_robin
            assignment = [[] for _ in range(self.n_chips)]
            for i in range(n):
                assignment[i % self.n_chips].append(i)
            return assignment

    def run_ring_inference(self, model_spec: dict,
                           prefill_length: int = None,
                           decode_length: int = None,
                           model_name: str = "") -> RingResult:
        """Simulate full inference on the ring.

        Prefill: batch GEMM — all tokens processed per chip sequentially through ring.
          Each chip processes the full batch through its assigned layers, then passes
          the output to the next chip. No token-level pipelining during prefill.

        Decode: token-level pipeline — each token flows through all chips.
          Steady-state: one token exits per stage_cycles (bottleneck chip + transfer).
        """
        if prefill_length is None:
            prefill_length = self.config.prefill_length
        if decode_length is None:
            decode_length = self.config.decode_length

        n_embd = model_spec['n_embd']
        layers = model_spec['layers']
        total_tokens = prefill_length + decode_length

        # Assign layers to chips
        assignment = self.assign_layers(
            layers, n_embd, self.config.layer_assignment)

        # Create per-chip model specs
        chip_specs = []
        for chip_layers in assignment:
            chip_specs.append({
                'n_embd': n_embd,
                'layers': [layers[i] for i in chip_layers],
            })

        # ── Evaluate per-chip costs for both modes ──

        chip_decode_cycles = []    # per-token decode cost
        chip_decode_energy = []
        chip_prefill_cycles = []   # full-batch prefill cost
        chip_prefill_energy = []

        for ci, spec in enumerate(chip_specs):
            sim = ChipSimulator(self.op_model, self.config)
            if not spec['layers']:
                chip_decode_cycles.append(0)
                chip_decode_energy.append(0)
                chip_prefill_cycles.append(0)
                chip_prefill_energy.append(0)
                continue

            # Decode: single token through this chip's layers
            r_dec = sim.run_inference(spec, prefill_length=0, decode_length=1)
            if r_dec.token_traces:
                chip_decode_cycles.append(r_dec.token_traces[0].total_cycles)
                chip_decode_energy.append(r_dec.token_traces[0].total_energy_pJ)
            else:
                chip_decode_cycles.append(0)
                chip_decode_energy.append(0)

            # Prefill: full batch through this chip's layers
            if prefill_length > 0:
                sim2 = ChipSimulator(self.op_model, self.config)
                r_pf = sim2.run_inference(spec, prefill_length=prefill_length, decode_length=0)
                chip_prefill_cycles.append(r_pf.total_cycles)
                chip_prefill_energy.append(r_pf.total_energy_uJ * 1e6)  # convert uJ → pJ
            else:
                chip_prefill_cycles.append(0)
                chip_prefill_energy.append(0)

        # Inter-chip transfer cost (one embedding vector per hop)
        transfer_energy_pJ, transfer_cycles = self._inter_chip_transfer_cost(n_embd)

        active_chips = sum(1 for c in chip_decode_cycles if c > 0)
        pipeline_depth = active_chips

        # ── Prefill: sequential batch through ring ──
        # Each chip processes all prefill tokens as a batch GEMM, then passes to next chip.
        # Prefill energy scales with batch size (already captured by ChipSimulator).
        # For batch transfer: all tokens' embeddings pass between chips.
        prefill_compute_cycles = sum(chip_prefill_cycles)
        prefill_transfer_cycles = (active_chips - 1) * transfer_cycles * prefill_length
        prefill_total_cycles = prefill_compute_cycles + prefill_transfer_cycles

        prefill_total_energy = sum(chip_prefill_energy)
        prefill_comm_energy = (active_chips - 1) * transfer_energy_pJ * prefill_length

        # ── Decode: token-level pipeline ──
        max_decode_chip = max(chip_decode_cycles) if chip_decode_cycles else 0
        stage_cycles = max_decode_chip + transfer_cycles

        # First decode token must fill the pipeline
        decode_ramp_cycles = (pipeline_depth - 1) * stage_cycles if pipeline_depth > 1 else 0
        decode_steady_cycles = decode_length * stage_cycles
        decode_total_cycles = decode_ramp_cycles + decode_steady_cycles

        decode_energy_per_token = sum(chip_decode_energy)
        decode_comm_per_token = (active_chips - 1) * transfer_energy_pJ
        decode_total_energy = (decode_energy_per_token + decode_comm_per_token) * decode_length

        # ── Totals ──
        total_cycles = prefill_total_cycles + decode_total_cycles
        total_latency_ms = total_cycles * CLOCK_PERIOD_NS / 1e6

        total_energy_pJ = prefill_total_energy + prefill_comm_energy + decode_total_energy
        total_energy_uJ = total_energy_pJ / 1e6
        comm_total_uJ = (prefill_comm_energy + decode_comm_per_token * decode_length) / 1e6

        # TTFT: prefill completes + first decode token exits pipeline
        ttft_cycles = prefill_total_cycles + stage_cycles if prefill_length > 0 else stage_cycles * pipeline_depth
        ttft_ms = ttft_cycles * CLOCK_PERIOD_NS / 1e6

        # TPOT: decode steady-state (one token per stage)
        tpot_us = stage_cycles * CLOCK_PERIOD_NS / 1e3

        # Steady-state throughput
        ss_tps = 1.0 / (stage_cycles * CLOCK_PERIOD_NS * 1e-9) if stage_cycles > 0 else 0

        # Per-chip utilization
        per_chip_util = []
        for ci, dec_cyc in enumerate(chip_decode_cycles):
            if dec_cyc > 0 and stage_cycles > 0:
                per_chip_util.append(min(1.0, dec_cyc / stage_cycles))
            else:
                per_chip_util.append(0.0)

        # Energy per token (averaged over full sequence)
        energy_per_token_uJ = total_energy_uJ / total_tokens if total_tokens > 0 else 0

        prefill_latency_ms = prefill_total_cycles * CLOCK_PERIOD_NS / 1e6
        decode_latency_ms = decode_total_cycles * CLOCK_PERIOD_NS / 1e6
        bubble_cycles = decode_ramp_cycles

        result = RingResult(
            n_chips=self.n_chips,
            layers_per_chip=assignment,
            total_tokens=total_tokens,
            prefill_length=prefill_length,
            decode_length=decode_length,
            pipeline_depth=pipeline_depth,
            pipeline_bubble_cycles=bubble_cycles,
            steady_state_throughput_tps=ss_tps,
            total_latency_ms=total_latency_ms,
            ttft_ms=ttft_ms,
            tpot_us=tpot_us,
            prefill_latency_ms=prefill_latency_ms,
            decode_latency_ms=decode_latency_ms,
            stage_cycles=stage_cycles,
            per_chip_cycles=chip_decode_cycles,
            per_chip_energy_uJ=[e / 1e6 for e in chip_decode_energy],
            per_chip_utilization=per_chip_util,
            total_energy_uJ=total_energy_uJ,
            inter_chip_comm_energy_uJ=comm_total_uJ,
            energy_per_token_uJ=energy_per_token_uJ,
            tokens_per_second=total_tokens / (total_latency_ms / 1e3) if total_latency_ms > 0 else 0,
        )

        return result

    def _inter_chip_transfer_cost(self, n_embd: int) -> Tuple[float, float]:
        """Energy (pJ) and cycles for transferring one embedding vector.

        n_embd bytes (INT8) over the inter-chip link.
        """
        n_bits = n_embd * 8
        energy_pJ = n_bits * INTER_CHIP_ENERGY_PJ_PER_BIT
        transfer_bytes = n_embd
        data_cycles = transfer_bytes / INTER_CHIP_BW_BYTES_PER_CYCLE
        cycles = INTER_CHIP_LATENCY_CYCLES + data_cycles
        return energy_pJ, cycles
