"""rDXE: graph-level simulator of the DXE decoder accelerator and its multi-chip ring.

Wraps Timeloop per-GEMM results with corrections for KV cache routing, GQA via VLINK, fused VRC
(Softmax and RMSNorm), WMEM layer-switch reload, and multi-chip ring pipeline simulation.

Subpackages and modules:
  core/        building blocks: constants, config, OpModel, KVCacheState, LayerGraph,
               ScaledChipSpec, the cached Timeloop evaluator, and the analytical GEMM model
  simulator/   ChipSimulator (Layer 2), RingSimulator (Layer 3), Timeloop-backed layer costs
  reporting/   CSV writers and matplotlib plots
  workflow.py  profile, pack, and ring-simulate a model: the three-stage rDXE workflow
  cosearch.py  inner chip co-search run for every candidate architecture of the outer search
  scaling.py   analytical edge-LLM scaling study on chip-scaled rings
  cli.py       eval, ring, and compare commands

Convenience re-exports for common names:
  from llmforge.hw.rdxe import OpModel, ChipSimulator, RingSimulator, DXEConfig
"""

# Re-export commonly used names so downstream code can do:
#   from llmforge.hw.rdxe import OpModel, RingSimulator, ...
from .core import (
    DXEConfig,
    OpModel, OpResult, OpType,
    KVCacheState,
    LayerGraph, LayerOp,
    ScaledChipSpec, DXE_REFERENCE, EDGE_MODELS,
    scale_chip_for_layer, scale_ring_for_model,
)
from .simulator import (
    ChipSimulator, InferenceResult, TokenTrace,
    RingSimulator, RingResult, RingEvent,
)
from .reporting import (
    save_inference_csv, save_summary_csv,
    plot_inference, plot_ring,
)
