"""Core building blocks: hardware constants, config, per-op model, KV cache, layer graph, chip scaling."""

from .constants import *  # noqa: F401,F403
from .config import DXEConfig
from .op_model import OpModel, OpResult, OpType
from .kv_cache_model import KVCacheState
from .layer_graph import LayerGraph, LayerOp
from .scaled_arch import (
    ScaledChipSpec, DXE_REFERENCE, EDGE_MODELS,
    layer_weight_bytes, kv_bytes_per_token_per_layer,
    scale_chip_for_layer, scale_ring_for_model,
)
