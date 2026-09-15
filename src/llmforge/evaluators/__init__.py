"""Software and hardware evaluators for the co-search dispatcher.

    base          evaluator protocols and the standard hardware metric names
    cache         shared evaluation caches and the GPU lock
    sw_supernet   held-out loss of a supernet slice
    hw_analytic   parameter, FLOP and KV-cache estimates, always merged into every record
    hw_zeus       measured latency and energy on a local NVIDIA GPU through ZEUS
    hw_device     Pixel Watch 5 latency and energy from a predictor fitted to device measurements
    hw_timeloop   Timeloop accelerator substrates
    hw_rdxe       rDXE ring accelerator with an inner chip co-search
"""
from .base import STANDARD_HW_KEYS, HwEvaluator, SwEvaluator, merge_hw_dicts

__all__ = ["STANDARD_HW_KEYS", "HwEvaluator", "SwEvaluator", "merge_hw_dicts"]
