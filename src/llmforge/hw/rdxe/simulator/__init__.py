"""Simulation engines: single-chip (Layer 2) and multi-chip ring (Layer 3)."""

from .chip_sim import ChipSimulator, InferenceResult, TokenTrace
from .ring_sim import RingSimulator, RingResult, RingEvent
