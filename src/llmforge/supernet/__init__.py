"""llmforge.supernet -- elastic supernets over published checkpoints, the software evaluator."""
from .config import SPECS, ModelSpec, get
from .space import ElasticConfig

__version__ = "0.1.0"
__all__ = ["ModelSpec", "SPECS", "get", "ElasticConfig"]
