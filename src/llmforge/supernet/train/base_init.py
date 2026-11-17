"""Base-initialised students with the supernet's parameterisation.

A student extracted from a trained supernet carries two extra parameter sets: the KV-group
alignment angles, without which n_kv below full width cannot run at all, and the per-width logit
temperature. A student started from the published checkpoint has neither. Attaching both at their
zero initialisation, which is the identity, gives the base-initialised student the same
parameterisation as the supernet-initialised one, so the two arms differ only in where the
weights start.
"""
from __future__ import annotations


def attach_supernet_params(model):
    """Add zero-initialised KV alignment and logit temperature to an elastic model.

    Call after enable_elastic(). Both additions are the identity until training moves them, so
    the full-width model is unchanged.
    """
    from ..elastic import add_elastic_temperature, add_kv_alignment

    add_kv_alignment(model)
    add_elastic_temperature(model)
    return model


def needs_kv_alignment(n_kv, full_n_kv: int) -> bool:
    """True when any layer pools KV groups below the model's own count."""
    if n_kv is None:
        return False
    vals = n_kv if isinstance(n_kv, (list, tuple)) else [n_kv]
    return any(v is not None and int(v) != int(full_n_kv) for v in vals)
