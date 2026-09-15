"""Regression tests for geometry and the cost model.

Every case here corresponds to a defect a pre-code review found in the M2/M4 code. They are
cheap, CPU-only, and run without a GPU or a checkpoint.
"""
import pytest

from llmforge.supernet import SPECS, ElasticConfig, ModelSpec
from llmforge.supernet.elastic import head_index
from llmforge.supernet.elastic.sampler import head_grid


# ---- published geometry --------------------------------------------------------------

@pytest.mark.parametrize("key,params", [
    ("qwen3-0.6b", 596_049_920),
    ("qwen3-1.7b", 1_720_574_976),
    ("qwen3-4b", 4_022_468_096),
])
def test_param_count_matches_published(key, params):
    """Geometry must reproduce the published parameter count to within rounding."""
    assert abs(SPECS[key].n_params() - params) / params < 1e-3


@pytest.mark.parametrize("key", sorted(SPECS))
def test_registry_matches_config_json(key):
    """Guards against a registry typo silently surviving a run (needs network)."""
    pytest.importorskip("transformers")
    ModelSpec.from_pretrained(key)


# ---- the corrupting bug: KV fraction on a non-28-layer model -------------------------

@pytest.mark.parametrize("key", sorted(SPECS))
def test_full_config_is_exactly_one(key):
    """The old kv_cost hardcoded 28 layers: 4B's all-128 config reported 1.2857, not 1.0,
    so a '50% KV' search on 4B was really a ~39% search."""
    assert ElasticConfig.full(SPECS[key]).kv_frac() == pytest.approx(1.0)


@pytest.mark.parametrize("key", sorted(SPECS))
def test_half_dims_is_exactly_half(key):
    s = SPECS[key]
    assert ElasticConfig.uniform(s, s.head_dim // 2, s.head_dim // 2).kv_frac() == pytest.approx(0.5)


# ---- search-space combinatorics ------------------------------------------------------

def test_head_pair_counts():
    """13 valid (n_h, n_kv) pairs at 16Q/8KV, 15 at 32Q/8KV."""
    assert len(SPECS["qwen3-1.7b"].head_pairs()) == 13
    assert len(SPECS["qwen3-4b"].head_pairs()) == 15


def test_gqa_condition_holds_for_every_pair():
    for spec in SPECS.values():
        for n_h, n_kv in spec.head_pairs():
            assert n_kv <= n_h and n_h % n_kv == 0


def test_states_per_layer():
    """1 + |pairs|*|qk|^2 attention states, times 1 + |mlp| MLP states."""
    assert SPECS["qwen3-1.7b"].states_per_layer() == (1 + 13 * 16) * 5 == 1045
    assert SPECS["qwen3-4b"].states_per_layer() == (1 + 15 * 16) * 5 == 1205


def test_group_layout_divides_evenly():
    assert SPECS["qwen3-1.7b"].groups() == 7
    assert SPECS["qwen3-4b"].groups() == 9   # 36 layers, not 28


def test_head_grid_is_a_uniform_8x_range():
    """Resolves the flagged asymmetry: both 16Q and 32Q span exactly 8x.

    Scoped to the specs the halving rule applies to. A power-of-two n_q is what makes an 8x
    span expressible at all; SmolLM2's 9Q and 15Q carry an explicit grid instead and are
    covered by the legality test below.
    """
    for spec in SPECS.values():
        if spec.nh_grid_override is not None:
            continue
        g = spec.nh_grid
        assert g[-1] == spec.n_q and g[-1] // g[0] == 8


def test_head_grid_entries_are_runnable():
    """Every rung must be a head count head_index() can actually build, for every spec.

    Two different spaces, and conflating them makes the test vacuous. `nh_grid` is the COST-MODEL
    space, where n_kv is elastic too, so a rung only has to work for SOME n_kv on the grid --
    qwen3-4b's rung 4 is unbuildable at n_kv=8 and fine at n_kv=1,2,4. `head_grid(spec)` is what
    the sampler can run TODAY, where n_kv is fixed at the checkpoint's value; those must all build
    at that n_kv, and there must be at least two of them or the head knob does not exist.
    """
    for spec in SPECS.values():
        g = spec.nh_grid
        assert g == sorted(set(g)) and g[-1] == spec.n_q, spec.key
        for h in g:
            ok = [k for k in spec.nkv_grid if k <= h and h % k == 0
                  and h // k <= spec.n_q // k]
            assert ok, f"{spec.key}: n_h={h} builds at no n_kv in {spec.nkv_grid}"
            head_index(spec.n_q, ok[0], h)                 # raises if unbuildable

        runnable = head_grid(spec)
        assert len(runnable) >= 2, f"{spec.key}: head knob is degenerate, grid {runnable}"
        for h in runnable:
            head_index(spec.n_q, spec.n_kv, h)             # the sampler's actual call, verbatim


# ---- gates and the cost vector -------------------------------------------------------

def test_gated_attention_contributes_zero_kv():
    s = SPECS["qwen3-4b"]
    c = ElasticConfig.full(s)
    c.attn_on = [False] * s.n_layers
    assert c.kv_bytes_per_token() == 0 and c.attn_params() == 0


def test_kv_floor_requires_n_kv_one():
    """The advertised 3.125% floor needs n_kv=1; at n_kv=2 the floor is 6.25%."""
    s = SPECS["qwen3-1.7b"]
    lo = s.qk_grid[0]
    assert ElasticConfig.uniform(s, lo, lo, n_kv=1).kv_frac() == pytest.approx(0.03125)
    assert ElasticConfig.uniform(s, lo, lo, n_kv=2).kv_frac() == pytest.approx(0.0625)


def test_decode_macs_includes_lm_head():
    """hw_eval.py omitted the tied LM head -- 26% of decode MACs at 0.6B."""
    s = SPECS["qwen3-0.6b"]
    c = ElasticConfig.full(s)
    assert c.decode_macs(4096) - (c.attn_params() + c.mlp_params()
                                  + sum(s.n_q * 4096 * (s.head_dim * 2)
                                        for _ in range(s.n_layers))) == s.vocab * s.hidden


# ---- validation ----------------------------------------------------------------------

def test_validate_rejects_gqa_violation():
    s = SPECS["qwen3-1.7b"]
    c = ElasticConfig.uniform(s, 64, 64)
    c.n_kv = [4] * s.n_layers
    c.n_h = [2] * s.n_layers          # 4 does not divide 2
    with pytest.raises(ValueError, match="GQA"):
        c.validate()


def test_roundtrip_serialization():
    s = SPECS["qwen3-4b"]
    c = ElasticConfig.uniform(s, 64, 96, n_kv=2, n_h=8)
    assert ElasticConfig.from_dict(c.to_dict()).to_dict() == c.to_dict()
