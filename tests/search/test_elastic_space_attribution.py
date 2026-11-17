"""Search-space attribution ablations: sub-grids (fixed n_kv, Flextron-style) and tied head dimensions.

Both restrictions are switchable and off by default, so the unrestricted space is unchanged.
"""
import random
from pathlib import Path

import pytest
import yaml

from llmforge.search.elastic_space import ElasticSearchSpace

CONFIGS = Path(__file__).resolve().parents[2] / "configs" / "search_spaces"


def _evolve(s, n=200, seed=1):
    rng = random.Random(seed)
    pop = [s.sample() for _ in range(40)]
    for _ in range(n):
        a, b = rng.sample(pop, 2)
        c, d = s.crossover(a, b)
        pop += [s.mutate(c), s.mutate(d), s.repair(c)]
    return pop


def _values(pop, knob):
    return {li[knob] for ind in pop for li in ind["layers"]}


def test_defaults_leave_the_space_unchanged():
    s = ElasticSearchSpace("smollm2-135m", with_nkv=True)
    assert s.grid_subset == {} and s.tie_dims is False and s.tied == {}
    d = s.to_yaml_dict()
    assert "grid_subset" not in d and "tie_dims" not in d
    assert s.log10_size() == pytest.approx(12.92, abs=0.01)


@pytest.mark.parametrize("model, base_kv", [("smollm2-135m", 3), ("smollm2-360m", 5)])
def test_fixed_nkv_never_moves_n_kv(model, base_kv):
    s = ElasticSearchSpace(model, with_nkv=True, grid_subset={"n_kv_group": [base_kv]}, seed=0)
    pop = _evolve(s)
    assert _values(pop, "n_kv_group") == {base_kv}
    assert len(_values(pop, "n_head")) > 1 and len(_values(pop, "n_qk_head_dim")) > 1
    assert all(li["n_head"] % li["n_kv_group"] == 0 for ind in pop for li in ind["layers"])


def test_tied_dims_stay_equal_and_still_move():
    s = ElasticSearchSpace("smollm2-135m", with_nkv=True, tie_dims=True, seed=0)
    pop = _evolve(s)
    assert all(li["n_qk_head_dim"] == li["n_v_head_dim"] for ind in pop for li in ind["layers"])
    assert len(_values(pop, "n_qk_head_dim")) == 4
    # One shared knob instead of two: 5 blocks * log10(4) fewer digits.
    full = ElasticSearchSpace("smollm2-135m", with_nkv=True)
    assert full.log10_size() - s.log10_size() == pytest.approx(5 * 0.60206, abs=1e-3)
    uni = ElasticSearchSpace("smollm2-135m", partition="uniform", with_nkv=True, tie_dims=True)
    assert all(li["n_qk_head_dim"] == li["n_v_head_dim"] for ind in uni.enumerate_uniform() for li in ind["layers"])
    with pytest.raises(ValueError):
        s.uniform(9, 64, 32, 1536, 3)


def test_flextron_space_frees_only_heads_and_mlp():
    s = ElasticSearchSpace("smollm2-135m", with_nkv=True, gqa_dims=True, grid_subset={"n_kv_group": [3]}, seed=0)
    pop = _evolve(s)
    assert _values(pop, "n_kv_group") == {3}
    assert _values(pop, "n_qk_head_dim") == {64} and _values(pop, "n_v_head_dim") == {64}
    assert len(_values(pop, "n_head")) > 1 and len(_values(pop, "mlp_size")) > 1


def test_restricted_full_anchor_matches_the_unrestricted_space():
    full = ElasticSearchSpace("smollm2-135m", with_nkv=True)
    for kw in ({"tie_dims": True}, {"grid_subset": {"n_kv_group": [3]}},
               {"gqa_dims": True, "grid_subset": {"n_kv_group": [3]}}):
        s = ElasticSearchSpace("smollm2-135m", with_nkv=True, **kw)
        assert s.key(s.full()) == full.key(full.full())


def test_invalid_restrictions_are_rejected():
    with pytest.raises(ValueError):
        ElasticSearchSpace("smollm2-135m", with_nkv=True, grid_subset={"n_kv_group": [2]})
    with pytest.raises(ValueError):
        ElasticSearchSpace("smollm2-135m", with_nkv=False, grid_subset={"n_kv_group": [3]})
    with pytest.raises(ValueError):
        ElasticSearchSpace("smollm2-135m", with_nkv=True, grid_subset={"n_head": [4]})
    with pytest.raises(ValueError):
        ElasticSearchSpace("smollm2-135m", with_nkv=True, gqa_dims=True, tie_dims=False,
                           grid_subset={"n_v_head_dim": [32]})


def test_yaml_round_trip(tmp_path):
    s = ElasticSearchSpace("smollm2-360m", with_nkv=True, tie_dims=True, grid_subset={"n_kv_group": [5]})
    path = tmp_path / "space.yaml"
    path.write_text(yaml.safe_dump(s.to_yaml_dict(), sort_keys=False))
    t = ElasticSearchSpace.from_yaml(str(path))
    assert t.tie_dims and t.grid_subset == {"n_kv_group": [5]} and t.grids == s.grids


@pytest.mark.parametrize("model", ["smollm2-135m", "smollm2-360m"])
@pytest.mark.parametrize("variant", ["tied", "fixkv", "flextron", "gqa"])
def test_shipped_attribution_configs_load(model, variant):
    s = ElasticSearchSpace.from_yaml(str(CONFIGS / f"{model}_nkv_{variant}.yaml"))
    assert s.with_nkv
    s.to_elastic_config(s.full())
    s.to_elastic_config(s.smallest())


# ---- kv_cap: n_kv = min(n_h, cap) per block (Qwen3-0.6B fixkv and Flextron spaces) --------------------------

def test_kv_cap_derives_n_kv_from_n_head():
    s = ElasticSearchSpace("qwen3-0.6b", with_nkv=True, kv_cap=8, seed=0)
    assert s.grids["n_kv_group"] == [2, 4, 8]
    assert s.head_pairs() == [(2, 2), (4, 4), (8, 8), (16, 8)]
    pop = _evolve(s)
    assert all(li["n_kv_group"] == min(li["n_head"], 8) for ind in pop for li in ind["layers"])
    assert _values(pop, "n_head") == {2, 4, 8, 16} and len(_values(pop, "n_qk_head_dim")) > 1
    assert s.smallest()["layers"][0]["n_kv_group"] == 2
    # Uniform grid: one n_kv per n_h.
    uni = ElasticSearchSpace("qwen3-0.6b", partition="uniform", with_nkv=True, kv_cap=8)
    assert all(li["n_kv_group"] == min(li["n_head"], 8) for ind in uni.enumerate_uniform() for li in ind["layers"])
    assert len(uni.enumerate_uniform()) == 4 * 4 * 4 * 4
    with pytest.raises(ValueError):
        s.uniform(16, 128, 128, 3072, 4)


def test_kv_cap_repairs_an_unrestricted_individual():
    full = ElasticSearchSpace("qwen3-0.6b", with_nkv=True, seed=0)
    s = ElasticSearchSpace("qwen3-0.6b", with_nkv=True, kv_cap=8, seed=0)
    ind = full.uniform(16, 64, 64, 1536, 1)
    assert all(li["n_kv_group"] == 8 for li in s.repair(ind)["layers"])


def test_kv_cap_flextron_frees_only_heads_and_mlp():
    s = ElasticSearchSpace("qwen3-0.6b", with_nkv=True, gqa_dims=True, kv_cap=8, seed=0)
    pop = _evolve(s)
    assert _values(pop, "n_qk_head_dim") == {128} and _values(pop, "n_v_head_dim") == {128}
    assert all(li["n_kv_group"] == min(li["n_head"], 8) for ind in pop for li in ind["layers"])
    assert len(_values(pop, "n_head")) > 1 and len(_values(pop, "mlp_size")) > 1


def test_kv_cap_off_by_default_and_validated(tmp_path):
    assert ElasticSearchSpace("qwen3-0.6b", with_nkv=True).kv_cap is None
    assert "kv_cap" not in ElasticSearchSpace("qwen3-0.6b", with_nkv=True).to_yaml_dict()
    with pytest.raises(ValueError):
        ElasticSearchSpace("qwen3-0.6b", with_nkv=False, kv_cap=8)
    with pytest.raises(ValueError):
        ElasticSearchSpace("smollm2-135m", with_nkv=True, kv_cap=2)  # min(3, 2) = 2 does not divide 3
    s = ElasticSearchSpace("qwen3-0.6b", with_nkv=True, kv_cap=8)
    path = tmp_path / "space.yaml"
    path.write_text(yaml.safe_dump(s.to_yaml_dict(), sort_keys=False))
    t = ElasticSearchSpace.from_yaml(str(path))
    assert t.kv_cap == 8 and t.grids == s.grids


@pytest.mark.parametrize("variant", ["tied", "fixkv", "flextron", "gqa"])
def test_shipped_qwen_attribution_configs_load(variant):
    s = ElasticSearchSpace.from_yaml(str(CONFIGS / f"qwen3-0.6b_nkv_{variant}.yaml"))
    full = ElasticSearchSpace.from_yaml(str(CONFIGS / "qwen3-0.6b_nkv.yaml"))
    assert s.with_nkv and s.key(s.full()) == full.key(full.full())
    s.to_elastic_config(s.full())
    s.to_elastic_config(s.smallest())
