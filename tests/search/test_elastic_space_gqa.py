"""The grouped-query ablation of the elastic search space: pinned head dimensions, anchors, operators.

The ablation asks what the Infinite-Head parameterization buys. A conventional grouped-query model gives
query, key and value one shared head dimension and does not search it, so the ablation pins both dimension
grids to the base head dimension and leaves head count, key--value groups and MLP width free.
"""
import random

import pytest
import yaml

from llmforge.search.elastic_space import ElasticSearchSpace


@pytest.mark.parametrize("model, head_dim", [("smollm2-135m", 64), ("smollm2-360m", 64),
                                             ("qwen3-0.6b", 128), ("qwen3-1.7b", 128)])
def test_dimension_grids_are_pinned_to_the_base_head_dimension(model, head_dim):
    s = ElasticSearchSpace(model, gqa_dims=True)
    assert s.grids["n_qk_head_dim"] == [head_dim] and s.grids["n_v_head_dim"] == [head_dim]
    # The knob tuple is untouched, so nothing downstream has to know the space is restricted.
    assert s.knobs == ElasticSearchSpace(model).knobs


@pytest.mark.parametrize("with_nkv", [False, True])
def test_full_width_anchor_is_shared_with_the_unrestricted_space(with_nkv):
    """The hypervolume box comes from full() and smallest(), so a shared full() keeps one corner common."""
    iha = ElasticSearchSpace("smollm2-135m", with_nkv=with_nkv)
    gqa = ElasticSearchSpace("smollm2-135m", with_nkv=with_nkv, gqa_dims=True)
    assert iha.key(iha.full()) == gqa.key(gqa.full())
    # The other corner differs, which is why hypervolume has to be recomputed on a common box before the
    # two spaces are compared.
    assert iha.key(iha.smallest()) != gqa.key(gqa.smallest())


def test_the_space_shrinks_to_the_grouped_query_knobs():
    iha = ElasticSearchSpace("smollm2-135m", with_nkv=True)
    gqa = ElasticSearchSpace("smollm2-135m", with_nkv=True, gqa_dims=True)
    assert gqa.log10_size() < iha.log10_size()
    assert gqa.log10_size() == pytest.approx(6.90, abs=0.01)
    assert len(ElasticSearchSpace("smollm2-135m", partition="uniform", with_nkv=True,
                                  gqa_dims=True).enumerate_uniform()) == 24


def test_operators_never_move_the_pinned_dimensions():
    s = ElasticSearchSpace("smollm2-135m", with_nkv=True, gqa_dims=True, seed=0)
    rng = random.Random(1)
    pop = [s.sample() for _ in range(40)]
    for _ in range(200):
        a, b = rng.sample(pop, 2)
        c, d = s.crossover(a, b)
        pop += [s.mutate(c), s.mutate(d)]
    assert {li["n_qk_head_dim"] for ind in pop for li in ind["layers"]} == {64}
    assert {li["n_v_head_dim"] for ind in pop for li in ind["layers"]} == {64}
    # The free knobs still move, so mutation is not silently stuck on the pinned ones.
    assert len({li["n_head"] for ind in pop for li in ind["layers"]}) > 1
    assert len({li["mlp_size"] for ind in pop for li in ind["layers"]}) > 1
    assert all(li["n_head"] % li["n_kv_group"] == 0 for ind in pop for li in ind["layers"])


def test_yaml_round_trip_carries_the_flag(tmp_path):
    s = ElasticSearchSpace("smollm2-135m", with_nkv=True, gqa_dims=True)
    d = s.to_yaml_dict()
    assert d["gqa_dims"] is True and d["knobs"]["n_qk_head_dim"] == [64]
    path = tmp_path / "space.yaml"
    path.write_text(yaml.safe_dump(d, sort_keys=False))
    t = ElasticSearchSpace.from_yaml(str(path))
    assert t.gqa_dims and t.grids == s.grids


def test_unrestricted_space_is_unaffected():
    s = ElasticSearchSpace("smollm2-135m", with_nkv=True)
    assert s.gqa_dims is False
    assert s.grids["n_qk_head_dim"] == [16, 32, 48, 64]
    assert "gqa_dims" not in s.to_yaml_dict()
