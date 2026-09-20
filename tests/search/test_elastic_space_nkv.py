"""The KV-group gene of the elastic search space: grids, the n_kv | n_h rule, keys and space sizes."""
import random

import pytest
import yaml

from llmforge.search.elastic_space import KV_KNOB, ElasticSearchSpace


def test_width_only_space_is_unchanged():
    s = ElasticSearchSpace("smollm2-135m")
    assert s.knobs == ("n_head", "n_qk_head_dim", "n_v_head_dim", "mlp_size")
    assert s.grids["n_head"] == [3, 6, 9] and KV_KNOB not in s.grids
    full = s.full()
    assert all(li["n_kv_group"] == 3 for li in full["layers"])
    assert s.key(full) == "smollm2-135m|n_h=9x30;d_qk=64x30;d_v=64x30;d_mlp=1536x30"
    assert len(ElasticSearchSpace("smollm2-135m", partition="uniform").enumerate_uniform()) == 192
    d = s.to_yaml_dict()
    assert "with_nkv" not in d and d["fixed"]["n_kv_group"] == 3
    with pytest.raises(ValueError):
        s.uniform(9, 64, 64, 1536, n_kv_group=1)


@pytest.mark.parametrize("model, kv_grid, n_uniform, log10", [
    ("smollm2-135m", [1, 3], 384, 12.92), ("smollm2-360m", [1, 5], 384, 12.92),
    ("qwen3-1.7b", [1, 2, 4, 8], 832, 14.60), ("qwen3-4b", [1, 2, 4, 8], 960, 14.31)])
def test_kv_space_grids_and_sizes(model, kv_grid, n_uniform, log10):
    s = ElasticSearchSpace(model, with_nkv=True)
    assert s.grids[KV_KNOB] == kv_grid
    assert s.log10_size() == pytest.approx(log10, abs=0.01)
    assert len(ElasticSearchSpace(model, partition="uniform", with_nkv=True).enumerate_uniform()) == n_uniform


def test_operators_keep_n_kv_dividing_n_h():
    s = ElasticSearchSpace("qwen3-1.7b", with_nkv=True, seed=0)
    rng = random.Random(1)
    pop = [s.sample() for _ in range(64)]
    for _ in range(300):
        a, b = rng.sample(pop, 2)
        c, d = s.crossover(a, b)
        pop += [s.mutate(c), s.mutate(d)]
    assert all(li["n_head"] % li["n_kv_group"] == 0 for ind in pop for li in ind["layers"])
    assert {li["n_kv_group"] for ind in pop for li in ind["layers"]} == {1, 2, 4, 8}
    assert {li["n_head"] for ind in pop for li in ind["layers"]} == {2, 4, 8, 16}


def test_illegal_pairs_are_rejected_and_repaired():
    s = ElasticSearchSpace("qwen3-1.7b", with_nkv=True)
    with pytest.raises(ValueError):
        s.uniform(2, 128, 128, 6144, n_kv_group=8)
    bad = s.full()
    for li in bad["layers"]:
        li["n_head"] = 4
    assert all(li["n_kv_group"] == 4 for li in s.repair(bad)["layers"])


def test_keys_configs_and_yaml_round_trip(tmp_path):
    s = ElasticSearchSpace("smollm2-135m", with_nkv=True)
    ind = s.uniform(9, 64, 64, 1536, n_kv_group=1)
    assert s.key(ind) == "smollm2-135m|n_h=9x30;d_qk=64x30;d_v=64x30;d_mlp=1536x30;n_kv=1x30"
    assert list(s.to_elastic_config(ind).n_kv) == [1] * 30
    assert s.key(s.full()) != s.key(ind)
    path = tmp_path / "space.yaml"
    path.write_text(yaml.safe_dump(s.to_yaml_dict(), sort_keys=False))
    t = ElasticSearchSpace.from_yaml(str(path))
    assert t.with_nkv and t.grids == s.grids and t.key(ind) == s.key(ind)
