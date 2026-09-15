"""CPU tests for the elastic search space, NSGA-II, Pareto utilities, and the dispatcher.

The dispatcher tests replace the supernet evaluator with a deterministic stand-in, so they need no
GPU and no checkpoint.
"""
import json
import random

import pytest

from llmforge.paths import CONFIGS
from llmforge.search.elastic_space import ElasticSearchSpace
from llmforge.search.individual import Individual
from llmforge.search.nsga2 import EvaluationResult, Population, cons_value, dominates
from llmforge.search.pareto import hypervolume, non_dominated, normalized_hypervolume

MODELS = ["smollm2-135m", "smollm2-360m", "qwen3-0.6b", "qwen3-1.7b", "qwen3-4b"]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("partition", ["uniform", "blocks", "per_layer"])
def test_operators_stay_in_space(model, partition):
    s = ElasticSearchSpace(model, partition=partition, seed=0)
    for _ in range(40):
        a, b = s.sample(), s.sample()
        c, _ = s.crossover(a, b)
        m = s.mutate(c)
        assert s.key(m) != s.key(c)
        s.to_elastic_config(m)
        assert s.key(s.repair(m)) == s.key(m)
        assert s.individual(s.genome_of(m)) == m


@pytest.mark.parametrize("model", MODELS)
def test_params_match_supernet_cost_model(model):
    s = ElasticSearchSpace(model, partition="blocks", seed=1)
    for ind in (s.full(), s.smallest(), s.sample(), s.sample()):
        assert Individual.from_dict(ind).estimate_params() == s.to_elastic_config(ind).weight_params()


def test_shipped_configs_validate():
    paths = sorted((CONFIGS / "search_spaces").glob("*.yaml"))
    assert len(paths) == 2 * len(MODELS)
    for p in paths:
        ElasticSearchSpace.from_yaml(str(p))


def test_key_is_partition_independent_and_grid_size():
    u = ElasticSearchSpace("smollm2-135m", "uniform")
    b = ElasticSearchSpace("smollm2-135m", "blocks")
    assert u.key(u.full()) == b.key(b.full())
    assert len(u.enumerate_uniform()) == 3 * 4 * 4 * 4


def test_hypervolume_exact_values():
    assert hypervolume([[0, 1], [1, 0]], [2, 2]) == pytest.approx(3.0)
    assert hypervolume([[0, 0, 0]], [1, 1, 1]) == pytest.approx(1.0)
    assert hypervolume([[0, 1, 0], [1, 0, 0]], [2, 2, 1]) == pytest.approx(3.0)
    assert normalized_hypervolume([[0.0, 0.0]], [0, 0], [1, 1], margin=0.0) == pytest.approx(1.0)
    assert non_dominated([[1, 2], [2, 1], [2, 2]]) == [0, 1]


def test_constrained_domination():
    assert dominates([5, 5], [0.0], [1, 1], [0.5]) == 1
    assert dominates([1, 1], [2.0], [1, 1], [0.5]) == -1
    assert dominates([1, 2], [0.0], [2, 1], [0.0]) == 0
    assert cons_value("params_M", 100, {"params_M": 120}) == pytest.approx(20)
    assert cons_value("tok_s_min", 10, {"tok_s": 12}) == pytest.approx(-2)


def test_elimination_keeps_first_front_and_checkpoint_roundtrip(tmp_path):
    s = ElasticSearchSpace("smollm2-135m", "blocks", seed=3)
    rng = random.Random(0)
    inds = [s.sample() for _ in range(12)]
    evals = [EvaluationResult([rng.random(), rng.random()], [0.0], {}) for _ in inds]
    pop = Population(inds[:6], evals[:6], s, rng=random.Random(5), n_population=6, n_offspring=6)
    pop.offspring, pop.offspring_evaluations = inds[6:], evals[6:]
    pop.update_elimination(key_fn=s.key)
    front = non_dominated([e.objs for e in evals])
    kept = {s.key(i) for i in pop.individuals}
    assert all(s.key(inds[i]) in kept for i in front[:6])
    path = pop.save_checkpoint(str(tmp_path / "ck.json"))
    back = Population.load_checkpoint(path, s)
    assert [s.key(i) for i in back.individuals] == [s.key(i) for i in pop.individuals]
    assert back.rng.random() == pop.rng.random()


class FakeSupernet:
    """Deterministic stand-in for SwSupernet: loss falls with width and attention capacity."""

    def __init__(self, space, *args, **kwargs):
        self.space, self.n_scored, self.settings = space, 0, {"fake": True}

    def score(self, ind):
        x = Individual.from_dict(ind)
        body = x.estimate_params() - x.embedding_params()
        attn = sum(li["n_head"] * li["n_qk_head_dim"] for li in x["layers"])
        return 2.5 + 30.0 / (1 + body / 1e7) + 50.0 / (1 + attn)

    def evaluate(self, inds):
        self.n_scored += len(inds)
        return [self.score(i) for i in inds], [0.0] * len(inds)

    def with_docs(self, n_docs, skip_docs):
        return self


def _run(monkeypatch, out, *extra):
    import llmforge.evaluators.sw_supernet as sw
    from llmforge.search import cosearch

    monkeypatch.setattr(sw, "SwSupernet", FakeSupernet)
    cosearch.main(["--space", str(CONFIGS / "search_spaces" / "smollm2-135m.yaml"), "--supernet", str(out),
                   "--hw", "analytic", "--objectives", "val_loss", "params_M", "--pop", "8",
                   "--generations", "3", "--no-gpu-lock", "--out", str(out), *extra])
    return [json.loads(l) for l in (out / "evals.jsonl").read_text().splitlines()]


def test_dispatcher_is_deterministic_and_complete(monkeypatch, tmp_path):
    a = _run(monkeypatch, tmp_path / "a", "--seed", "1")
    b = _run(monkeypatch, tmp_path / "b", "--seed", "1")
    c = _run(monkeypatch, tmp_path / "c", "--seed", "2")
    assert [r["key"] for r in a] == [r["key"] for r in b]
    assert [r["key"] for r in a] != [r["key"] for r in c]
    assert (tmp_path / "a" / "DONE").exists()
    assert len((tmp_path / "a" / "trace.jsonl").read_text().splitlines()) == 4
    front = json.loads((tmp_path / "a" / "front.json").read_text())["front"]
    objs = [r["objs"] for r in front]
    assert non_dominated(objs) == list(range(len(objs)))


def test_dispatcher_constraint_marks_infeasible(monkeypatch, tmp_path):
    recs = _run(monkeypatch, tmp_path / "k", "--constraint", "params_M<=100")
    assert any(not r["feasible"] for r in recs)
    front = json.loads((tmp_path / "k" / "front.json").read_text())["front"]
    assert all(r["metrics"]["params_M"] <= 100 for r in front)


def test_dispatcher_random_grid_and_list(monkeypatch, tmp_path):
    rand = _run(monkeypatch, tmp_path / "r", "--algo", "random")
    assert len(rand) == 8 + 3 * 8
    grid = _run(monkeypatch, tmp_path / "g", "--algo", "grid", "--partition", "uniform")
    assert len({r["key"] for r in grid}) == 192
    lst = _run(monkeypatch, tmp_path / "l", "--algo", "list", "--partition", "per_layer",
               "--archs", str(tmp_path / "r" / "front.json"))
    assert {r["key"] for r in lst} >= {r["key"] for r in json.loads((tmp_path / "r" / "front.json").read_text())["front"]}
