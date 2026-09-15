"""Calibrated cost models: a fit recovers an exact cost, and the evaluator reproduces the fitted model."""
import json

import numpy as np
import pytest

from llmforge.evaluators.hw_analytic import HwAnalytic
from llmforge.evaluators.hw_fitted import HwFitted, fit_cost_model, int8_group_size
from llmforge.paths import CONFIGS
from llmforge.search.elastic_space import ElasticSearchSpace


def write_run(tmp_path, cost_fn, seq_len=1024):
    space = ElasticSearchSpace.from_yaml(str(CONFIGS / "search_spaces" / "smollm2-135m_uniform.yaml"))
    inds = list(space.enumerate_uniform())
    metrics = HwAnalytic(seq_len=seq_len).evaluate(inds)
    run = tmp_path / "run"
    run.mkdir()
    args = {"objectives": ["val_loss", "cost_uJ"], "analytic_seq_len": seq_len}
    (run / "run.json").write_text(json.dumps({"args": args}))
    with open(run / "evals.jsonl", "w") as f:
        for ind, m in zip(inds, metrics):
            rec = {"feasible": True, "individual": ind, "metrics": {**m, "cost_uJ": cost_fn(ind, m)}}
            f.write(json.dumps(rec) + "\n")
    return run, inds


def test_linear_model_recovers_an_exact_cost(tmp_path):
    run, inds = write_run(tmp_path, lambda ind, m: 3.0 * m["params_M"] + 7.0 * m["kv_cache_MB"] + 11.0)
    model = fit_cost_model(str(run), "params_M+kv_cache_MB")
    assert model["coef"]["params_M"] == pytest.approx(3.0, rel=1e-6)
    assert model["coef"]["kv_cache_MB"] == pytest.approx(7.0, rel=1e-6)
    assert model["cv_median_abs_rel_err"] < 1e-6
    assert model["recorded_metric_max_rel_diff"] < 1e-9
    path = tmp_path / "model.json"
    path.write_text(json.dumps(model))
    out = HwFitted(str(path)).evaluate(inds[:5])
    expected = [3.0 * m["params_M"] + 7.0 * m["kv_cache_MB"] + 11.0 for m in HwAnalytic().evaluate(inds[:5])]
    assert [o["fitted_cost_uJ"] for o in out] == pytest.approx(expected, rel=1e-6)
    assert all(o["hw_feasible"] for o in out)


def test_log_model_recovers_a_group_size_factor(tmp_path):
    def cost(ind, m):
        return 2.0 * m["params_M"] ** 1.5 * (2.5 if int8_group_size(ind) == 64 else 1.0)

    run, inds = write_run(tmp_path, cost)
    model = fit_cost_model(str(run), "log:params_M+int8_group_64+int8_group_16")
    assert model["coef"]["params_M"] == pytest.approx(1.5, rel=1e-6)
    assert np.exp(model["coef"]["int8_group_64"]) == pytest.approx(2.5, rel=1e-6)
    assert model["cv_median_abs_rel_err"] < 1e-6


def test_group_size_follows_the_widest_common_divisor():
    space = ElasticSearchSpace.from_yaml(str(CONFIGS / "search_spaces" / "smollm2-135m_uniform.yaml"))
    assert int8_group_size(space.uniform(n_head=3, n_qk_head_dim=64, n_v_head_dim=64, mlp_size=1536)) == 64
    assert int8_group_size(space.uniform(n_head=3, n_qk_head_dim=64, n_v_head_dim=32, mlp_size=1536)) == 32
    assert int8_group_size(space.uniform(n_head=3, n_qk_head_dim=64, n_v_head_dim=16, mlp_size=1536)) == 16
