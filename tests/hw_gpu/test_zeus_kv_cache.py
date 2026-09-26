"""GPU tests for the measured decode path: the preallocated KV cache must reproduce the uncached model.

Skipped without CUDA or without the vendored GPT implementation. Run on an idle GPU, since the
tests allocate models while other GPU work may be measuring energy.
"""
import pytest

torch = pytest.importorskip("torch")

from llmforge.paths import GPT_MODEL

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not (GPT_MODEL / "model.py").exists(),
    reason="needs CUDA and the vendored GPT implementation")


def _individual(space_yaml, seed, uniform):
    from llmforge.paths import CONFIGS
    from llmforge.search.elastic_space import ElasticSearchSpace

    s = ElasticSearchSpace.from_yaml(str(CONFIGS / "search_spaces" / space_yaml), seed=seed)
    if uniform:
        return s.uniform(**{k: s.grids[k][len(s.grids[k]) // 2] for k in s.grids})
    return s.sample()


@pytest.mark.parametrize("space_yaml,uniform", [("smollm2-135m_uniform.yaml", True),
                                                ("smollm2-135m.yaml", False),
                                                ("qwen3-0.6b.yaml", False)])
def test_cached_decode_matches_uncached_forward(space_yaml, uniform):
    from llmforge.evaluators.cache import gpu_lock
    from llmforge.hw.zeus.kv_cache import attach_iha_kv_cache, detach_iha_kv_cache, parity_check
    from llmforge.hw.zeus.measure import build_model_from_individual

    ind = _individual(space_yaml, seed=11, uniform=uniform)
    with gpu_lock():
        model = build_model_from_individual(ind, 64, torch.device("cuda"), torch.float32)
        try:
            attach_iha_kv_cache(model)
            ok, max_abs, max_rel = parity_check(model, prefill_len=24, decode_len=8, atol=1e-3,
                                                rtol=1e-3, verbose=False)
            assert ok, (max_abs, max_rel)
        finally:
            detach_iha_kv_cache(model)
            del model
            torch.cuda.empty_cache()


@pytest.mark.parametrize("space_yaml", ["smollm2-135m.yaml", "qwen3-0.6b.yaml"])
def test_cuda_graph_decode_matches_eager_cached_decode(space_yaml):
    from llmforge.evaluators.cache import gpu_lock
    from llmforge.hw.zeus.kv_cache import attach_iha_kv_cache, detach_iha_kv_cache, graph_parity_check
    from llmforge.hw.zeus.measure import build_model_from_individual

    ind = _individual(space_yaml, seed=5, uniform=False)
    with gpu_lock():
        model = build_model_from_individual(ind, 64, torch.device("cuda"), torch.float32)
        try:
            attach_iha_kv_cache(model)
            assert graph_parity_check(model, prefill_len=24, decode_len=8, batch_size=2) <= 1e-4
        finally:
            detach_iha_kv_cache(model)
            del model
            torch.cuda.empty_cache()
