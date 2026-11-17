"""Base-initialised reference training and the search's scoring protocol.

groundtruth_het.py --init base --base-align must give the student the supernet's extra
parameters without changing the model, and must make n_kv below full width trainable at all.
eval_het.py --docs must score on exactly the inputs the search's evaluator builds.

The model tests run SmolLM2-135M on CPU and skip when its weights are not cached locally.
"""
import os

import pytest

torch = pytest.importorskip("torch")

from llmforge.supernet.train.base_init import needs_kv_alignment  # noqa: E402


def test_needs_kv_alignment():
    assert not needs_kv_alignment(None, 3)
    assert not needs_kv_alignment([3] * 30, 3)
    assert needs_kv_alignment([3] * 12 + [1] * 6 + [3] * 12, 3)
    assert needs_kv_alignment(1, 3)


@pytest.fixture(scope="module")
def smol():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llmforge.supernet.config import SPECS
    spec = SPECS["smollm2-135m"]
    try:
        tok = AutoTokenizer.from_pretrained(spec.repo)
        model = AutoModelForCausalLM.from_pretrained(
            spec.repo, torch_dtype=torch.float32, attn_implementation="eager").eval()
    except Exception as e:  # not cached and offline
        pytest.skip(f"SmolLM2-135M not available: {e}")
    return spec, tok, model


def _cfg(spec, kv_block):
    from llmforge.supernet.space import ElasticConfig
    full = ElasticConfig.full(spec)
    n_kv = [spec.n_kv] * spec.n_layers
    for i in kv_block:
        n_kv[i] = 1
    return ElasticConfig(spec, full.d_qk, full.d_v, n_kv, full.n_h, full.d_mlp)


def test_base_align_is_identity_and_unlocks_nkv(smol):
    from llmforge.supernet.elastic import (disable_elastic, enable_elastic, pair_order_for,
                                           set_elastic_backend, set_elastic_config)
    from llmforge.supernet.train.base_init import attach_supernet_params
    spec, _, model = smol
    order = pair_order_for(spec)
    ids = torch.arange(1, 33).unsqueeze(0)
    full = _cfg(spec, [])
    reduced = _cfg(spec, range(12, 18))
    enable_elastic(model)
    try:
        set_elastic_backend("eager")
        model.config.use_cache = False
        set_elastic_config(model, full.d_qk, full.d_v, order, n_h=full.n_h, d_mlp=full.d_mlp)
        with torch.no_grad():
            before = model(input_ids=ids).logits.clone()
        # Without the alignment the reduced architecture cannot even be configured, which is why
        # --init base needs the switch.
        with pytest.raises(RuntimeError, match="alignment"):
            set_elastic_config(model, reduced.d_qk, reduced.d_v, order, n_h=reduced.n_h,
                               d_mlp=reduced.d_mlp, n_kv=reduced.n_kv)
        attach_supernet_params(model)
        set_elastic_config(model, full.d_qk, full.d_v, order, n_h=full.n_h, d_mlp=full.d_mlp,
                           n_kv=full.n_kv)
        with torch.no_grad():
            after = model(input_ids=ids).logits
        assert torch.equal(before, after), (before - after).abs().max().item()
        set_elastic_config(model, reduced.d_qk, reduced.d_v, order, n_h=reduced.n_h,
                           d_mlp=reduced.d_mlp, n_kv=reduced.n_kv)
        out = model(input_ids=ids).logits
        out.float().logsumexp(-1).mean().backward()
        angles = [p for n, p in model.named_parameters() if "kv_ang" in n]
        assert angles and any(p.grad is not None and p.grad.abs().sum() > 0 for p in angles)
    finally:
        disable_elastic()


def test_search_protocol_matches_make_evaluator(smol):
    from llmforge.supernet.data.heldout import heldout_texts
    from llmforge.supernet.elastic import (disable_elastic, enable_elastic, pair_order_for,
                                           set_elastic_backend)
    from llmforge.supernet.eval.protocol import search_protocol_ids
    from llmforge.supernet.search.nsga import make_evaluator
    from llmforge.supernet.train.base_init import attach_supernet_params
    import torch.nn.functional as F
    spec, tok, model = smol
    order = pair_order_for(spec)
    texts = heldout_texts(3, skip=32)
    cfg = _cfg(spec, range(12, 18))
    enable_elastic(model)
    try:
        set_elastic_backend("eager")
        model.config.use_cache = False
        if not any("kv_ang" in n for n, _ in model.named_parameters()):
            attach_supernet_params(model)
        with torch.no_grad():
            loss_of, n = make_evaluator(model, tok, order, spec, texts, dev="cpu", max_len=64)
            ref = loss_of(cfg)                      # also pins cfg on the model
            ids = search_protocol_ids(tok, texts, max_len=64, dev="cpu")
            assert len(ids) == n == 3
            tot = cnt = 0.0
            for x in ids:
                lg = model(input_ids=x).logits[:, :-1].float()
                tot += F.cross_entropy(lg.reshape(-1, lg.shape[-1]), x[:, 1:].reshape(-1),
                                       reduction="sum").item()
                cnt += x[:, 1:].numel()
        assert abs(tot / cnt - ref) < 1e-6
    finally:
        disable_elastic()


def _units(tmp_path):
    import json
    return json.load(open(str(tmp_path / "nll_units.json")))


def test_save_scores_merges_summary_and_units(tmp_path):
    """A later call that passes only new models keeps the earlier models in both files."""
    import json
    from llmforge.supernet.eval.protocol import META, save_scores
    out = str(tmp_path / "nll.json")
    save_scores(out, {"S90": {"nll": 1.0}}, {"S90": [[1.0, 2.0], [3.0, 4.0]]}, {"S90": "a"})
    save_scores(out, {"S65": {"nll": 2.0}}, {"S65": [[5.0, 2.0], [6.0, 4.0]]}, {"S65": "b"})
    assert json.load(open(out)) == {"S90": {"nll": 1.0}, "S65": {"nll": 2.0}}
    u = _units(tmp_path)
    assert u["S90"] == [[1.0, 2.0], [3.0, 4.0]] and u["S65"] == [[5.0, 2.0], [6.0, 4.0]]
    assert u[META] == {"S90": "a", "S65": "b"}
    assert not [f for f in os.listdir(tmp_path) if ".tmp." in f]


def test_rescore_replaces_summary_and_units(tmp_path):
    import json
    from llmforge.supernet.eval.protocol import META, needs_score, save_scores
    out = str(tmp_path / "nll.json")
    save_scores(out, {"S90": {"nll": 1.0, "model_sha256": "a", "cfg_sha256": "c"}},
                {"S90": [[1.0, 2.0]]}, {"S90": "a"})
    res = json.load(open(out))
    assert not needs_score(res, "S90", "a", "c")
    assert needs_score(res, "S90", "b", "c")            # retrained under the same name
    assert needs_score(res, "S90", "a", "d")            # same weights, another config
    assert needs_score(res, "S90", "a", "c", force=True)
    assert needs_score(res, "U90", "a", "c")
    save_scores(out, {"S90": {"nll": 0.5, "model_sha256": "b", "cfg_sha256": "c"}},
                {"S90": [[0.25, 2.0]]}, {"S90": "b"})
    assert json.load(open(out))["S90"]["nll"] == 0.5
    assert _units(tmp_path)["S90"] == [[0.25, 2.0]] and _units(tmp_path)[META] == {"S90": "b"}


def test_save_scores_refuses_another_doc_range(tmp_path):
    import json
    from llmforge.supernet.eval.protocol import save_scores
    out = str(tmp_path / "nll.json")
    save_scores(out, {"S90": {"nll": 1.0}}, {"S90": [[1.0, 2.0], [3.0, 4.0]]})
    before = (open(out).read(), open(str(tmp_path / "nll_units.json")).read())
    with pytest.raises(ValueError, match="token counts"):
        save_scores(out, {"U90": {"nll": 1.0}}, {"U90": [[1.0, 5.0], [3.0, 4.0]]})
    with pytest.raises(ValueError, match="token counts"):
        save_scores(out, {"U90": {"nll": 1.0}}, {"U90": [[1.0, 2.0], [3.0, 4.0], [1.0, 1.0]]})
    save_scores(out, {"S65": {"nll": 1.0, "docs": [32, 96]}}, {"S65": [[0.0, 2.0], [0.0, 4.0]]})
    with pytest.raises(ValueError, match="another range"):
        save_scores(out, {"U65": {"nll": 1.0, "docs": [96, 192]}}, {"U65": [[0.0, 2.0], [0.0, 4.0]]})
    assert set(json.load(open(out))) == {"S90", "S65"}
    assert open(out).read() != before[0]                # only the accepted S65 merge landed


def test_load_base_elastic_is_base_plus_identity_params(smol):
    """The base slice starts from exactly the published weights, plus only the zero-init extras."""
    from llmforge.supernet.elastic import disable_elastic
    from llmforge.supernet.eval.protocol import load_base_elastic
    from transformers import AutoModelForCausalLM
    spec, _, _ = smol
    ref = AutoModelForCausalLM.from_pretrained(spec.repo, torch_dtype=torch.float32).state_dict()
    try:
        m = load_base_elastic(spec, dev="cpu", dtype=torch.float32)
        sd = m.state_dict()
        extra = set(sd) - set(ref)
        assert extra and all("kv_ang" in k or "logit_scale" in k for k in extra), sorted(extra)[:5]
        assert set(ref) <= set(sd)
        assert all(torch.equal(sd[k], ref[k]) for k in ref)
    finally:
        disable_elastic()
