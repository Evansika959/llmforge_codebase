"""M0 gate: the elastic attention module must equal stock Qwen3 at qk=v=128.

Needs CUDA and a model download, so it skips at MODULE level. `pytestmark` is not enough:
this file previously ran its body at import time, and a mark cannot prevent an import, so
collection hard-errored on any CPU machine and took the whole CPU suite down with it.
`pytest.skip(..., allow_module_level=True)` is the construct that actually works here.
"""
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs CUDA and a model download", allow_module_level=True)

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

import llmforge.supernet.elastic as eq  # noqa: E402

MODEL = "Qwen/Qwen3-0.6B-Base"
TEXT = (
    "In a quiet village nestled between two mountains, an old clockmaker named Tomas "
    "spent his days repairing timepieces. One winter morning, a stranger arrived carrying "
    "a broken pocket watch engraved with the initials E.J. Tomas recognized the engraving "
    "immediately, for it matched the watch his father had lost decades ago in the war."
)


def _nll(logits, ids):
    lp = torch.log_softmax(logits[:, :-1].float(), -1)
    return -lp.gather(-1, ids[:, 1:].unsqueeze(-1)).mean().item()


@pytest.fixture(scope="module")
def fixture():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float32, attn_implementation="eager").to("cuda").eval()
    ids = tok(TEXT, return_tensors="pt").input_ids.to("cuda")
    with torch.no_grad():
        stock = model(ids).logits
    eq.enable_elastic(model)
    order = eq.build_pair_order("hilo")
    yield model, ids, stock, order
    eq.disable_elastic()


def test_full_width_matches_stock(fixture):
    """At qk=v=128 the gather is a permutation, so only float reassociation should differ."""
    model, ids, stock, order = fixture
    eq.set_elastic_config(model, 128, 128, order)
    with torch.no_grad():
        full = model(ids).logits
    maxdiff = (stock - full).abs().max().item()
    assert maxdiff < 1e-3, f"elastic module != stock HF at full width (max|dlogit|={maxdiff:.3e})"


@pytest.mark.parametrize("qk,v", [(128, 128), (96, 128), (128, 96), (64, 64),
                                  (32, 128), (128, 32), (32, 32)])
def test_slices_run_and_stay_finite(fixture, qk, v):
    model, ids, stock, order = fixture
    eq.set_elastic_config(model, qk, v, order)
    with torch.no_grad():
        lg = model(ids).logits
    assert torch.isfinite(lg).all(), f"non-finite logits at qk={qk} v={v}"


def test_smaller_slices_degrade(fixture):
    """The minimum slice must be worse than full width, or slicing is not doing anything."""
    model, ids, stock, order = fixture
    eq.set_elastic_config(model, 128, 128, order)
    with torch.no_grad():
        best = _nll(model(ids).logits, ids)
    eq.set_elastic_config(model, 32, 32, order)
    with torch.no_grad():
        worst = _nll(model(ids).logits, ids)
    assert worst > best, f"min slice ({worst:.4f}) not worse than full ({best:.4f})"
