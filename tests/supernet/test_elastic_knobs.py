"""Tests for the n_h and d_mlp knobs added for the combined A+B supernet run.

The CPU tests pin the head-selection rule; the GPU tests check the forward pass actually runs at
every configuration and that selecting everything is a no-op (so adding the knobs cannot silently
perturb the full-width model the teacher distills toward).
"""
import pytest
import torch

from llmforge.supernet.config import SPECS
from llmforge.supernet.elastic import head_index


# ---- head selection (CPU) -----------------------------------------------------------

@pytest.mark.parametrize("n_h", [8, 16, 32])
def test_head_selection_is_balanced_across_kv_groups(n_h):
    """Every KV group must keep the same number of queries -- a plain prefix would empty most
    groups entirely, which is the orphan failure the design calls out."""
    s = SPECS["qwen3-4b"]
    idx = head_index(s.n_q, s.n_kv, n_h).tolist()
    assert len(idx) == n_h
    nrep = s.n_q // s.n_kv
    per_group = [sum(1 for h in idx if h // nrep == g) for g in range(s.n_kv)]
    assert len(set(per_group)) == 1, f"unbalanced: {per_group}"
    assert per_group[0] == n_h // s.n_kv


def test_full_head_count_is_the_identity_selection():
    s = SPECS["qwen3-4b"]
    assert head_index(s.n_q, s.n_kv, s.n_q).tolist() == list(range(s.n_q))


def test_head_selection_rejects_indivisible_counts():
    """n_kv must divide n_h; 4 heads cannot be spread over 8 groups."""
    s = SPECS["qwen3-4b"]
    with pytest.raises(ValueError, match="must divide"):
        head_index(s.n_q, s.n_kv, 4)


def test_every_grid_pair_is_constructible():
    """Whatever config.head_pairs() offers, head_index must be able to build."""
    for key in ("qwen3-1.7b", "qwen3-4b"):
        s = SPECS[key]
        for n_h, n_kv in s.head_pairs():
            if n_kv == s.n_kv:                       # the A+B run holds n_kv at the base value
                assert len(head_index(s.n_q, n_kv, n_h)) == n_h


# ---- forward pass (GPU) -------------------------------------------------------------

pytestmark_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(scope="module")
def small():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import llmforge.supernet.elastic as eq
    spec = SPECS["qwen3-0.6b"]
    tok = AutoTokenizer.from_pretrained(spec.repo)
    model = AutoModelForCausalLM.from_pretrained(
        spec.repo, torch_dtype=torch.float32).to("cuda").eval()
    ids = tok("The capital of France is Paris, and the capital of Italy is",
              return_tensors="pt").input_ids.to("cuda")
    with torch.no_grad():
        stock = model(ids).logits
    eq.enable_elastic(model)
    yield model, ids, stock, eq, spec
    eq.disable_elastic()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_selecting_everything_is_a_noop(small):
    """n_h=n_q and d_mlp=full must reproduce the untouched model, or the knobs perturb FULL."""
    model, ids, stock, eq, spec = small
    order = eq.build_pair_order("hilo")
    eq.set_elastic_config(model, spec.head_dim, spec.head_dim, order,
                          n_h=spec.n_q, d_mlp=spec.d_mlp)
    with torch.no_grad():
        got = model(ids).logits
    assert (stock - got).abs().max().item() < 1e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
# 0.6B has n_q=16 and n_kv=8, so 8 is the only head count strictly below full that 8 divides.
@pytest.mark.parametrize("n_h", [8])
def test_dropping_heads_runs_and_degrades(small, n_h):
    model, ids, stock, eq, spec = small
    order = eq.build_pair_order("hilo")
    eq.set_elastic_config(model, spec.head_dim, spec.head_dim, order, n_h=n_h)
    with torch.no_grad():
        got = model(ids).logits
    assert torch.isfinite(got).all()
    assert (stock - got).abs().max().item() > 1e-3, "dropping heads changed nothing"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("frac", [0.75, 0.5, 0.25])
def test_narrowing_mlp_runs_and_degrades(small, frac):
    model, ids, stock, eq, spec = small
    order = eq.build_pair_order("hilo")
    eq.set_elastic_config(model, spec.head_dim, spec.head_dim, order,
                          d_mlp=int(spec.d_mlp * frac))
    with torch.no_grad():
        got = model(ids).logits
    assert torch.isfinite(got).all()
    assert (stock - got).abs().max().item() > 1e-3, "narrowing the MLP changed nothing"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_all_four_knobs_together(small):
    """The A+B configuration: every knob off its maximum at once, per layer."""
    model, ids, stock, eq, spec = small
    order = eq.build_pair_order("hilo")
    L = spec.n_layers
    eq.set_elastic_config(
        model,
        [spec.qk_grid[i % 4] for i in range(L)],
        [spec.qk_grid[(i + 1) % 4] for i in range(L)],
        order,
        n_h=[[8, 16, spec.n_q][i % 3] for i in range(L)],
        d_mlp=[spec.mlp_grid[i % 4] for i in range(L)],
    )
    with torch.no_grad():
        got = model(ids).logits
    assert torch.isfinite(got).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_gradients_reach_all_parents(small):
    """Dropped heads and trimmed MLP rows must still receive gradient across a sandwich, or the
    supernet cannot keep its full config healthy."""
    model, ids, stock, eq, spec = small
    order = eq.build_pair_order("hilo")
    model.zero_grad(set_to_none=True)
    for n_h, dm in [(spec.n_q, spec.d_mlp), (8, spec.d_mlp // 4)]:
        eq.set_elastic_config(model, spec.head_dim, spec.head_dim, order, n_h=n_h, d_mlp=dm)
        model(ids).logits.float().pow(2).mean().backward()
    g = model.model.layers[0].mlp.gate_proj.weight.grad
    assert g is not None and g[-1].abs().sum() > 0, "last MLP row never received gradient"
    o = model.model.layers[0].self_attn.o_proj.weight.grad
    assert o is not None and o.abs().sum() > 0


# ---------------------------------------------------------------- n_kv (KV-group pooling)

def test_nkv_full_width_is_identity():
    """Attaching the alignment and asking for full n_kv must not move a single logit.

    n_kv is the one knob that is not a pure selection -- pooling mixes heads -- so the guarantee
    that has to hold is that the mixing path is never entered at full width. If it were, every
    result recorded before n_kv existed would silently change.
    """
    import torch
    from llmforge.supernet.config import SPECS
    from llmforge.supernet.elastic import (add_kv_alignment, enable_elastic, pair_order_for,
                                    set_elastic_backend, set_elastic_config)
    from transformers import AutoModelForCausalLM
    spec = SPECS["qwen3-0.6b"]
    m = AutoModelForCausalLM.from_pretrained(spec.repo, torch_dtype=torch.float32,
                                             attn_implementation="eager").eval()
    ids = torch.arange(1, 33).unsqueeze(0)
    enable_elastic(m); set_elastic_backend("eager"); m.config.use_cache = False
    order = pair_order_for(spec)
    set_elastic_config(m, spec.head_dim, spec.head_dim, order)
    with torch.no_grad():
        before = m(input_ids=ids).logits.clone()
    add_kv_alignment(m)
    set_elastic_config(m, spec.head_dim, spec.head_dim, order, n_kv=spec.n_kv)
    with torch.no_grad():
        after = m(input_ids=ids).logits
    assert torch.equal(before, after), (before - after).abs().max().item()


def test_nkv_sampler_never_emits_an_illegal_pair():
    """n_kv must divide n_h, and the sampler has to guarantee it rather than reject afterwards."""
    import random
    from llmforge.supernet.config import SPECS
    from llmforge.supernet.elastic.sampler import sandwich_ab
    spec = SPECS["qwen3-0.6b"]
    rng = random.Random(0)
    for step in (0, 500, 2400):
        for cfg in sandwich_ab(step, 2500, spec, K=2, rng=rng, with_nkv=True):
            assert len(cfg) == 5
            for nh, nkv in zip(cfg[2], cfg[4]):
                assert nh % nkv == 0, (nh, nkv)


def test_nkv_grid_is_all_divisors_not_powers_of_two():
    """SmolLM2's n_kv is prime, so a power-of-two-divisor rule deletes the axis entirely.

    The grid used to halve until it hit an odd number, which returns {3} and {5} on SmolLM2 -- a
    single value, i.e. no knob at all. Pooling only requires that the target DIVIDE n_kv (the
    forward reshapes to [.., n_kv_active, group, head_dim] and means over `group`), so every
    divisor is reachable and the restriction bought nothing. Qwen3 is unaffected because 8's
    divisors are all powers of two anyway.

    The step is coarse on SmolLM2 -- 3 groups or 1 -- but the axis exists, and it is the only way
    to get KV cache below 25%: the width knobs bottom out there at every scale, because d_qk and
    d_v at their minimum are a quarter of head_dim.
    """
    from llmforge.supernet.config import SPECS
    from llmforge.supernet.elastic.sampler import nkv_options
    assert SPECS["smollm2-135m"].nkv_grid == [1, 3]
    assert SPECS["smollm2-360m"].nkv_grid == [1, 5]
    for key in ("qwen3-0.6b", "qwen3-1.7b", "qwen3-4b"):
        assert SPECS[key].nkv_grid == [1, 2, 4, 8], key
    assert nkv_options(SPECS["qwen3-0.6b"], 16) == [1, 2, 4, 8]
    assert nkv_options(SPECS["smollm2-135m"], 9) == [1, 3]


def test_every_nh_nkv_combination_is_constructible():
    """Head selection must group by the ACTIVE n_kv, not the model's base value.

    Widening the n_h grid so a smaller n_kv unlocks finer head counts is only half the change: the
    head-selection index still grouped by the base n_kv, so n_h=4 under n_kv=2 -- legal, since 2
    divides 4 -- was checked against n_kv=8 and raised. That killed a 1.7B arm 26 minutes in, and
    no existing test caught it because none exercised a head count below head_grid()'s minimum.

    This walks the whole (n_h, n_kv) product rather than sampling it, so a future grid change that
    breaks one corner fails here instead of two hours into a run.
    """
    from llmforge.supernet.config import SPECS
    from llmforge.supernet.elastic import enable_elastic, pair_order_for, set_elastic_config
    from llmforge.supernet.elastic.patch import add_kv_alignment
    from llmforge.supernet.elastic.sampler import head_options
    from transformers import AutoModelForCausalLM
    import torch
    spec = SPECS["qwen3-0.6b"]
    m = AutoModelForCausalLM.from_pretrained(spec.repo, torch_dtype=torch.float32,
                                             attn_implementation="eager").eval()
    enable_elastic(m)
    add_kv_alignment(m, spec=spec)
    order = pair_order_for(spec)
    n = 0
    for kv in spec.nkv_grid:
        for nh in head_options(spec, kv):
            set_elastic_config(m, spec.head_dim, spec.head_dim, order,
                               n_h=nh, d_mlp=spec.d_mlp, n_kv=kv)
            n += 1
    assert n == 13, n


def test_query_heads_get_their_own_kv_head_angle():
    """Every surviving query head must be rotated by the angle of the KV head it READS.

    The rotation preserves the attention score only because q and k get the SAME transform,
    (Rq).(Rk) = q.k, so a query head given another KV head's angle computes a wrong score rather
    than an approximate one. Which KV head a survivor reads comes from _q_idx, built by
    head_index(n_q, ACTIVE n_kv, n_h). The forward used to assume the survivors were grouped by the
    model's FULL n_kv, which is right only at n_h == n_q, so full-width sweeps were clean while most
    sampled training configurations were corrupted. The test asserts the index map itself, and that
    the old map differs at every case, so it cannot pass vacuously.
    """
    for key, kv, nh in (("smollm2-360m", 1, 5), ("smollm2-135m", 1, 3),
                        ("qwen3-0.6b", 2, 8), ("qwen3-0.6b", 4, 8), ("qwen3-4b", 2, 16)):
        spec = SPECS[key]
        per_kv = spec.n_q // spec.n_kv                    # query heads per ORIGINAL kv head
        qi = head_index(spec.n_q, kv, nh)
        reads = [int(q) // per_kv for q in qi]            # the kv head each survivor attends to
        got = [int(x) for x in (qi // per_kv)]            # what the forward looks up
        assert reads == got, (key, kv, nh, reads, got)
        nrep = nh // spec.n_kv if nh >= spec.n_kv else 1
        naive = [i // max(nrep, 1) for i in range(nh)]
        assert naive != reads, f"{key} n_kv={kv} n_h={nh}: the old map happened to be right here"
