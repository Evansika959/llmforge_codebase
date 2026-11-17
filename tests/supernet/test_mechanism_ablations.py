"""The trainer's mechanism-ablation switches, CPU-only on a tiny random Llama.

  --align none        the alignment angles stay exactly zero through training steps
  --align procrustes  the closed-form angles lower the KV merge error, and on heads that are exact
                      per-pair rotations of one another they make pooling lossless end to end
  --rope-order prefix a narrowed d_qk keeps a prefix of the rotary pairs, and a checkpoint
                      records the order so the search slices it the same way
  --space             the sampler draws only from the restricted (GQA-only) grids
"""
import json
import random
from pathlib import Path

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from llmforge.supernet.config import SPECS
from llmforge.supernet.elastic import (add_elastic_temperature, add_kv_alignment, build_pair_order,
                                       disable_elastic, enable_elastic, pair_order_for,
                                       qk_gather_index, set_elastic_backend, set_elastic_config)
from llmforge.supernet.elastic.align_init import (collect_kv, fit_procrustes_alignment,
                                                  merge_error, rotate_pairs)
from llmforge.supernet.elastic.sampler import restrict_spec, sandwich_ab
from llmforge.supernet.search.nsga import pair_order_of
from llmforge.supernet.train.uptrain import optimizer_groups, rope_pair_order

ROOT = Path(__file__).resolve().parents[2]
HD = 16


@pytest.fixture
def tiny():
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=64, intermediate_size=96, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=HD,
                      max_position_embeddings=64, attn_implementation="eager")
    m = LlamaForCausalLM(cfg).float().eval()
    m.config.use_cache = False
    enable_elastic(m)
    set_elastic_backend("eager")
    add_elastic_temperature(m)
    add_kv_alignment(m, rungs=[1])
    yield m
    disable_elastic(m)


def _order():
    return build_pair_order("hilo", HD // 2)


def _ids(n=2, s=24, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 64, (n, s), generator=g)


def _logits(m, ids, order, n_kv):
    set_elastic_config(m, HD, HD, order, n_kv=n_kv)
    with torch.no_grad():
        return m(input_ids=ids).logits


# ---------------------------------------------------------------- align=none

def test_align_none_angles_stay_zero_through_training(tiny):
    groups = optimizer_groups(tiny, "none", 1e-2, 0.1, 0.5)
    ang = [p for n, p in tiny.named_parameters() if "kv_ang" in n]
    assert ang and all(not p.requires_grad for p in ang)
    in_opt = {id(p) for g in groups for p in g["params"]}
    assert not any(id(p) in in_opt for p in ang), "frozen angles must not reach the optimizer"
    opt = torch.optim.AdamW(groups, lr=1e-2, weight_decay=0.1)
    tiny.train()
    for step in range(3):
        set_elastic_config(tiny, HD, HD, _order(), n_kv=1)       # pooled: angles are read
        loss = tiny(input_ids=_ids(seed=step), labels=_ids(seed=step)).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert all(torch.count_nonzero(p) == 0 for p in ang)


def test_align_learned_angles_move(tiny):
    """The control for the test above: under the default the same steps do move the angles."""
    opt = torch.optim.AdamW(optimizer_groups(tiny, "learned", 1e-2, 0.1, 0.5), lr=1e-2)
    tiny.train()
    set_elastic_config(tiny, HD, HD, _order(), n_kv=1)
    loss = tiny(input_ids=_ids(), labels=_ids()).loss
    opt.zero_grad()
    loss.backward()
    opt.step()
    assert any(torch.count_nonzero(p) > 0 for n, p in tiny.named_parameters() if "kv_ang" in n)


def test_no_other_alignment_parameters(tiny):
    """Beyond the base weights the elastic model adds only the temperature and the KV angles, so
    freezing the angles removes every learned alignment path."""
    base = set(LlamaForCausalLM(tiny.config).state_dict())
    added = {n.rsplit(".", 1)[-1] for n, _ in tiny.named_parameters() if n not in base}
    assert added == {"logit_scale", "kv_ang_k", "kv_ang_v"}


# ---------------------------------------------------------------- align=procrustes

def test_procrustes_lowers_merge_error(tiny):
    order = _order()
    set_elastic_config(tiny, HD, HD, order)
    kv = collect_kv(tiny, [(_ids(), None, None)], order)
    rep = fit_procrustes_alignment(tiny, kv)
    for r in rep:
        assert r["k_after"] < r["k_before"]
        assert r["v_after"] < r["v_before"]
    assert any(torch.count_nonzero(p) > 0 for n, p in tiny.named_parameters() if "kv_ang" in n)
    # The stored angles reproduce the reported error when applied the way kv_rotate applies them.
    m = tiny.model.layers[0].self_attn
    k = kv[0]["k"]
    assert merge_error(rotate_pairs(k, m.kv_ang_k[0].detach().float()), [[0, 1]]) == \
        pytest.approx(rep[0]["k_after"], rel=1e-4)


def _make_heads_rotations_of_each_other(m, seed=1):
    """Make KV head 1 an exact per-pair rotation of KV head 0, for k and for v."""
    g = torch.Generator().manual_seed(seed)
    half = HD // 2
    for L in m.model.layers:
        for proj in (L.self_attn.k_proj, L.self_attn.v_proj):
            W = proj.weight.data                                    # [n_kv*HD, hidden]
            phi = torch.rand(half, generator=g) * 6 - 3
            c, s = torch.cos(phi)[:, None], torch.sin(phi)[:, None]
            a, b = W[:half], W[half:HD]
            W[HD:HD + half] = a * c - b * s
            W[HD + half:2 * HD] = a * s + b * c


def test_procrustes_makes_pooling_exact_when_heads_are_rotations(tiny):
    """End to end through the elastic forward: if the two KV heads differ only by a per-pair
    rotation, the fitted angles undo it, pooling to one KV head loses nothing, and the pooled
    model reproduces the full one. Zero angles (align=none) do not."""
    _make_heads_rotations_of_each_other(tiny)
    order = _order()
    ids = _ids(seed=5)
    full = _logits(tiny, ids, order, n_kv=None)
    plain = _logits(tiny, ids, order, n_kv=1)
    set_elastic_config(tiny, HD, HD, order)
    fit_procrustes_alignment(tiny, collect_kv(tiny, [(_ids(seed=7), None, None)], order))
    aligned = _logits(tiny, ids, order, n_kv=1)
    assert (aligned - full).abs().max() < 1e-4
    assert (plain - full).abs().max() > 1e-2


# ---------------------------------------------------------------- rope_order=prefix

@pytest.mark.parametrize("model", ["smollm2-135m", "qwen3-0.6b"])
def test_prefix_order_keeps_a_prefix_of_pairs(model):
    spec = SPECS[model]
    half = spec.head_dim // 2
    order = rope_pair_order(spec, "prefix")
    assert order == list(range(half))
    for qk in spec.qk_grid:
        n = qk // 2
        assert qk_gather_index(order, qk).tolist() == list(range(n)) + list(range(half, half + n))
    assert rope_pair_order(spec, "interleaved") == pair_order_for(spec)
    assert rope_pair_order(spec, "interleaved")[:2] == [0, half - 1]


def test_checkpoint_records_the_order(tmp_path):
    spec = SPECS["smollm2-135m"]
    assert pair_order_of(str(tmp_path), spec) == pair_order_for(spec)       # no run.json
    (tmp_path / "run.json").write_text(json.dumps({"model": spec.key}))
    assert pair_order_of(str(tmp_path), spec) == pair_order_for(spec)       # field absent
    (tmp_path / "run.json").write_text(json.dumps({"rope_order": "prefix"}))
    assert pair_order_of(str(tmp_path), spec) == list(range(spec.head_dim // 2))


# ---------------------------------------------------------------- --space (GQA-only)

def test_gqa_space_restricts_sampling():
    spec = SPECS["smollm2-135m"]
    r = restrict_spec(spec, ROOT / "configs/search_spaces/smollm2-135m_nkv_gqa.yaml")
    assert r.qk_grid == [64] and r.nh_grid == [3, 6, 9] and r.nkv_grid == [1, 3]
    assert r.mlp_grid == spec.mlp_grid and r.n_layers == spec.n_layers
    rng = random.Random(0)
    seen_h, seen_kv = set(), set()
    for step in range(0, 400, 7):
        for qk, v, nh, dm, kv in sandwich_ab(step, 400, r, K=2, rng=rng, with_nkv=True):
            assert set(qk) == {64} and set(v) == {64}
            assert set(dm) <= set(spec.mlp_grid)
            assert all(h % k == 0 for h, k in zip(nh, kv))
            seen_h |= set(nh)
            seen_kv |= set(kv)
    assert seen_h == {3, 6, 9} and seen_kv == {1, 3}


def test_full_space_is_unrestricted():
    spec = SPECS["smollm2-135m"]
    r = restrict_spec(spec, ROOT / "configs/search_spaces/smollm2-135m_nkv.yaml")
    assert (r.qk_grid, r.nh_grid, r.nkv_grid, r.mlp_grid) == \
        (spec.qk_grid, spec.nh_grid, spec.nkv_grid, spec.mlp_grid)


def test_space_rejects_another_model():
    with pytest.raises(ValueError):
        restrict_spec(SPECS["smollm2-360m"], ROOT / "configs/search_spaces/smollm2-135m_nkv.yaml")


# ---------------------------------------------------------------- --min-free-gb

def test_free_space_guard(tmp_path):
    import shutil
    from llmforge.supernet.train.uptrain import check_free_space, checkpoint_bytes
    free = shutil.disk_usage(tmp_path).free
    assert check_free_space(tmp_path, 0.0) == free
    with pytest.raises(SystemExit):
        check_free_space(tmp_path, free / 1e9 + 1)
    with pytest.raises(SystemExit):
        check_free_space(tmp_path, 0.0, need_bytes=free + 1)
    lin = torch.nn.Linear(4, 4, bias=False)                       # 16 fp32 weights
    assert checkpoint_bytes(lin) == 16 * (4 + 8)
