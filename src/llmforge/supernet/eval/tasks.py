"""Evaluation tasks: retrieval (the search objective) + guardrail (the preservation constraint).

 * passkey     : one numeric needle in ~n_ctx tokens of filler at a given depth -> long-context copy.
 * multikey    : query one record's value among many distractor records -> retrieval precision.
 * guardrail   : avg next-token NLL on held-out math text -> reasoning/QA preservation.

Greedy generation runs through the elastic forward's KV-cache path (eager backend). Set the
config once with set_elastic_config before calling; all tasks take an already-configured model.
"""
import os, sys, json, random
import torch
import torch.nn.functional as F
from ..paths import RAW, RUNS
from ..paths import ROOT as _ROOT
ROOT = str(_ROOT)
from ..elastic import set_elastic_config

FILLER = "The sky is blue. The grass is green. The sun is bright. Birds fly over the hills. "


def _gen(model, tok, text, max_new=8, dev="cuda"):
    enc = tok(text, return_tensors="pt").to(dev)  # includes attention_mask -> reliable generation
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.eos_token_id, use_cache=True)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def make_passkey(tok, n_ctx, depth, rng):
    key = str(rng.randint(10000, 99999))
    needle = f" The special passkey is {key}. Remember it well. "
    reps = max(1, n_ctx // len(tok(FILLER, add_special_tokens=False).input_ids))
    n_before = int(reps * depth)
    text = (FILLER * n_before) + needle + (FILLER * (reps - n_before)) + \
        "\nQuestion: What is the special passkey?\nAnswer: The special passkey is"
    return text, key


def make_multikey(tok, n_pairs, rng, n_ctx=0):
    pairs = [(rng.randint(100000, 999999), rng.randint(100000, 999999)) for _ in range(n_pairs)]
    records = "".join(f"Record {k}: value {v}. " for k, v in pairs)
    pad = ""
    if n_ctx:
        reps = max(0, (n_ctx - len(tok(records, add_special_tokens=False).input_ids))
                   // max(1, len(tok(FILLER, add_special_tokens=False).input_ids)))
        pad = FILLER * reps
    qi = rng.randrange(n_pairs)
    text = "Here is a list of records.\n" + pad + records + \
        f"\nQuestion: What is the value for Record {pairs[qi][0]}?\nAnswer: value"
    return text, str(pairs[qi][1])


def eval_passkey(model, tok, cfg, order, n_ctx=3500, depths=(0.1, 0.5, 0.9), n=4, seed=0, dev="cuda"):
    set_elastic_config(model, cfg[0], cfg[1], order)
    rng = random.Random(seed)
    hit = tot = 0
    for d in depths:
        for _ in range(n):
            text, key = make_passkey(tok, n_ctx, d, rng)
            hit += key in _gen(model, tok, text, dev=dev)
            tot += 1
    return hit / tot


def eval_multikey(model, tok, cfg, order, n_pairs=30, n_ctx=3000, n=12, seed=1, dev="cuda"):
    set_elastic_config(model, cfg[0], cfg[1], order)
    rng = random.Random(seed)
    hit = tot = 0
    for _ in range(n):
        text, val = make_multikey(tok, n_pairs, rng, n_ctx=n_ctx)
        hit += val in _gen(model, tok, text, dev=dev)
        tot += 1
    return hit / tot


def _load_jsonl_texts(fname, n, max_chars):
    path = os.path.join(str(RAW), fname)
    texts = []
    with open(path) as f:
        for line in f:
            try:
                t = json.loads(line).get("text")
            except Exception:
                continue
            if t and len(t) > 200:
                texts.append(t[:max_chars])
            if len(texts) >= n:
                break
    return texts


def load_guardrail_texts(n=40, max_chars=4000):
    """math NLL proxy texts (openmath). Tracks MATH ability, ~uncorrelated with general LM."""
    return _load_jsonl_texts("openmath.jsonl", n, max_chars)


def load_fineweb_texts(n=40, max_chars=4000):
    """general-web-text NLL proxy texts (fineweb_edu). Tracks lambada/wikitext (corr +0.98);
    DISTINCT from those benchmarks, so it can be a search objective without benchmark-overfitting."""
    return _load_jsonl_texts("fineweb_edu.jsonl", n, max_chars)


@torch.no_grad()
def eval_guardrail_nll(model, tok, cfg, order, texts, max_len=1024, dev="cuda"):
    set_elastic_config(model, cfg[0], cfg[1], order)
    tot_nll = tot_tok = 0
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=max_len).input_ids.to(dev)
        if ids.shape[1] < 2:
            continue
        logits = model(ids).logits
        nll = F.cross_entropy(logits[0, :-1].float(), ids[0, 1:], reduction="sum")
        tot_nll += nll.item(); tot_tok += ids.shape[1] - 1
    return tot_nll / max(1, tot_tok)


@torch.no_grad()
def eval_multikey_nll(model, tok, cfg, order, n_pairs=12, n_ctx=1500, n=15, seed=1, dev="cuda"):
    """Cheap retrieval-sensitive proxy: teacher-forced NLL of the correct answer value given the
    multi-key context (one forward, no generation). Lower = better retrieval. For the DSE search
    loop, if this predicts the real (generated) multikey accuracy it can replace it."""
    set_elastic_config(model, cfg[0], cfg[1], order)
    rng = random.Random(seed)
    tot_nll = tot_tok = 0
    for _ in range(n):
        text, val = make_multikey(tok, n_pairs, rng, n_ctx=n_ctx)  # text ends "...Answer: value"
        ans_ids = tok(" " + val, add_special_tokens=False).input_ids
        k = len(ans_ids)
        ids = tok(text + " " + val, return_tensors="pt").input_ids.to(dev)
        logits = model(ids).logits
        logp = torch.log_softmax(logits[0, -k - 1:-1].float(), dim=-1)
        nll = -logp[range(k), torch.tensor(ans_ids, device=dev)].sum()
        tot_nll += nll.item(); tot_tok += k
    return tot_nll / max(1, tot_tok)


if __name__ == "__main__":
    from .loader import load_supernet
    ck = str(RUNS / "m2_v1" / "final")
    model, tok, order = load_supernet(ck)
    texts = load_guardrail_texts(n=10)
    # Quarter-steps of the loaded model's head_dim, not the Qwen3 numbers 128/64/32 -- on a
    # head_dim=64 model those mislabel full as "mid" and trip qk_gather_index's assert.
    hd = model.config.head_dim
    for name, cfg in [("full", (hd, hd)), ("mid", (hd // 2, hd // 2)),
                      ("min", (hd // 4, hd // 4))]:
        pk = eval_passkey(model, tok, cfg, order, n=3)
        mk = eval_multikey(model, tok, cfg, order, n=6)
        gn = eval_guardrail_nll(model, tok, cfg, order, texts)
        print(f"{name:5s} {cfg}: passkey={pk:.2f} multikey={mk:.2f} guardrail_nll={gn:.3f}")
