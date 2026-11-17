"""The search's held-out scoring protocol, as a reusable input builder.

The search scores a candidate on held-out documents [skip, skip + n), each truncated to its first
`max_len` tokens, dropping documents shorter than 8 tokens, and reports the token-weighted mean
cross-entropy (llmforge.supernet.search.nsga.make_evaluator). Final fronts are re-scored on
documents 32 to 95. Scoring a trained reference model on any other windowing measures a different
benchmark, so reference models that are compared against slice scores use these inputs.
"""
from __future__ import annotations


def search_protocol_ids(tok, texts, max_len: int = 1024, dev: str = "cuda"):
    """One tensor of input ids per document, built exactly as make_evaluator builds them."""
    ids = [tok(t, return_tensors="pt", truncation=True, max_length=max_len).input_ids.to(dev)
           for t in texts]
    return [i for i in ids if i.shape[1] >= 8]


META = "__model_sha256__"   # reserved key in a units file: {name: sha256 of the weights scored}


def file_sha256(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def ckpt_sha256(path: str) -> str:
    """sha256 of a weights file, or of a checkpoint directory's *.safetensors (one shard: its own hash)."""
    import glob
    import hashlib
    import os
    if os.path.isfile(path):
        return file_sha256(path)
    shards = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"no *.safetensors in {path}")
    if len(shards) == 1:
        return file_sha256(shards[0])
    return hashlib.sha256("".join(f"{os.path.basename(s)} {file_sha256(s)}\n"
                                  for s in shards).encode()).hexdigest()


def base_sha256(repo: str) -> str:
    """sha256 of the published checkpoint's weights in the local HF cache."""
    from huggingface_hub import try_to_load_from_cache
    p = try_to_load_from_cache(repo, "model.safetensors")
    if not isinstance(p, str):
        raise FileNotFoundError(f"{repo}/model.safetensors not in the local cache")
    return file_sha256(p)


def cfg_sha256(A: dict):
    """sha256 of an architecture's widths, or None for the unsliced checkpoint."""
    import hashlib
    import json
    keys = ("d_qk", "d_v", "n_h", "d_mlp", "n_kv")
    if not any(k in A for k in keys):
        return None
    return hashlib.sha256(json.dumps({k: A.get(k) for k in keys}, sort_keys=True).encode()).hexdigest()


def needs_score(res: dict, name: str, sha: str, cfg=None, force: bool = False) -> bool:
    """Score unless `name` is stored from exactly these weights and this config."""
    e = res.get(name)
    return force or e is None or e.get("model_sha256") != sha or e.get("cfg_sha256") != cfg


def _write_json_atomic(path: str, obj, indent=None) -> None:
    import json
    import os
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=indent)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save_scores(out: str, res: dict, units: dict, shas: dict | None = None) -> None:
    """Merge scores into the summary file `out` and per-document units into `<out>_units.json`.

    Both files are merged with what is already on disk: entries passed here replace entries of the
    same name, and every other entry is kept. `shas` ({name: weights sha256}) is recorded under the
    reserved units key META. A merge is refused when the new per-document token counts differ from
    those already in the file, since that means a different document range or tokenisation. Both
    files are written atomically.
    """
    import json
    import os
    upath = out.replace(".json", "_units.json")
    old_units = json.load(open(upath)) if os.path.exists(upath) else {}
    old_res = json.load(open(out)) if os.path.exists(out) else {}
    ref = next(([c for _, c in v] for k, v in old_units.items() if k != META), None)
    for k, v in units.items():
        cnt = [c for _, c in v]
        if ref is None:
            ref = cnt
        elif cnt != ref:
            raise ValueError(f"{out}: per-document token counts of {k} differ from the file's "
                             f"({len(cnt)} vs {len(ref)} documents); refusing to mix ranges")
    for k, v in res.items():
        d = v.get("docs")
        if d is not None and any(e.get("docs") not in (None, d) for e in old_res.values()):
            raise ValueError(f"{out}: {k} scored on documents {d}, the file holds another range")
    merged_res = {**old_res, **res}
    merged_units = {**old_units, **units}
    if shas:
        merged_units[META] = {**old_units.get(META, {}), **shas}
    _write_json_atomic(upath, merged_units)
    _write_json_atomic(out, merged_res, indent=1)


def load_base_elastic(spec, dev: str = "cuda", dtype=None, attn: str = "eager"):
    """The published checkpoint as an elastic model with the supernet's parameterisation.

    This is the starting point of groundtruth_het.py --init base --base-align, so slicing it with an
    architecture's config gives the "base slice" that base-initialised training starts from.
    """
    import torch
    from transformers import AutoModelForCausalLM

    from ..elastic import enable_elastic, set_elastic_backend
    from ..train.base_init import attach_supernet_params

    m = AutoModelForCausalLM.from_pretrained(
        spec.repo, torch_dtype=dtype or torch.bfloat16, attn_implementation=attn).to(dev).eval()
    enable_elastic(m)
    attach_supernet_params(m)
    set_elastic_backend(attn)
    m.config.use_cache = False
    return m
