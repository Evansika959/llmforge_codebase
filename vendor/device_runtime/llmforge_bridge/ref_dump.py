"""
Reference greedy decode from a LLMForge checkpoint, on CPU in fp32.
Emits the SAME sequence the patched run.c `-I ids -O` path prints:
  prompt[1:] (the forced prompt tokens) followed by greedy-generated tokens.
Used to validate llama2.c numeric parity. See doc/llmforge2c.md Component D.

Usage:
  python ref_dump.py <ckpt_dir> <ids_file> <steps>
"""
import sys, os, contextlib
import torch

RF = "$HOME/LLMForge"
sys.path.insert(0, RF)
from model import GPT
from gpt_conf import GPTConfig
from dataclasses import fields

def main():
    ckpt_dir, ids_file, steps = sys.argv[1], sys.argv[2], int(sys.argv[3])
    ck = torch.load(os.path.join(ckpt_dir, "ckpt.pt"), map_location="cpu", weights_only=False)
    sd = { (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
           for k, v in ck["model"].items() }
    ma = ck["model_args"]
    valid = {f.name for f in fields(GPTConfig)}
    cfg = GPTConfig(**{k: v for k, v in ma.items() if k in valid})
    with contextlib.redirect_stdout(sys.stderr):   # silence model-construction tables
        model = GPT(cfg)
        model.load_state_dict(sd, strict=True)
        model.eval().float()

    with open(ids_file) as f:
        prompt = [int(x) for x in f.read().split()]
    n = len(prompt)
    printed = list(prompt[1:n])               # forced prompt tokens (mirror run.c)
    cur = list(prompt)
    with torch.no_grad():
        while len(printed) < steps:
            idx = torch.tensor([cur], dtype=torch.long)
            out = model(idx)
            logits = out[0] if isinstance(out, tuple) else out
            logits = logits[:, -1, :]          # last position
            nxt = int(torch.argmax(logits, dim=-1).item())
            printed.append(nxt)
            cur.append(nxt)
    for t in printed[:steps]:
        print(t)

if __name__ == "__main__":
    main()
