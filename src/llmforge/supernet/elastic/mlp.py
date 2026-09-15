"""Elastic MLP width (`d_mlp`) — MatFormer-style nested prefixes.

Split out of attention.py, which had grown to cover both sublayers. The two are patched together
by `patch.py`, which owns the model-level enable/disable/select API.
"""
import torch.nn.functional as F


def elastic_mlp_forward(self, x):
    """MatFormer-style nested prefix: first `mlp_active` rows of gate/up, columns of down.

    Weight slices are views, so no copy and no new parameters -- the full MLP is retained and a
    narrower one is a prefix of it.
    """
    m = getattr(self, "mlp_active", None)
    if m is None or m >= self.gate_proj.weight.shape[0]:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    g = F.linear(x, self.gate_proj.weight[:m])
    u = F.linear(x, self.up_proj.weight[:m])
    return F.linear(self.act_fn(g) * u, self.down_proj.weight[:, :m])
