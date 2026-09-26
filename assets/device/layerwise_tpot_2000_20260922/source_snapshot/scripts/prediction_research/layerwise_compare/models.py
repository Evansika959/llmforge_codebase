"""A small tabular Transformer over the same 110 features as XGBoost."""
import torch
from torch import nn


PREFIXES = ['n_h_', 'n_kv_', 'd_qk_', 'd_v_', 'd_mlp_', 'gqa_ratio_',
            'attention_out_width_', 'kv_bytes_per_token_',
            'layer_matrix_params_', 'layer_q8_bytes_']


def token_groups(names):
    groups = [[i for i, n in enumerate(names) if n.startswith(prefix)] for prefix in PREFIXES]
    used = {i for g in groups for i in g}
    groups = [[i for i in range(len(names)) if i not in used]] + groups
    flat = [i for g in groups for i in g]
    if any(not g for g in groups) or sorted(flat) != list(range(len(names))):
        raise ValueError('Each input feature must belong to exactly one nonempty token')
    return groups


class Transformer(nn.Module):
    def __init__(self, groups, width=64, layers=2, dropout=.1):
        super().__init__()
        self.groups = groups
        self.tokenizers = nn.ModuleList([nn.Linear(len(g), width) for g in groups])
        self.cls = nn.Parameter(torch.zeros(1, 1, width))
        self.position = nn.Parameter(torch.randn(1, len(groups)+1, width)*.02)
        block = nn.TransformerEncoderLayer(width, 4, width*2, dropout,
                  activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 3))

    def forward(self, x):
        tokens = torch.stack([layer(x[:, g]) for layer, g in zip(self.tokenizers, self.groups)], dim=1)
        tokens = torch.cat([self.cls.expand(len(x), -1, -1), tokens], dim=1)+self.position
        return self.head(self.encoder(tokens)[:, 0])
