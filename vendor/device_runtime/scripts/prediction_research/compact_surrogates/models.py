import torch
from torch import nn
from scripts.prediction.features.physics import TOKEN_GROUPS


class TwoHiddenMLP(nn.Module):
    def __init__(self, width=64, dropout=.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(32, width), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(width, width), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(width, 3))

    def forward(self, x):
        return self.net(x)


class CompactTransformer(nn.Module):
    def __init__(self, width=32, layers=1, dropout=.1):
        super().__init__()
        self.tokenizers = nn.ModuleList([nn.Linear(len(g), width) for g in TOKEN_GROUPS])
        self.cls = nn.Parameter(torch.zeros(1, 1, width))
        self.position = nn.Parameter(torch.randn(1, 8, width)*.02)
        block = nn.TransformerEncoderLayer(width, 4, width*2, dropout,
                    activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 3))

    def forward(self, x):
        tokens = torch.stack([layer(x[:, g]) for layer, g in zip(self.tokenizers, TOKEN_GROUPS)], dim=1)
        tokens = torch.cat([self.cls.expand(len(x), -1, -1), tokens], dim=1)+self.position
        return self.head(self.encoder(tokens)[:, 0])


def build(config):
    kwargs = {k: v for k, v in config.items() if k in ('width', 'layers', 'dropout')}
    if config['family'] == 'mlp':
        kwargs.pop('layers', None)
        return TwoHiddenMLP(**kwargs)
    if config['family'] == 'transformer':
        return CompactTransformer(**kwargs)
    raise ValueError('Unknown model family')
