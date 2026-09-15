"""Neural model definitions only; optimization belongs to training/."""
import torch
from torch import nn
from ..config import TARGETS
from ..features.physics import TOKEN_GROUPS

class TinyTransformer(nn.Module):
    def __init__(self, n_features, width=32, dropout=0.1):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_features, width) * 0.02)
        self.bias = nn.Parameter(torch.randn(n_features, width) * 0.02)
        self.cls = nn.Parameter(torch.zeros(1, 1, width))
        layer = nn.TransformerEncoderLayer(width, 4, width*2, dropout,
                                           activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, len(TARGETS)))

    def forward(self, x):
        tokens = x[:, :, None] * self.weight + self.bias
        tokens = torch.cat([self.cls.expand(len(x), -1, -1), tokens], dim=1)
        return self.head(self.encoder(tokens)[:, 0])
class GroupedTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizers=nn.ModuleList([nn.Linear(len(g),128) for g in TOKEN_GROUPS])
        self.cls=nn.Parameter(torch.zeros(1,1,128));self.position=nn.Parameter(torch.randn(1,8,128)*.02)
        layer=nn.TransformerEncoderLayer(128,8,512,.1,activation='gelu',batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,4,enable_nested_tensor=False)
        self.head=nn.Sequential(nn.LayerNorm(128),nn.Linear(128,64),nn.GELU(),nn.Linear(64,3))

    def forward(self,x):
        tokens=torch.stack([layer(x[:,g]) for layer,g in zip(self.tokenizers,TOKEN_GROUPS)],dim=1)
        tokens=torch.cat([self.cls.expand(len(x),-1,-1),tokens],dim=1)+self.position
        return self.head(self.encoder(tokens)[:,0])


class ResidualMLP(nn.Module):
    def __init__(self):
        super().__init__();self.input=nn.Linear(32,128)
        self.blocks=nn.ModuleList([nn.Sequential(nn.LayerNorm(128),nn.Linear(128,128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,128)) for _ in range(2)])
        self.head=nn.Sequential(nn.LayerNorm(128),nn.Linear(128,3))

    def forward(self,x):
        x=self.input(x)
        for block in self.blocks:x=x+block(x)
        return self.head(x)
