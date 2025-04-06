# model.py -- DiT
# Adapted https://arxiv.org/pdf/2310.16834 : AdaLN

import torch
import torch.nn as nn

from dataclasses import dataclass

@dataclass
class DiTConfig:
    dim: int = 768
    num_heads: int = 12
    cond_dim: int = 768
    mlp_ratio: float = 4.0
    dropout: float = 0.1

class DitBlock(nn.Module):
    def __init__(self, config: DiTConfig):
        super().__init__()
        self.config = config

        self.num_heads = config.num_heads
        self.layer_norm1 = nn.LayerNorm(config.dim)
        self.attn_qkv = nn.Linear(config.dim, config.dim * 3, bias=False)
        self.attn_out = nn.Linear(config.dim, config.dim)
        self.attn_dropout = nn.Dropout(config.dropout)

        self.layer_norm2 = nn.LayerNorm(config.dim)
        self.mlp_fc1 = nn.Linear(config.dim, config.dim * config.mlp_ratio)
        self.mlp_act = nn.GELU()
        self.mlp_fc2 = nn.Linear(config.dim * config.mlp_ratio, config.dim)
        self.mlp_dropout = nn.Dropout(config.dropout)

        # adaLN-zero time information network
        # ???
        self.adaLN_modulation = nn.Linear(config.cond_dim, config.dim * 6, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x, cond):
        pass
        
        
        
