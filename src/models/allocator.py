from __future__ import annotations
import torch
from torch import nn

class Allocator(nn.Module):
    def __init__(self, in_features: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_features, hidden), nn.GELU(), nn.Linear(hidden, 1), nn.Tanh())
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.net(x).squeeze(-1)
        return raw / raw.abs().sum(dim=-1, keepdim=True).clamp_min(1e-8)
