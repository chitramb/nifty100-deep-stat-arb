from __future__ import annotations
import torch
from torch import nn

class ResidualCNN(nn.Module):
    def __init__(self, channels: int = 1, filters: int = 8, kernel_size: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, filters, kernel_size=kernel_size),
            nn.GELU(),
            nn.Conv1d(filters, filters, kernel_size=kernel_size),
            nn.GELU(),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
