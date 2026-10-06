from __future__ import annotations
import torch
from torch import nn

class TemporalTransformer(nn.Module):
    def __init__(self, d_model: int, nhead: int = 4, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
