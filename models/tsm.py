from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class LSTMTSM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.out_dim = hidden_dim

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        y, _ = self.rnn(x)
        return y


class TransformerTSM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, layers: int = 4, heads: int = 8, dropout: float = 0.1, max_len: int = 256):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.pos = nn.Embedding(max_len, hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.out_dim = hidden_dim

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, t, _ = x.shape
        pos = torch.arange(t, device=x.device).unsqueeze(0).expand(b, t)
        h = self.in_proj(x) + self.pos(pos)
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        return self.norm(h)
