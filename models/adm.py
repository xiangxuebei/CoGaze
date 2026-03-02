from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


class AttentionDifferenceMap(nn.Module):
    def __init__(self, joint_dim: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(joint_dim)
        self.score = nn.Sequential(
            nn.Conv1d(joint_dim, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, 1, kernel_size=1, bias=True),
        )

    def forward(self, h_prev: torch.Tensor, h_curr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        delta = h_curr - h_prev
        d = self.norm(delta).transpose(1, 2).contiguous()
        att = torch.sigmoid(self.score(d)).transpose(1, 2).contiguous()
        h_bar = att * h_curr
        return att, h_bar


class ADMSpatiotemporalBuilder(nn.Module):
    def __init__(self, joint_dim: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.adm = AttentionDifferenceMap(joint_dim, hidden=hidden, dropout=dropout)

    def forward(self, h_seq: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        b, t, l, d = h_seq.shape

        h_bar_seq = []
        att_seq = []

        h_prev = h_seq[:, 0]
        att0, h_bar0 = self.adm(h_prev, h_prev)
        h_bar_seq.append(h_bar0)
        att_seq.append(att0)

        for i in range(1, t):
            h_curr = h_seq[:, i]
            att, h_bar = self.adm(h_prev, h_curr)
            h_bar_seq.append(h_bar)
            att_seq.append(att)
            h_prev = h_curr

        h_bar = torch.stack(h_bar_seq, dim=1)
        att = torch.stack(att_seq, dim=1)

        s_tokens = []
        for i in range(t):
            h_bar_prev = h_bar[:, max(i - 1, 0)]
            h_prev = h_seq[:, max(i - 1, 0)]
            h_curr = h_seq[:, i]
            h_bar_curr = h_bar[:, i]
            delta = h_curr - h_prev
            s_t = torch.cat([h_bar_prev, delta, h_bar_curr], dim=-1)
            s_tokens.append(s_t)

        s_tokens = torch.stack(s_tokens, dim=1)
        s_vec = s_tokens.mean(dim=2)
        aux = {"adm_att": att, "h_bar": h_bar, "s_tokens": s_tokens}
        return s_vec, aux
