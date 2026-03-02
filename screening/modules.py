from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExpert(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ZeroExpert(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape[:-1] + (self.out_dim,)
        return x.new_zeros(shape)


class TaskPhaseGating(nn.Module):
    def __init__(self, embed_dim: int, task_embed_dim: int, phase_embed_dim: int, num_modalities: int, hidden_dim: int = 64):
        super().__init__()
        self.num_modalities = num_modalities
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * num_modalities + task_embed_dim + phase_embed_dim + num_modalities, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_modalities),
        )

    def forward(self, modality_embeds: list[torch.Tensor], task_e: torch.Tensor, phase_e: torch.Tensor, avail: torch.Tensor) -> torch.Tensor:
        x = torch.cat(modality_embeds + [task_e, phase_e, avail], dim=-1)
        logits = self.mlp(x)
        mask = avail > 0.5
        logits = logits.masked_fill(~mask, -1e9)
        return F.softmax(logits, dim=-1)


class AttnMILPool(nn.Module):
    def __init__(self, embed_dim: int, attn_dim: int = 64):
        super().__init__()
        self.v = nn.Linear(embed_dim, attn_dim)
        self.w = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, h: torch.Tensor, attn_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        u = torch.tanh(self.v(h))
        scores = self.w(u).squeeze(-1)
        scores = scores.masked_fill(attn_mask <= 0, -1e9)
        alpha = F.softmax(scores, dim=-1)
        z = torch.bmm(alpha.unsqueeze(1), h).squeeze(1)
        return z, alpha
