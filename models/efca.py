from __future__ import annotations

import torch
import torch.nn as nn


class CrossAttentionBlock(nn.Module):
    def __init__(self, token_dim: int, num_heads: int = 4, dropout: float = 0.1, ff_mult: int = 4):
        super().__init__()
        self.norm_q = nn.LayerNorm(token_dim)
        self.norm_kv = nn.LayerNorm(token_dim)
        self.attn = nn.MultiheadAttention(token_dim, num_heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim * ff_mult),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(token_dim * ff_mult, token_dim),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        qn = self.norm_q(q)
        kvn = self.norm_kv(kv)
        out, _ = self.attn(qn, kvn, kvn, need_weights=False)
        out = q + self.drop(out)
        out = out + self.drop(self.ff(out))
        return out


class EyeFaceCrossAttention(nn.Module):
    def __init__(
        self,
        token_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        ff_mult: int = 4,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.eye_to_face = CrossAttentionBlock(token_dim, num_heads, dropout, ff_mult)
        self.face_to_eye = CrossAttentionBlock(token_dim, num_heads, dropout, ff_mult)

    def forward(self, eye_prev: torch.Tensor, eye_curr: torch.Tensor, face_curr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        face_hat = self.eye_to_face(eye_prev, face_curr)
        if self.bidirectional:
            eye_hat = self.face_to_eye(face_hat, eye_curr)
        else:
            eye_hat = eye_curr
        return face_hat, eye_hat
