from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeAnchorDictionary(nn.Module):
    def __init__(self, dict_size: int, token_dim: int):
        super().__init__()
        self.content_proto = nn.Parameter(torch.randn(dict_size, token_dim) * 0.02)
        self.anchor_proto = nn.Parameter(torch.randn(dict_size, token_dim) * 0.02)
        self.norm = nn.LayerNorm(token_dim)

    def tokens(self) -> torch.Tensor:
        return self.norm(self.content_proto + self.anchor_proto)


class CosineCrossAttention(nn.Module):
    def __init__(self, token_dim: int, num_heads: int = 4, tau: float = 0.07, dropout: float = 0.1):
        super().__init__()
        assert token_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = token_dim // num_heads
        self.tau = tau

        self.q_proj = nn.Linear(token_dim, token_dim)
        self.k_proj = nn.Linear(token_dim, token_dim)
        self.v_proj = nn.Linear(token_dim, token_dim)
        self.out_proj = nn.Linear(token_dim, token_dim)
        self.dropout = nn.Dropout(dropout)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, l, d = x.shape
        return x.view(b, l, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        b, h, l, dh = x.shape
        return x.transpose(1, 2).reshape(b, l, h * dh)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        qh = F.normalize(self._split(self.q_proj(q)), p=2, dim=-1)
        kh = F.normalize(self._split(self.k_proj(k)), p=2, dim=-1)
        vh = self._split(self.v_proj(v))

        attn = torch.matmul(qh, kh.transpose(-2, -1)) / max(self.tau, 1e-6)
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, vh)
        return self.out_proj(self._merge(out))


class OcularPriorEnhancer(nn.Module):
    def __init__(self, dict_size: int, token_dim: int, num_heads: int = 4, tau: float = 0.07, dropout: float = 0.1):
        super().__init__()
        self.dictionary = PrototypeAnchorDictionary(dict_size, token_dim)
        self.attn = CosineCrossAttention(token_dim, num_heads=num_heads, tau=tau, dropout=dropout)
        self.norm = nn.LayerNorm(token_dim)

    def forward(self, eye_tokens: torch.Tensor) -> torch.Tensor:
        d = self.dictionary.tokens().unsqueeze(0)
        eye_enh = self.attn(eye_tokens, d, d)
        return self.norm(eye_tokens + eye_enh)
