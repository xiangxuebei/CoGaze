from __future__ import annotations

from typing import Dict, List

try:
    from typing import Literal
except ImportError:  # pragma: no cover - Python <3.8 fallback
    from typing_extensions import Literal

import torch
import torch.nn as nn


class ScaleMLP(nn.Module):
    def __init__(self, in_ch: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_ch),
            nn.Linear(in_ch, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.net(x))


class MultiScaleFaceTokenizer(nn.Module):
    def __init__(self, token_dim: int, dropout: float = 0.1, face_scales: int = 4):
        super().__init__()
        if face_scales < 1 or face_scales > 4:
            raise ValueError(f"face_scales must be in [1, 4], got {face_scales}")
        self.face_scales = int(face_scales)
        self.scale_order = ["s1", "s2", "s3", "s4"]
        # Single-scale ablation uses the most semantic face map.
        if self.face_scales == 1:
            self.enabled_scales = ["s3"]
        else:
            self.enabled_scales = self.scale_order[: self.face_scales]

        self.scale_mlps = nn.ModuleDict(
            {
                "s1": ScaleMLP(64, token_dim, dropout),
                "s2": ScaleMLP(128, token_dim, dropout),
                "s3": ScaleMLP(256, token_dim, dropout),
                "s4": ScaleMLP(512, token_dim, dropout),
            }
        )
        self.scale_embed = nn.Embedding(4, token_dim)
        self.norm = nn.LayerNorm(token_dim)

    @staticmethod
    def flatten_map(feat: torch.Tensor) -> torch.Tensor:
        b, c, h, w = feat.shape
        return feat.view(b, c, h * w).transpose(1, 2).contiguous()

    def forward(self, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        tokens: List[torch.Tensor] = []
        for i, key in enumerate(self.enabled_scales):
            x = self.flatten_map(feats[key])
            x = self.scale_mlps[key](x)
            scale_idx = self.scale_order.index(key)
            x = x + self.scale_embed.weight[scale_idx].view(1, 1, -1)
            tokens.append(x)
        return self.norm(torch.cat(tokens, dim=1))


class EyeTokenizer(nn.Module):
    def __init__(self, token_dim: int, use_layer: int = 3, dropout: float = 0.1):
        super().__init__()
        self.use_layer = use_layer
        in_ch = {1: 64, 2: 128, 3: 256, 4: 512}[use_layer]
        self.mlp = ScaleMLP(in_ch, token_dim, dropout)

    def forward(self, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        key = {1: "s1", 2: "s2", 3: "s3", 4: "s4"}[self.use_layer]
        feat = feats[key]
        b, c, h, w = feat.shape
        x = feat.view(b, c, h * w).transpose(1, 2).contiguous()
        return self.mlp(x)


def fuse_left_right_eye_tokens(
    left: torch.Tensor,
    right: torch.Tensor,
    mode: Literal["mean", "concat"] = "mean",
) -> torch.Tensor:
    if mode == "mean":
        return 0.5 * (left + right)
    if mode == "concat":
        return torch.cat([left, right], dim=1)
    raise ValueError(f"Unknown fuse mode: {mode}")
