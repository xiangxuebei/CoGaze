from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    from typing import Literal
except ImportError:  # pragma: no cover - Python <3.8 fallback
    from typing_extensions import Literal


TSMType = Literal["transformer", "lstm"]


@dataclass
class CoGazeConfig:
    # sequence
    max_seq_len: int = 120

    # token dims
    token_dim: int = 256

    # face branch (Msc ablation: set face_scales=1)
    face_scales: int = 4
    face_dropout: float = 0.1

    # eye branch / prior dictionary (Pd ablation)
    eye_use_layer: int = 3
    eye_fuse: Literal["mean", "concat"] = "mean"
    use_ocular_prior: bool = True

    # ocular prototype-anchor dictionary (Pd)
    dict_size: int = 64
    dict_heads: int = 4
    dict_tau: float = 0.07
    dict_dropout: float = 0.1

    # reciprocal eye-face cross-attention (v ablation)
    efca_heads: int = 4
    efca_dropout: float = 0.1
    efca_ff_mult: int = 4
    use_bidirectional_efca: bool = True

    # attention difference map (ADM)
    adm_hidden: int = 128
    adm_dropout: float = 0.1

    # temporal sequence module (TSM)
    tsm_type: TSMType = "transformer"
    tsm_hidden: int = 512
    tsm_layers: int = 4
    tsm_heads: int = 8
    tsm_dropout: float = 0.1

    # prediction head
    head_hidden: int = 256
    head_dropout: float = 0.1

    # loss
    lambda_cons: float = 0.5
    smooth_l1_beta: float = 1.0

    # backbone
    backbone: Literal["resnet18"] = "resnet18"
    gazeclr_weights_path: Optional[str] = None
    torchvision_pretrained: bool = False
