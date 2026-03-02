from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from cogaze.config import CoGazeConfig
from cogaze.models.adm import ADMSpatiotemporalBuilder
from cogaze.models.backbone import ResNet18MultiScale
from cogaze.models.efca import EyeFaceCrossAttention
from cogaze.models.head import PoRRegressor
from cogaze.models.icp_tokens import EyeTokenizer, MultiScaleFaceTokenizer, fuse_left_right_eye_tokens
from cogaze.models.ocular_dictionary import OcularPriorEnhancer
from cogaze.models.tsm import LSTMTSM, TransformerTSM


class CoGaze(nn.Module):
    def __init__(self, cfg: CoGazeConfig):
        super().__init__()
        self.cfg = cfg

        self.face_backbone = ResNet18MultiScale(cfg.gazeclr_weights_path)
        self.eye_backbone = ResNet18MultiScale(cfg.gazeclr_weights_path)

        self.face_tokenizer = MultiScaleFaceTokenizer(cfg.token_dim, cfg.face_dropout, face_scales=cfg.face_scales)
        self.eye_tokenizer = EyeTokenizer(cfg.token_dim, cfg.eye_use_layer, cfg.dict_dropout)
        self.ocular_prior = OcularPriorEnhancer(cfg.dict_size, cfg.token_dim, cfg.dict_heads, cfg.dict_tau, cfg.dict_dropout)
        self.efca = EyeFaceCrossAttention(
            cfg.token_dim,
            cfg.efca_heads,
            cfg.efca_dropout,
            cfg.efca_ff_mult,
            bidirectional=cfg.use_bidirectional_efca,
        )

        joint_dim = cfg.token_dim * 2
        self.adm_builder = ADMSpatiotemporalBuilder(joint_dim, cfg.adm_hidden, cfg.adm_dropout)

        st_dim = joint_dim * 3
        if cfg.tsm_type == "lstm":
            self.tsm = LSTMTSM(st_dim, cfg.tsm_hidden, cfg.tsm_layers, cfg.tsm_dropout)
        else:
            self.tsm = TransformerTSM(st_dim, cfg.tsm_hidden, cfg.tsm_layers, cfg.tsm_heads, cfg.tsm_dropout, cfg.max_seq_len)

        self.head = PoRRegressor(self.tsm.out_dim, cfg.head_hidden, cfg.head_dropout)

    def _encode_face(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        feats = self.face_backbone(x.view(b * t, c, h, w))
        tokens = self.face_tokenizer(feats)
        return tokens.view(b, t, tokens.shape[1], tokens.shape[2])

    def _encode_eye(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = left.shape
        left_feats = self.eye_backbone(left.view(b * t, c, h, w))
        right_feats = self.eye_backbone(right.view(b * t, c, h, w))
        left_tokens = self.eye_tokenizer(left_feats)
        right_tokens = self.eye_tokenizer(right_feats)
        eye_tokens = fuse_left_right_eye_tokens(left_tokens, right_tokens, mode=self.cfg.eye_fuse)
        return eye_tokens.view(b, t, eye_tokens.shape[1], eye_tokens.shape[2])

    @staticmethod
    def _prev_shift(x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x[:, 0:1], x[:, :-1]], dim=1)

    def forward(self, batch: Dict[str, torch.Tensor], return_aux: bool = False) -> Dict[str, torch.Tensor]:
        face = batch["face_rgb"]
        left = batch["left_eye_rgb"]
        right = batch["right_eye_rgb"]

        valid = batch.get("valid_mask", None)
        key_padding_mask = (~valid.bool()).to(face.device) if valid is not None else None

        face_tokens = self._encode_face(face)
        eye_tokens = self._encode_eye(left, right)

        b, t, l_eye, d = eye_tokens.shape
        if self.cfg.use_ocular_prior:
            eye_enh = self.ocular_prior(eye_tokens.view(b * t, l_eye, d)).view(b, t, l_eye, d)
        else:
            eye_enh = eye_tokens
        eye_prev = self._prev_shift(eye_enh)

        face_hat, eye_hat = self.efca(
            eye_prev.view(b * t, l_eye, d),
            eye_enh.view(b * t, l_eye, d),
            face_tokens.view(b * t, face_tokens.shape[2], d),
        )

        h_joint = torch.cat([face_hat, eye_hat], dim=-1).view(b, t, l_eye, d * 2)
        s_seq, aux = self.adm_builder(h_joint)
        z_seq = self.tsm(s_seq, key_padding_mask=key_padding_mask)
        pred_pog_cm = self.head(z_seq)

        out = {"pred_pog_cm": pred_pog_cm}
        if return_aux:
            out.update({
                "face_tokens": face_tokens,
                "eye_tokens": eye_tokens,
                "eye_tokens_enh": eye_enh,
                "h_joint": h_joint,
                "s_seq": s_seq,
                **aux,
            })
        return out
