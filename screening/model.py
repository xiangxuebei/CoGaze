from __future__ import annotations

import torch
import torch.nn as nn

from cogaze.screening.modules import AttnMILPool, FeatureExpert, TaskPhaseGating, ZeroExpert


class CognitiveScreeningMILModel(nn.Module):
    def __init__(
        self,
        gaze_dim: int,
        audio_dim: int,
        interaction_dim: int,
        num_tasks: int,
        num_phases: int,
        num_classes: int = 2,
        embed_dim: int = 64,
        task_embed_dim: int = 16,
        phase_embed_dim: int = 8,
        expert_hidden_dim: int = 128,
        gating_hidden_dim: int = 64,
        attn_dim: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        self.gaze_expert = FeatureExpert(gaze_dim, expert_hidden_dim, embed_dim, dropout) if gaze_dim > 0 else ZeroExpert(embed_dim)
        self.audio_expert = FeatureExpert(audio_dim, expert_hidden_dim, embed_dim, dropout) if audio_dim > 0 else ZeroExpert(embed_dim)
        self.inter_expert = FeatureExpert(interaction_dim, expert_hidden_dim, embed_dim, dropout) if interaction_dim > 0 else ZeroExpert(embed_dim)

        self.task_emb = nn.Embedding(num_tasks, task_embed_dim)
        self.phase_emb = nn.Embedding(num_phases, phase_embed_dim)
        self.gating = TaskPhaseGating(embed_dim, task_embed_dim, phase_embed_dim, num_modalities=3, hidden_dim=gating_hidden_dim)
        self.pool = AttnMILPool(embed_dim, attn_dim)

        self.post = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.cls_head = nn.Linear(embed_dim, num_classes)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        gaze = batch["gaze"]
        audio = batch["audio"]
        inter = batch["interaction"]
        avail = batch["avail"]
        task_id = batch["task_id"]
        phase_id = batch["phase_id"]
        attn_mask = batch["attn_mask"]

        gaze_e = self.gaze_expert(gaze)
        audio_e = self.audio_expert(audio)
        inter_e = self.inter_expert(inter)

        task_e = self.task_emb(task_id)
        phase_e = self.phase_emb(phase_id)

        weights = self.gating([gaze_e, audio_e, inter_e], task_e, phase_e, avail)
        h = (
            weights[..., 0:1] * gaze_e
            + weights[..., 1:2] * audio_e
            + weights[..., 2:3] * inter_e
        )

        z, alpha = self.pool(h, attn_mask)
        z = self.post(z)
        logits = self.cls_head(z)

        return {
            "logits": logits,
            "subject_emb": z,
            "task_attn": alpha,
            "modality_weights": weights,
        }
