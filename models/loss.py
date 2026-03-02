from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CoGazeLossOutput:
    total: torch.Tensor
    reg: torch.Tensor
    cons: torch.Tensor


class CoGazeLoss(nn.Module):
    def __init__(self, lambda_cons: float = 0.5, smooth_l1_beta: float = 1.0):
        super().__init__()
        self.lambda_cons = lambda_cons
        self.beta = smooth_l1_beta

    def forward(
        self,
        pred_pog_cm: torch.Tensor,
        gt_pog_cm: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        segment_id: Optional[torch.Tensor] = None,
    ) -> CoGazeLossOutput:
        b, t, _ = pred_pog_cm.shape
        device = pred_pog_cm.device
        if valid_mask is None:
            valid_mask = torch.ones((b, t), dtype=torch.bool, device=device)
        else:
            valid_mask = valid_mask.bool().to(device)

        reg = F.smooth_l1_loss(pred_pog_cm, gt_pog_cm.to(device), beta=self.beta, reduction="none").sum(-1)
        reg = (reg * valid_mask.float()).sum() / valid_mask.float().sum().clamp_min(1.0)

        if t <= 1 or self.lambda_cons <= 0:
            cons = pred_pog_cm.new_zeros(())
        else:
            diff = torch.abs(pred_pog_cm[:, 1:] - pred_pog_cm[:, :-1]).sum(-1)
            pair_mask = valid_mask[:, 1:] & valid_mask[:, :-1]
            if segment_id is not None:
                seg = segment_id.to(device)
                pair_mask = pair_mask & (seg[:, 1:] == seg[:, :-1])
            cons = (diff * pair_mask.float()).sum() / pair_mask.float().sum().clamp_min(1.0)

        total = reg + self.lambda_cons * cons
        return CoGazeLossOutput(total=total, reg=reg, cons=cons)
