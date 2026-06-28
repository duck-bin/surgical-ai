"""Focal loss for class-imbalanced semantic segmentation (gamma = 2.0)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class focal loss with optional per-class weighting.

    Expects raw logits of shape (B, C, H, W) and integer targets (B, H, W).
    The focal modulation ``(1 - p_t) ** gamma`` down-weights well-classified
    pixels; ``class_weights`` additionally up-weights rare classes.
    """

    def __init__(self, gamma: float = 2.0, class_weights=None,
                 ignore_index: int = -100):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        if class_weights is not None:
            # persistent=False: class weights are recomputed from the data every
            # run (not learned), so they must NOT live in the checkpoint -- else
            # a run with weighting can't resume one without it (or vice versa),
            # which fails strict state_dict loading on "focal.class_weights".
            self.register_buffer(
                "class_weights",
                torch.as_tensor(class_weights, dtype=torch.float32),
                persistent=False,
            )
        else:
            self.class_weights = None

    def forward(self, logits, target):
        log_prob = F.log_softmax(logits, dim=1)
        target_safe = target.clamp(min=0)
        log_pt = log_prob.gather(1, target_safe.unsqueeze(1)).squeeze(1)
        focal = -((1.0 - log_pt.exp()) ** self.gamma) * log_pt  # (B, H, W)

        if self.class_weights is not None:
            focal = focal * self.class_weights[target_safe]

        valid = target != self.ignore_index
        focal = focal * valid
        return focal.sum() / valid.sum().clamp(min=1)
