"""Soft Dice loss for semantic segmentation."""
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Multi-class soft Dice loss.

    Expects raw logits of shape (B, C, H, W) and integer targets (B, H, W).
    The loss is ``1 - mean Dice`` over classes, computed on softmax
    probabilities so it is differentiable.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        num_classes = logits.shape[1]
        prob = F.softmax(logits, dim=1)
        target_1h = F.one_hot(target.clamp(min=0), num_classes)
        target_1h = target_1h.permute(0, 3, 1, 2).to(prob.dtype)  # (B, C, H, W)

        dims = (0, 2, 3)
        intersection = (prob * target_1h).sum(dims)
        cardinality = prob.sum(dims) + target_1h.sum(dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()
