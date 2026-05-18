"""Focal loss for class-imbalanced semantic segmentation (gamma = 2.0).

Implemented in Step 3.
"""
from __future__ import annotations

import torch.nn as nn


class FocalLoss(nn.Module):
    """Multi-class focal loss with optional per-class weighting.

    Implemented in Step 3.
    """

    def __init__(self, gamma: float = 2.0, class_weights=None):
        super().__init__()
        raise NotImplementedError("Focal loss is implemented in Step 3.")

    def forward(self, logits, target):
        raise NotImplementedError("Implemented in Step 3.")
