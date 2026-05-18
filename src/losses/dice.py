"""Soft Dice loss for semantic segmentation.

Implemented in Step 3.
"""
from __future__ import annotations

import torch.nn as nn


class DiceLoss(nn.Module):
    """Multi-class soft Dice loss.

    Implemented in Step 3.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        raise NotImplementedError("Dice loss is implemented in Step 3.")

    def forward(self, logits, target):
        raise NotImplementedError("Implemented in Step 3.")
