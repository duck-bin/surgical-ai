"""EfficientNet-B4 U-Net baseline for semantic segmentation.

Standard semantic segmentation baseline built on ``segmentation_models_pytorch``
(EfficientNet-B4 encoder, ImageNet weights). This baseline is the credibility
anchor for the paper — SAM2 results are never reported without it.
Implemented in Step 3.
"""
from __future__ import annotations

import torch.nn as nn


class UNetBaseline(nn.Module):
    """EfficientNet-B4 U-Net producing 6-class segmentation logits.

    Implemented in Step 3.
    """

    def __init__(self, encoder: str = "efficientnet-b4", num_classes: int = 6):
        super().__init__()
        raise NotImplementedError("U-Net baseline is implemented in Step 3.")

    def forward(self, x):
        raise NotImplementedError("Implemented in Step 3.")
