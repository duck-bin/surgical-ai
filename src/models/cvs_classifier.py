"""ViT-based CVS classifier for Strasberg's 3 binary criteria.

Input: a 9-channel tensor (6-channel segmentation mask + 3-channel RGB frame).
Backbone: timm ``vit_small_patch16_224`` (ImageNet pretrained), with the patch
embedding adapted to 9 input channels. Three independent binary heads emit one
sigmoid logit per Strasberg criterion; the CVS score is their sum (0-3).
Implemented in Step 7.
"""
from __future__ import annotations

import torch.nn as nn


class CVSClassifier(nn.Module):
    """ViT-Small backbone with 3 independent binary criterion heads.

    Implemented in Step 7.
    """

    def __init__(self, backbone: str = "vit_small_patch16_224",
                 in_channels: int = 9, num_criteria: int = 3):
        super().__init__()
        raise NotImplementedError("CVS classifier is implemented in Step 7.")

    def forward(self, x):
        """Return 3 binary criterion logits per sample."""
        raise NotImplementedError("Implemented in Step 7.")
