"""ViT-based CVS classifier for Strasberg's 3 binary criteria.

Input: a 9-channel tensor (6-channel segmentation softmax map + 3-channel RGB
frame). Backbone: timm ``vit_small_patch16_224`` (ImageNet pretrained), with the
patch embedding adapted to 9 input channels. Three independent binary heads
emit one logit per Strasberg criterion; the CVS score is their sum (0-3).
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn


class CVSClassifier(nn.Module):
    """ViT-Small backbone with 3 independent binary criterion heads.

    Args:
        backbone: timm model name (a ViT variant).
        in_channels: number of input channels (6 segmentation + 3 RGB).
        num_criteria: number of binary CVS criteria (Strasberg: 3).
        pretrained: load ImageNet-pretrained backbone weights (needs network).
            ``False`` randomly initialises the backbone (offline tests).
    """

    def __init__(self, backbone: str = "vit_small_patch16_224",
                 in_channels: int = 9, num_criteria: int = 3,
                 pretrained: bool = True):
        super().__init__()
        self.num_criteria = num_criteria
        # num_classes=0 -> the backbone returns pooled features.
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, in_chans=in_channels,
            num_classes=0,
        )
        feature_dim = self.backbone.num_features
        self.heads = nn.ModuleList(
            nn.Linear(feature_dim, 1) for _ in range(num_criteria))

    def forward(self, x):
        """(B, in_channels, H, W) -> (B, num_criteria) criterion logits."""
        features = self.backbone(x)
        return torch.cat([head(features) for head in self.heads], dim=1)

    def param_groups(self, lr_backbone: float, lr_heads: float) -> list[dict]:
        """Optimizer parameter groups — backbone at ``lr_backbone``, the binary
        criterion heads at ``lr_heads``.
        """
        return [
            {"params": list(self.backbone.parameters()), "lr": lr_backbone},
            {"params": list(self.heads.parameters()), "lr": lr_heads},
        ]
