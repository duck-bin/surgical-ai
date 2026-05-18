"""EfficientNet-B4 U-Net baseline for semantic segmentation.

Standard semantic segmentation baseline built on ``segmentation_models_pytorch``
(EfficientNet-B4 encoder, ImageNet weights). This baseline is the credibility
anchor for the paper — SAM2 results are never reported without it.
"""
from __future__ import annotations

import segmentation_models_pytorch as smp
import torch.nn as nn


class UNetBaseline(nn.Module):
    """EfficientNet-B4 U-Net producing ``num_classes`` segmentation logits.

    Args:
        encoder: ``segmentation_models_pytorch`` encoder name.
        encoder_weights: pretrained-weights tag (``"imagenet"``) or ``None``.
            Use ``None`` to skip the weight download (tests / offline use).
        in_channels: number of input channels (3 for RGB frames).
        num_classes: number of output segmentation classes.
    """

    def __init__(self, encoder: str = "efficientnet-b4",
                 encoder_weights: str | None = "imagenet",
                 in_channels: int = 3, num_classes: int = 6):
        super().__init__()
        self.net = smp.Unet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
        )

    def forward(self, x):
        """(B, in_channels, H, W) -> (B, num_classes, H, W) logits."""
        return self.net(x)

    def param_groups(self, lr_encoder: float, lr_decoder: float) -> list[dict]:
        """Optimizer parameter groups — encoder at ``lr_encoder``, decoder and
        segmentation head at ``lr_decoder``.
        """
        decoder_params = (list(self.net.decoder.parameters())
                          + list(self.net.segmentation_head.parameters()))
        return [
            {"params": list(self.net.encoder.parameters()), "lr": lr_encoder},
            {"params": decoder_params, "lr": lr_decoder},
        ]
