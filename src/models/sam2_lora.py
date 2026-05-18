"""SAM2 with LoRA adapters for surgical anatomical segmentation (primary model).

Base: ``facebook/sam2-hiera-base-plus`` (~89M params). LoRA adapters are added
to the image encoder (rank 8, alpha 16, qkv projections) and the mask decoder
(rank 16, alpha 32) via PEFT. At inference the model runs in automatic-mask-
generation mode (32x32 point grid + NMS); a small MLP head classifies each
SAM2 mask-token embedding into one of the 6 target classes.

Motivated by SurgiSAM2 (Kamtam et al., arXiv:2503.03942, 2025).
Implemented in Step 4 (architecture) and Step 5 (fine-tuning).
"""
from __future__ import annotations

import torch.nn as nn


class SAM2LoRASegmenter(nn.Module):
    """SAM2 + LoRA segmentation model with a 6-class mask classifier head.

    Implemented in Step 4.
    """

    def __init__(self, base_checkpoint: str = "facebook/sam2-hiera-base-plus",
                 num_classes: int = 6):
        super().__init__()
        raise NotImplementedError("SAM2 + LoRA model is implemented in Step 4.")

    def forward(self, x):
        raise NotImplementedError("Implemented in Step 4.")
