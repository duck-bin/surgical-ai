"""Albumentations augmentation pipelines for surgical frames.

Train-time augmentations (implemented in Step 2):
    Geometric    - HorizontalFlip(p=0.5), ShiftScaleRotate(p=0.3, limit=0.1)
    Photometric  - RandomBrightnessContrast(p=0.5), HueSaturationValue(p=0.3)
    Surgical     - GaussianBlur(blur_limit=(3,7), p=0.2)  -> cautery smoke
                   CoarseDropout(p=0.2)                   -> occlusion
NOTE: VerticalFlip is intentionally NOT used (anatomically nonsensical).

All pipelines letterbox-pad to a square ``image_size`` (default 512).
"""
from __future__ import annotations


def build_train_transforms(image_size: int = 512):
    """Albumentations training pipeline (geometric + photometric + surgical)."""
    raise NotImplementedError("Implemented in Step 2.")


def build_eval_transforms(image_size: int = 512):
    """Albumentations eval pipeline (letterbox resize + normalize only)."""
    raise NotImplementedError("Implemented in Step 2.")
