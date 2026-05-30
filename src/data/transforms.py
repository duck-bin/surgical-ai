"""Albumentations augmentation pipelines for surgical frames.

Train-time augmentations:
    Geometric    - HorizontalFlip(p=0.5), ShiftScaleRotate(p=0.3, limit=0.1)
    Photometric  - RandomBrightnessContrast(p=0.5), HueSaturationValue(p=0.3)
    Surgical     - GaussianBlur(blur_limit=(3,7), p=0.2)  -> cautery smoke
                   CoarseDropout(p=0.2)                   -> occlusion
NOTE: VerticalFlip is intentionally NOT used (anatomically nonsensical).

All pipelines letterbox-resize (aspect-preserving) and zero-pad to a square
``image_size`` (default 512), then normalize with ImageNet statistics and
convert to tensors. Masks use nearest-neighbour interpolation throughout.

Targets the albumentations 2.x API (what ``albumentations>=2.0`` resolves to).
"""
from __future__ import annotations

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

# ImageNet statistics — the SAM2 Hiera / EfficientNet-B4 / ViT-Small backbones
# are all ImageNet-pretrained, so frames are normalized to match.
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


def _letterbox(image_size: int) -> list:
    """Aspect-preserving resize, then zero-pad to a square (mask padded with 0)."""
    return [
        A.LongestMaxSize(max_size=image_size, interpolation=cv2.INTER_LINEAR),
        A.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=0,
            position="center",
        ),
    ]


def build_train_transforms(image_size: int = 512, replay: bool = False) -> A.Compose:
    """Albumentations training pipeline (geometric + photometric + surgical).

    ``replay=True`` returns an :class:`A.ReplayCompose` instead of a plain
    :class:`A.Compose`. The temporal path uses it so the *same* random
    augmentation is applied to every frame in a window (spatial coherence is
    required for the ConvGRU); see ``CholecSeg8kWindowDataset``.
    """
    compose = A.ReplayCompose if replay else A.Compose
    return compose(
        [
            *_letterbox(image_size),
            # --- Geometric (no VerticalFlip — anatomically nonsensical) ---
            A.HorizontalFlip(p=0.5),
            # Spec gives only `limit=0.1`; applied to shift + scale, with a
            # small rotation. Constant zero border keeps padding as background.
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=15,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                fill_mask=0,
                p=0.3,
            ),
            # --- Photometric ---
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.3),
            # --- Surgical-specific ---
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),  # simulate cautery smoke
            A.CoarseDropout(  # simulate occlusion; mask left intact (fill_mask=None)
                num_holes_range=(1, 4),
                hole_height_range=(0.05, 0.15),
                hole_width_range=(0.05, 0.15),
                fill=0,
                p=0.2,
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )


def build_eval_transforms(image_size: int = 512) -> A.Compose:
    """Albumentations eval pipeline (letterbox resize + normalize only)."""
    return A.Compose(
        [
            *_letterbox(image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )
