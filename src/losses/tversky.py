"""Tversky / Focal-Tversky loss for small, thin anatomical structures.

Dice loss weights false positives and false negatives equally. For a tiny,
thin class like ``cystic_duct`` (~0.03% of pixels) the model almost always errs
by *missing* the structure -- a false negative -- so a symmetric region loss
under-penalizes exactly the failure mode that keeps its Dice at zero.

The Tversky index (Salehi et al., 2017) generalizes Dice with separate weights
``alpha`` (false positives) and ``beta`` (false negatives):

    T = TP / (TP + alpha * FP + beta * FN)

with ``alpha = beta = 0.5`` it *is* Dice; choosing ``beta > alpha`` (e.g.
0.3 / 0.7) penalizes misses harder than spurious pixels, pulling the optimizer
toward higher recall on the rare class. The Focal-Tversky variant (Abraham &
Khan, 2019) raises the per-class loss to a power ``gamma > 1`` so that classes
which are still poorly segmented (low Tversky -- the rare ones) dominate the
gradient. ``gamma = 1`` recovers plain Tversky.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TverskyLoss(nn.Module):
    """Multi-class (Focal-)Tversky loss, macro-averaged over classes.

    Expects raw logits of shape (B, C, H, W) and integer targets (B, H, W).
    Probabilities come from a softmax, so the loss is differentiable. The index
    is computed per class over the whole batch and averaged across classes, so
    each class contributes equally regardless of pixel count -- the property
    that makes region losses robust to extreme imbalance.

    Args:
        alpha: false-positive weight.
        beta: false-negative weight (set ``beta > alpha`` to favour recall on
            small/thin structures).
        gamma: focal exponent on the per-class ``(1 - Tversky)`` term;
            ``gamma = 1`` is plain Tversky, ``gamma > 1`` focuses on the
            still-poorly-segmented (rare) classes.
        smooth: Laplace smoothing, also defining the loss of an absent class as
            0 (``smooth / smooth = 1`` -> ``(1 - 1) ** gamma = 0``).
        class_weights: optional per-class multipliers on the macro average.
            Usually left ``None`` -- the macro average already balances classes,
            and pixel-level class weighting belongs on the focal term.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7,
                 gamma: float = 1.0, smooth: float = 1.0, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        if class_weights is not None:
            self.register_buffer(
                "class_weights",
                torch.as_tensor(class_weights, dtype=torch.float32),
            )
        else:
            self.class_weights = None

    def forward(self, logits, target):
        num_classes = logits.shape[1]
        prob = F.softmax(logits, dim=1)
        target_1h = F.one_hot(target.clamp(min=0), num_classes)
        target_1h = target_1h.permute(0, 3, 1, 2).to(prob.dtype)  # (B, C, H, W)

        dims = (0, 2, 3)
        tp = (prob * target_1h).sum(dims)
        fp = (prob * (1.0 - target_1h)).sum(dims)
        fn = ((1.0 - prob) * target_1h).sum(dims)
        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth)

        per_class = (1.0 - tversky) ** self.gamma  # (C,)
        if self.class_weights is not None:
            weights = self.class_weights.to(per_class.dtype)
            return (per_class * weights).sum() / weights.sum().clamp(min=1e-8)
        return per_class.mean()
