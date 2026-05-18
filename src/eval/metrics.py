"""Evaluation metrics with 95% bootstrap confidence intervals.

Segmentation : per-class IoU, per-class Dice, mIoU, mDice.
CVS          : per-criterion AP, balanced accuracy, mAP across 3 criteria.
Composite    : Cohen's quadratic-weighted kappa between predicted and
               ground-truth CVS score (0-3).
All metrics support 95% bootstrap CIs (n_resamples = 1000).
Implemented in Step 3 (segmentation) and Step 7 (CVS).
"""
from __future__ import annotations


def iou_score(pred, target, num_classes: int = 6):
    """Per-class and mean Intersection-over-Union."""
    raise NotImplementedError("Implemented in Step 3.")


def dice_score(pred, target, num_classes: int = 6):
    """Per-class and mean Dice coefficient."""
    raise NotImplementedError("Implemented in Step 3.")


def bootstrap_ci(metric_fn, *args, n_resamples: int = 1000, alpha: float = 0.05):
    """95% bootstrap confidence interval for an arbitrary metric."""
    raise NotImplementedError("Implemented in Step 3.")


def cvs_metrics(pred_logits, targets):
    """Per-criterion AP, balanced accuracy and mAP for the 3 CVS criteria."""
    raise NotImplementedError("Implemented in Step 7.")


def quadratic_weighted_kappa(pred_score, true_score):
    """Cohen's quadratic-weighted kappa on the 0-3 CVS score."""
    raise NotImplementedError("Implemented in Step 7.")
