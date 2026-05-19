"""Evaluation metrics with 95% bootstrap confidence intervals.

Segmentation : per-class IoU, per-class Dice, mIoU, mDice.
CVS          : per-criterion AP, balanced accuracy, mAP across 3 criteria.
Composite    : Cohen's quadratic-weighted kappa between predicted and
               ground-truth CVS score (0-3).
Segmentation metrics are implemented here; the CVS metrics are added in Step 7.
"""
from __future__ import annotations

import numpy as np


def _to_numpy(x) -> np.ndarray:
    """Convert a torch tensor or array-like to a numpy array."""
    try:
        import torch

        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


def iou_score(pred, target, num_classes: int = 6):
    """Per-class IoU and mean IoU for integer label maps.

    Returns ``(per_class, miou)``. ``per_class`` is a float array of shape
    (num_classes,); a class absent from both prediction and target is NaN and
    excluded from the mean.
    """
    pred = _to_numpy(pred).astype(np.int64).ravel()
    target = _to_numpy(target).astype(np.int64).ravel()
    per_class = np.full(num_classes, np.nan)
    for c in range(num_classes):
        p, t = pred == c, target == c
        union = int(np.count_nonzero(p | t))
        if union:
            per_class[c] = np.count_nonzero(p & t) / union
    miou = float(np.nanmean(per_class)) if np.any(~np.isnan(per_class)) else 0.0
    return per_class, miou


def dice_score(pred, target, num_classes: int = 6):
    """Per-class Dice and mean Dice for integer label maps.

    Returns ``(per_class, mdice)`` with the same NaN convention as
    :func:`iou_score`.
    """
    pred = _to_numpy(pred).astype(np.int64).ravel()
    target = _to_numpy(target).astype(np.int64).ravel()
    per_class = np.full(num_classes, np.nan)
    for c in range(num_classes):
        p, t = pred == c, target == c
        denom = int(np.count_nonzero(p) + np.count_nonzero(t))
        if denom:
            per_class[c] = 2.0 * np.count_nonzero(p & t) / denom
    mdice = float(np.nanmean(per_class)) if np.any(~np.isnan(per_class)) else 0.0
    return per_class, mdice


def bootstrap_ci(values, n_resamples: int = 1000, alpha: float = 0.05,
                 seed: int = 42):
    """95% bootstrap confidence interval for the mean of per-sample metrics.

    Args:
        values: 1-D array of per-sample metric values (e.g. per-image IoU).
        n_resamples: number of bootstrap resamples.
        alpha: significance level (0.05 -> 95% CI).
        seed: RNG seed for reproducibility.

    Returns:
        ``(mean, ci_low, ci_high)``.
    """
    values = _to_numpy(values).astype(np.float64).ravel()
    values = values[~np.isnan(values)]
    if values.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_resamples, values.size))
    means = values[idx].mean(axis=1)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return (float(values.mean()), lo, hi)


def cvs_metrics(pred_logits, targets):
    """Per-criterion AP, balanced accuracy and mAP for the CVS criteria.

    Args:
        pred_logits: (N, C) raw logits (or probabilities) for C binary criteria.
        targets: (N, C) binary {0, 1} ground-truth criteria.

    Returns:
        dict with ``ap`` (per-criterion average precision; NaN for a criterion
        with a single class present), ``balanced_accuracy`` (per-criterion),
        and ``map`` (mean AP over criteria).
    """
    from sklearn.metrics import average_precision_score, balanced_accuracy_score

    logits = _to_numpy(pred_logits).astype(np.float64)
    target = _to_numpy(targets).astype(np.int64)
    if logits.ndim == 1:
        logits, target = logits.reshape(-1, 1), target.reshape(-1, 1)
    # Sigmoid is monotonic, so it does not affect the ranking-based AP; it only
    # matters for thresholding the balanced-accuracy predictions at 0.5.
    scores = 1.0 / (1.0 + np.exp(-logits))
    preds = (scores >= 0.5).astype(np.int64)

    ap, balanced = [], []
    for c in range(target.shape[1]):
        y_true = target[:, c]
        if y_true.min() == y_true.max():  # one class only -> metrics undefined
            ap.append(float("nan"))
            balanced.append(float("nan"))
            continue
        ap.append(float(average_precision_score(y_true, scores[:, c])))
        balanced.append(float(balanced_accuracy_score(y_true, preds[:, c])))
    mean_ap = float(np.nanmean(ap)) if np.any(~np.isnan(ap)) else 0.0
    return {"ap": ap, "balanced_accuracy": balanced, "map": mean_ap}


def quadratic_weighted_kappa(pred_score, true_score):
    """Cohen's quadratic-weighted kappa on the 0-3 CVS score."""
    from sklearn.metrics import cohen_kappa_score

    pred = _to_numpy(pred_score).astype(np.int64).ravel()
    true = _to_numpy(true_score).astype(np.int64).ravel()
    return float(cohen_kappa_score(true, pred, weights="quadratic",
                                   labels=[0, 1, 2, 3]))
