"""Metric tests: reproducibility, IoU, Dice, bootstrap CIs, and CVS metrics.

The known-input checks use hand-computed values.
"""
import numpy as np
import pytest
import torch

from src.eval.metrics import (
    bootstrap_ci,
    cvs_metrics,
    dice_score,
    iou_score,
    quadratic_weighted_kappa,
    temporal_consistency,
)
from src.utils.seeds import seed_everything


def test_seed_everything_is_reproducible():
    """Identical seeds produce identical NumPy and torch random draws."""
    seed_everything(123)
    np_a, torch_a = np.random.rand(8), torch.rand(8)

    seed_everything(123)
    np_b, torch_b = np.random.rand(8), torch.rand(8)

    assert np.allclose(np_a, np_b)
    assert torch.allclose(torch_a, torch_b)


def test_seed_everything_returns_seed():
    """seed_everything echoes back the seed it applied."""
    assert seed_everything(7) == 7


def test_iou_score_perfect_prediction():
    """A prediction identical to the target scores mIoU == 1.0."""
    mask = torch.randint(0, 6, (4, 8, 8))
    _, miou = iou_score(mask, mask, num_classes=6)
    assert miou == pytest.approx(1.0)


def test_iou_score_known_partial_overlap():
    """IoU on a hand-computed 2x2 mask pair matches the expected values."""
    pred = np.array([[0, 0], [1, 1]])
    target = np.array([[0, 1], [1, 1]])

    per_class, miou = iou_score(pred, target, num_classes=6)

    assert per_class[0] == pytest.approx(0.5)        # inter 1 / union 2
    assert per_class[1] == pytest.approx(2.0 / 3.0)  # inter 2 / union 3
    assert np.isnan(per_class[2])                    # class absent everywhere
    assert miou == pytest.approx((0.5 + 2.0 / 3.0) / 2.0)


def test_dice_score_known_partial_overlap():
    """Dice on the same hand-computed 2x2 mask pair matches expected values."""
    pred = np.array([[0, 0], [1, 1]])
    target = np.array([[0, 1], [1, 1]])

    per_class, _ = dice_score(pred, target, num_classes=6)

    assert per_class[0] == pytest.approx(2.0 / 3.0)  # 2*1 / (2 + 1)
    assert per_class[1] == pytest.approx(0.8)        # 2*2 / (2 + 3)


def test_bootstrap_ci_constant_input():
    """A constant array yields a degenerate CI equal to the value."""
    mean, lo, hi = bootstrap_ci(np.full(100, 0.5))
    assert mean == pytest.approx(0.5)
    assert lo == pytest.approx(0.5) and hi == pytest.approx(0.5)


def test_bootstrap_ci_brackets_the_mean():
    """The CI brackets the sample mean and is reproducible across seeds."""
    values = np.linspace(0.0, 1.0, 200)
    mean, lo, hi = bootstrap_ci(values, n_resamples=500)
    assert mean == pytest.approx(0.5, abs=0.02)
    assert lo <= mean <= hi
    assert bootstrap_ci(values, n_resamples=500) == (mean, lo, hi)


def test_cvs_metrics_perfect_separation():
    """Perfectly separable logits give per-criterion AP and mAP of 1.0."""
    logits = np.array([[-3.0, -3.0, -3.0],
                       [-3.0, -3.0, 3.0],
                       [3.0, 3.0, -3.0],
                       [3.0, 3.0, 3.0]])
    targets = np.array([[0, 0, 0], [0, 0, 1], [1, 1, 0], [1, 1, 1]])

    metrics = cvs_metrics(logits, targets)

    assert metrics["map"] == pytest.approx(1.0)
    assert all(ap == pytest.approx(1.0) for ap in metrics["ap"])
    assert all(ba == pytest.approx(1.0) for ba in metrics["balanced_accuracy"])


def test_cvs_metrics_single_class_criterion_is_nan():
    """A criterion with one class present yields NaN AP, excluded from mAP."""
    logits = np.array([[1.0, -1.0], [2.0, -2.0], [-1.0, 1.0]])
    targets = np.array([[1, 0], [1, 0], [0, 0]])  # criterion 1 is all-zero

    metrics = cvs_metrics(logits, targets)

    assert np.isnan(metrics["ap"][1])
    assert not np.isnan(metrics["ap"][0])
    assert metrics["map"] == pytest.approx(metrics["ap"][0])


def test_quadratic_weighted_kappa_perfect_agreement():
    """Identical CVS-score sequences give kappa 1.0."""
    scores = [0, 1, 2, 3, 1, 2]
    assert quadratic_weighted_kappa(scores, scores) == pytest.approx(1.0)


def test_quadratic_weighted_kappa_penalizes_larger_errors():
    """Quadratic weighting scores near-misses above far-misses."""
    near = quadratic_weighted_kappa([0, 1, 2, 3], [0, 1, 1, 3])
    far = quadratic_weighted_kappa([0, 1, 2, 3], [3, 2, 1, 0])
    assert near > far


def test_temporal_consistency_static_clip_is_perfect():
    """A clip whose frames are identical has perfect (1.0) consistency."""
    frame = np.array([[0, 0, 1], [2, 2, 1]])
    clip = np.stack([frame, frame, frame])  # (T=3, H, W)

    per_class, mean = temporal_consistency(clip, num_classes=3)

    assert mean == pytest.approx(1.0)
    for c in (0, 1, 2):
        assert per_class[c] == pytest.approx(1.0)


def test_temporal_consistency_flicker_lowers_score():
    """A class that appears/disappears between frames scores below a stable one."""
    # Class 1 is stable across both frames; class 2 flickers off then on.
    frame_a = np.array([[1, 1, 2], [0, 0, 0]])
    frame_b = np.array([[1, 1, 0], [0, 0, 2]])
    clip = np.stack([frame_a, frame_b])

    per_class, _ = temporal_consistency(clip, num_classes=3)

    assert per_class[1] == pytest.approx(1.0)   # stable class
    assert per_class[2] < per_class[1]          # flickering class is penalised


def test_temporal_consistency_single_frame_has_no_pairs():
    """A clip shorter than 2 frames has no pairs: NaN per class, 0.0 mean."""
    clip = np.array([[[0, 1], [2, 0]]])  # (T=1, H, W)

    per_class, mean = temporal_consistency(clip, num_classes=3)

    assert mean == 0.0
    assert np.all(np.isnan(per_class))
