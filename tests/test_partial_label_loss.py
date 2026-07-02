"""Tests for the marginal (partial-label) segmentation loss.

The marginal loss must (1) equal the ordinary focal + Dice/Tversky loss when
every class is labelled -- so single-dataset training is unaffected -- and (2)
stop penalising a background-labelled pixel for predicting a class the sample's
source does not annotate (the whole point: liver on an Endoscapes frame, or the
cystic duct on a CholecSeg8k frame). Pure-torch, tiny tensors, no data.
"""
import torch

from src.losses.focal import FocalLoss
from src.losses.partial_label import (
    PartialLabelSegLoss,
    marginal_focal_loss,
    marginal_tversky_loss,
)
from src.losses.tversky import TverskyLoss

torch.manual_seed(0)


def _logits(batch=2, classes=6, size=8):
    return torch.randn(batch, classes, size, size, requires_grad=True)


def _target(batch=2, classes=6, size=8):
    return torch.randint(0, classes, (batch, size, size))


def test_marginal_focal_matches_focal_under_full_supervision():
    logits, target = _logits(), _target()
    all_labeled = torch.ones(logits.shape[0], logits.shape[1], dtype=torch.bool)
    marginal = marginal_focal_loss(logits, target, all_labeled, gamma=2.0)
    reference = FocalLoss(gamma=2.0)(logits, target)
    assert torch.allclose(marginal, reference, atol=1e-6)


def test_marginal_dice_is_tversky_half_half():
    """The marginal 'dice' path is Tversky(alpha=beta=0.5) -- the same index.

    Soft Dice equals the Tversky index at alpha == beta == 0.5; the marginal
    region loss is a Tversky family, so its 'dice' setting is exactly
    ``TverskyLoss(0.5, 0.5)`` (they share the Laplace-smoothing convention).
    ``DiceLoss`` smooths the 2*TP form slightly differently, so we compare
    against the exact equivalent here.
    """
    logits, target = _logits(1), _target(1)
    all_labeled = torch.ones(logits.shape[0], logits.shape[1], dtype=torch.bool)
    marginal = marginal_tversky_loss(logits, target, all_labeled,
                                     alpha=0.5, beta=0.5, gamma=1.0)
    reference = TverskyLoss(alpha=0.5, beta=0.5, gamma=1.0)(logits, target)
    assert torch.allclose(marginal, reference, atol=1e-6)


def test_marginal_tversky_matches_tversky_under_full_supervision():
    logits, target = _logits(), _target()
    all_labeled = torch.ones(logits.shape[0], logits.shape[1], dtype=torch.bool)
    marginal = marginal_tversky_loss(logits[:1], target[:1], all_labeled[:1],
                                     alpha=0.3, beta=0.7, gamma=1.333)
    reference = TverskyLoss(alpha=0.3, beta=0.7, gamma=1.333)(logits[:1],
                                                              target[:1])
    assert torch.allclose(marginal, reference, atol=1e-6)


def test_unlabeled_prediction_on_background_pixel_is_not_penalised():
    """A background-labelled pixel predicting an *unlabelled* class costs ~0."""
    classes = 4
    # One pixel, target = background (0). Model is fully confident in class 3.
    logits = torch.full((1, classes, 1, 1), -10.0)
    logits[0, 3, 0, 0] = 10.0
    target = torch.zeros(1, 1, 1, dtype=torch.long)

    labeled_all = torch.ones(1, classes, dtype=torch.bool)
    # Class 3 unlabelled for this sample -> its mass folds into background.
    labeled_partial = labeled_all.clone()
    labeled_partial[0, 3] = False

    penalised = marginal_focal_loss(logits, target, labeled_all)
    absorbed = marginal_focal_loss(logits, target, labeled_partial)
    assert penalised.item() > 1.0          # confident wrong background -> big loss
    assert absorbed.item() < 1e-3          # folded into background -> ~no loss


def test_unlabeled_class_folds_in_region_loss_too():
    """Predicting an unlabelled class on background pixels is free in Tversky."""
    classes = 4
    logits = torch.full((1, classes, 4, 4), -10.0)
    logits[0, 3] = 10.0                     # confidently predict class 3 everywhere
    target = torch.zeros(1, 4, 4, dtype=torch.long)  # all background

    labeled_all = torch.ones(1, classes, dtype=torch.bool)
    labeled_partial = labeled_all.clone()
    labeled_partial[0, 3] = False

    penalised = marginal_tversky_loss(logits, target, labeled_all,
                                      alpha=0.5, beta=0.5, gamma=1.0)
    absorbed = marginal_tversky_loss(logits, target, labeled_partial,
                                     alpha=0.5, beta=0.5, gamma=1.0)
    assert penalised.item() > absorbed.item()
    assert absorbed.item() < 1e-2


def test_partial_label_loss_composite_is_differentiable_and_reduces():
    logits, target = _logits(), _target()
    labeled = torch.ones(logits.shape[0], logits.shape[1], dtype=torch.bool)
    loss = PartialLabelSegLoss(focal_weight=1.0, region_weight=1.0,
                               region_loss="focal_tversky")(logits, target, labeled)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_class_weights_do_not_enter_state_dict():
    """Data-derived focal weights stay out of the checkpoint (persistent=False)."""
    weights = torch.tensor([0.5, 1.0, 2.0, 3.0, 4.0, 30.0])
    loss = PartialLabelSegLoss(class_weights=weights)
    assert "class_weights" not in loss.state_dict()
