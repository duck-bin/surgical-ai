"""Tests for the segmentation losses.

Synthetic logits/targets only (no dataset, no GPU). The focus is the
small-structure behaviour the Tversky loss is meant to provide: that missing a
thin class (a false negative) is penalized harder than an equivalent false
positive, that beta > alpha amplifies that asymmetry, and that alpha = beta
recovers Dice up to smoothing.
"""
import torch

from src.losses.dice import DiceLoss
from src.losses.focal import FocalLoss
from src.losses.tversky import TverskyLoss


def test_focal_class_weights_excluded_from_state_dict():
    """class_weights are recomputed per run, so they must not be checkpointed.

    Otherwise a weighted run cannot resume an unweighted checkpoint (strict
    load fails on 'focal.class_weights'). The buffer is persistent=False -- absent
    from state_dict yet still applied at runtime.
    """
    weighted = FocalLoss(gamma=2.0, class_weights=[1.0, 5.0, 2.0, 10.0, 10.0, 1.0])
    plain = FocalLoss(gamma=2.0)
    assert "class_weights" not in weighted.state_dict()
    # Interchangeable state_dicts -> a weighted run can resume an unweighted one.
    assert weighted.state_dict().keys() == plain.state_dict().keys()
    assert weighted.class_weights is not None  # still applied at runtime


def _logits_from_prediction(pred_class, num_classes):
    """One-hot logits (large positive on the predicted class) for a label map."""
    onehot = torch.nn.functional.one_hot(pred_class, num_classes)
    return onehot.permute(0, 3, 1, 2).float() * 20.0 - 10.0


def _fn_fp_cases(num_classes=6, size=16, duct=3):
    """A thin-class target plus a 'miss' (FN) and an 'over-segment' (FP) pred.

    Returns ``(target, miss_logits, over_logits)``. ``miss`` predicts pure
    background (the duct stripe is entirely missed); ``over`` keeps the real
    duct and adds an equal-area spurious stripe elsewhere -- the same number of
    wrong pixels, but as false positives instead of false negatives.
    """
    target = torch.zeros(1, size, size, dtype=torch.long)
    target[:, 7:9, :] = duct  # a thin 2-row 'duct' stripe
    miss = torch.zeros(1, size, size, dtype=torch.long)
    over = target.clone()
    over[:, 0:2, :] = duct
    return (target,
            _logits_from_prediction(miss, num_classes),
            _logits_from_prediction(over, num_classes))


def test_tversky_matches_dice_at_half_half():
    """alpha = beta = 0.5, gamma = 1 reproduces Dice up to the smoothing term.

    The two losses smooth slightly differently (Dice adds the constant once to a
    doubled-TP numerator), so on a large frame -- where TP dominates the
    smoothing -- they coincide to a few parts in a thousand.
    """
    torch.manual_seed(0)
    logits = torch.randn(1, 6, 64, 64)
    target = torch.randint(0, 6, (1, 64, 64))

    dice = DiceLoss()(logits, target)
    tversky = TverskyLoss(alpha=0.5, beta=0.5, gamma=1.0)(logits, target)

    assert torch.allclose(dice, tversky, atol=5e-3)


def test_tversky_penalizes_false_negative_more_than_false_positive():
    """beta > alpha => missing a thin class costs more than over-segmenting it."""
    target, miss_logits, over_logits = _fn_fp_cases()
    loss = TverskyLoss(alpha=0.3, beta=0.7, gamma=1.0)

    fn_loss = loss(miss_logits, target)
    fp_loss = loss(over_logits, target)

    assert fn_loss > fp_loss  # the miss is the worse error for a small class


def test_beta_amplifies_false_negative_penalty():
    """Raising beta over alpha widens the miss-vs-over-segment loss gap.

    A full miss already costs more than over-segmenting (its per-class Tversky
    collapses toward zero either way), so the test is *relative*: the FN/FP loss
    ratio under an FN-weighted (0.3/0.7) Tversky exceeds the symmetric (0.5/0.5)
    one. That widening is exactly the recall pressure beta > alpha is meant to add.
    """
    target, miss_logits, over_logits = _fn_fp_cases()
    symmetric = TverskyLoss(alpha=0.5, beta=0.5, gamma=1.0)
    fn_weighted = TverskyLoss(alpha=0.3, beta=0.7, gamma=1.0)

    ratio_sym = symmetric(miss_logits, target) / symmetric(over_logits, target)
    ratio_fn = fn_weighted(miss_logits, target) / fn_weighted(over_logits, target)

    assert ratio_fn > ratio_sym


def test_tversky_perfect_prediction_is_near_zero():
    """A correct prediction drives the loss to ~0 for any gamma."""
    num_classes, size = 6, 16
    target = torch.randint(0, num_classes, (2, size, size))
    logits = _logits_from_prediction(target, num_classes)

    for gamma in (1.0, 1.333, 2.0):
        loss = TverskyLoss(alpha=0.3, beta=0.7, gamma=gamma)(logits, target)
        assert loss < 1e-2


def test_focal_gamma_changes_loss_on_hard_class():
    """gamma != 1 reshapes the loss when a class is poorly segmented.

    Right half of the frame is the rare 'duct'; the prediction misses it
    entirely. The focal exponent changes the loss value (it is not a no-op) and
    the missed class keeps the loss well above zero.
    """
    num_classes, size = 6, 16
    target = torch.zeros(1, size, size, dtype=torch.long)
    target[:, :, 8:] = 3
    pred = torch.zeros(1, size, size, dtype=torch.long)  # misses the duct entirely
    logits = _logits_from_prediction(pred, num_classes)

    plain = TverskyLoss(alpha=0.3, beta=0.7, gamma=1.0)(logits, target)
    focal = TverskyLoss(alpha=0.3, beta=0.7, gamma=2.0)(logits, target)

    assert not torch.allclose(plain, focal)
    assert focal > 0.1
