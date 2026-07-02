"""Marginal (partial-label) losses for joint multi-dataset segmentation.

The end-to-end CVS pipeline needs ONE segmentation model that emits all six
shared classes (background, liver, gallbladder, cystic_duct, cystic_artery,
tool), because the downstream CVS classifier consumes all six mask channels.
No single dataset provides them: CholecSeg8k labels the big anatomy
(background/liver/gallbladder/tool) but effectively not the cystic duct and
never the cystic artery, while Endoscapes-Seg labels the two CVS-critical
tubular structures (cystic_duct/cystic_artery) plus gallbladder/tool but has no
liver label. Each dataset is therefore *partially labelled* in the shared
6-class space: the classes it does not annotate are still present in its frames
but folded into its own "background" label.

Training the union of the two with ordinary cross-entropy is actively harmful.
An Endoscapes frame labels liver pixels as background, so full CE pushes the
model to *suppress* liver exactly where CholecSeg8k teaches it; symmetrically a
CholecSeg8k frame labels any cystic-duct/artery pixels as background, undoing
what Endoscapes teaches. The two datasets fight over the background channel.

The fix is the *marginal loss* (Shi et al., "Marginal loss and exclusion loss
for partially supervised multi-organ segmentation", Medical Image Analysis
2021; the same idea underlies DoDNet-style partial supervision). Per sample, we
supervise only the classes its source annotates. Every class the source does
*not* annotate is merged into the background probability, so a
background-labelled pixel is satisfied by predicting background OR any
unlabelled class -- the model is free to predict liver on an Endoscapes frame
(folded into that frame's "background") without penalty, and free to predict the
duct on a CholecSeg8k frame likewise. When every class is labelled the merge is
empty and these losses reduce *exactly* to ordinary focal + Dice/Tversky, so
single-dataset training is unchanged.

The ``labeled`` argument is a per-sample boolean mask of shape ``(B, C)``:
``labeled[b, c]`` is True when class ``c`` is annotated for sample ``b``.
Background (index 0) is always treated as labelled.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_EPS = 1e-8


def _foreground_unlabeled(labeled: torch.Tensor) -> torch.Tensor:
    """``(B, C)`` bool mask of the *foreground* classes not labelled per sample.

    Background (index 0) is forced False so its probability is never
    double-counted when it is merged with the unlabelled classes.
    """
    unlabeled = ~labeled.bool()
    unlabeled[:, 0] = False
    return unlabeled


def marginal_focal_loss(logits, target, labeled, *, gamma: float = 2.0,
                        class_weights=None, ignore_index: int = -100):
    """Focal loss with each sample's unlabelled classes merged into background.

    ``logits`` are ``(B, C, H, W)``, ``target`` is ``(B, H, W)`` of class
    indices, ``labeled`` is ``(B, C)`` bool. For a background-labelled pixel the
    supervised probability mass is ``p(background) + sum_c p(c)`` over the
    unlabelled foreground classes ``c`` -- so predicting an unlabelled class
    there costs nothing. Foreground-labelled pixels are supervised normally.
    """
    log_prob = F.log_softmax(logits, dim=1)
    prob = log_prob.exp()

    unlabeled = _foreground_unlabeled(labeled).to(prob.dtype)[:, :, None, None]
    background_extra = (prob * unlabeled).sum(dim=1)          # (B, H, W)

    target_safe = target.clamp(min=0)
    prob_t = prob.gather(1, target_safe.unsqueeze(1)).squeeze(1)
    # Only a background-labelled pixel absorbs the unlabelled-class mass.
    mass = torch.where(target_safe == 0, prob_t + background_extra, prob_t)
    log_mass = mass.clamp_min(_EPS).log()
    focal = -((1.0 - mass) ** gamma) * log_mass

    if class_weights is not None:
        focal = focal * class_weights.to(focal.dtype)[target_safe]

    valid = target != ignore_index
    focal = focal * valid
    return focal.sum() / valid.sum().clamp(min=1)


def marginal_tversky_loss(logits, target, labeled, *, alpha: float = 0.5,
                          beta: float = 0.5, gamma: float = 1.0,
                          smooth: float = 1.0):
    """Macro Tversky loss over each sample's *labelled* classes only.

    Unlabelled foreground classes are merged into the background channel (both
    prediction and target), then the Tversky index is computed over the kept
    channels and averaged. ``alpha == beta == 0.5`` and ``gamma == 1`` is soft
    Dice; ``beta > alpha`` favours recall on thin classes (see
    :class:`~src.losses.tversky.TverskyLoss`).
    """
    num_classes = logits.shape[1]
    prob_all = F.softmax(logits, dim=1)
    labeled = labeled.bool()

    losses = []
    for b in range(logits.shape[0]):
        keep = labeled[b].clone()
        keep[0] = True                                   # background always kept
        unlabeled = ~keep                                # foreground, unlabelled
        prob = prob_all[b]                               # (C, H, W)

        background = prob[0]
        if bool(unlabeled.any()):
            background = background + prob[unlabeled].sum(dim=0)
        kept_indices = [0] + [c for c in range(1, num_classes) if bool(keep[c])]
        probs = torch.stack(
            [background if c == 0 else prob[c] for c in kept_indices], dim=0)

        target_1h = F.one_hot(target[b].clamp(min=0), num_classes)
        target_1h = target_1h.permute(2, 0, 1).to(prob.dtype)  # (C, H, W)
        targets = target_1h[kept_indices]                # (K, H, W)

        dims = (1, 2)
        tp = (probs * targets).sum(dims)
        fp = (probs * (1.0 - targets)).sum(dims)
        fn = ((1.0 - probs) * targets).sum(dims)
        tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
        losses.append(((1.0 - tversky) ** gamma).mean())

    return torch.stack(losses).mean()


class PartialLabelSegLoss(nn.Module):
    """Marginal focal + marginal region loss for partially-labelled joint data.

    Mirrors the composition used by :class:`~src.train.lightning_modules.
    SegmentationModule` (``focal_weight * focal + region_weight * region``) but
    every term is marginalised over each sample's unlabelled classes. ``forward``
    takes the extra ``labeled`` mask ``(B, C)``; with an all-True mask it equals
    the ordinary focal + Dice/Tversky loss.

    Args:
        focal_weight / region_weight: weights on the two terms (``region_weight``
            is the segmentation config's ``dice_weight``).
        focal_gamma: focal exponent.
        class_weights: optional per-class weights for the focal term (not stored
            in the checkpoint; recomputed from the data each run).
        region_loss: ``dice`` | ``tversky`` | ``focal_tversky``.
        tversky_alpha/beta/gamma: Tversky parameters (ignored for plain Dice).
        ignore_index: target value excluded from the focal term.
    """

    def __init__(self, *, focal_weight: float = 1.0, region_weight: float = 1.0,
                 focal_gamma: float = 2.0, class_weights=None,
                 region_loss: str = "dice", tversky_alpha: float = 0.3,
                 tversky_beta: float = 0.7, tversky_gamma: float = 1.0,
                 smooth: float = 1.0, ignore_index: int = -100):
        super().__init__()
        self.focal_weight = focal_weight
        self.region_weight = region_weight
        self.focal_gamma = focal_gamma
        self.ignore_index = ignore_index
        self.smooth = smooth
        if str(region_loss).lower() in ("tversky", "focal_tversky"):
            self.tversky_alpha = tversky_alpha
            self.tversky_beta = tversky_beta
            self.tversky_gamma = tversky_gamma
        else:  # plain soft Dice is Tversky with alpha == beta == 0.5, gamma == 1
            self.tversky_alpha = 0.5
            self.tversky_beta = 0.5
            self.tversky_gamma = 1.0
        if class_weights is not None:
            # persistent=False: class weights are data-derived, not learned, so
            # they must stay out of the checkpoint (mirrors FocalLoss) -- else a
            # weighted run could not resume an unweighted one under strict load.
            self.register_buffer(
                "class_weights",
                torch.as_tensor(class_weights, dtype=torch.float32),
                persistent=False,
            )
        else:
            self.class_weights = None

    def forward(self, logits, target, labeled):
        focal = marginal_focal_loss(
            logits, target, labeled, gamma=self.focal_gamma,
            class_weights=self.class_weights, ignore_index=self.ignore_index)
        region = marginal_tversky_loss(
            logits, target, labeled, alpha=self.tversky_alpha,
            beta=self.tversky_beta, gamma=self.tversky_gamma, smooth=self.smooth)
        return self.focal_weight * focal + self.region_weight * region
