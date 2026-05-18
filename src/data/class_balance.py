"""Per-class pixel statistics and imbalance-handling utilities.

CholecSeg8k is severely imbalanced (liver ~30%, abdominal wall ~28%, while
cystic duct <1% and grasper ~0.6% of pixels). This module provides:
    - per-class pixel-frequency statistics,
    - inverse-sqrt-frequency loss weights, clipped to [0.5, 10.0],
    - a WeightedRandomSampler over the training set.
Implemented in Step 2.
"""
from __future__ import annotations


def compute_pixel_frequencies(dataset):
    """Return per-class pixel counts / frequencies over a dataset."""
    raise NotImplementedError("Implemented in Step 2.")


def class_loss_weights(frequencies, clip: tuple[float, float] = (0.5, 10.0)):
    """Inverse-sqrt-frequency class weights, clipped to ``clip``."""
    raise NotImplementedError("Implemented in Step 2.")


def make_weighted_sampler(dataset):
    """Build a WeightedRandomSampler from inverse class frequency."""
    raise NotImplementedError("Implemented in Step 2.")
