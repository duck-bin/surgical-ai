"""Per-class pixel statistics and imbalance-handling utilities.

CholecSeg8k is severely imbalanced (liver ~30%, abdominal wall ~28%, while
cystic duct <1% and grasper ~0.6% of pixels). This module provides:
    - per-class pixel-frequency statistics,
    - inverse-sqrt-frequency loss weights, clipped to [0.5, 10.0],
    - a WeightedRandomSampler over the training set.

For meaningful statistics, compute frequencies on a dataset configured with
the *eval* transform (deterministic) rather than the randomized train pipeline.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

from src.data.cholecseg8k import NUM_CLASSES


def _mask_of(item, mask_key: str) -> np.ndarray:
    """Extract an integer label mask from a dataset item as a numpy array."""
    mask = item[mask_key] if isinstance(item, dict) else item
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()
    return np.asarray(mask)


def compute_pixel_frequencies(dataset, num_classes: int = NUM_CLASSES,
                              mask_key: str = "mask"):
    """Return ``(counts, frequencies)`` — per-class pixel totals over a dataset.

    ``counts`` is an int64 array of shape (num_classes,); ``frequencies`` is the
    same normalized to sum to 1.
    """
    counts = np.zeros(num_classes, dtype=np.int64)
    for i in range(len(dataset)):
        mask = _mask_of(dataset[i], mask_key)
        binc = np.bincount(mask.ravel().astype(np.int64), minlength=num_classes)
        counts += binc[:num_classes]
    total = int(counts.sum())
    frequencies = counts / total if total > 0 else counts.astype(np.float64)
    return counts, frequencies


def class_loss_weights(frequencies, clip: tuple[float, float] = (0.5, 10.0)
                       ) -> torch.Tensor:
    """Inverse-sqrt-frequency class weights, clipped to ``clip``.

    Returns a float32 tensor of shape (num_classes,). Classes with zero
    frequency (e.g. ``cystic_artery`` in CholecSeg8k) receive the upper clip.
    """
    freqs = np.asarray(frequencies, dtype=np.float64)
    with np.errstate(divide="ignore"):
        weights = 1.0 / np.sqrt(freqs)
    weights[~np.isfinite(weights)] = clip[1]
    weights = np.clip(weights, clip[0], clip[1])
    return torch.tensor(weights, dtype=torch.float32)


def make_weighted_sampler(dataset, num_classes: int = NUM_CLASSES,
                          mask_key: str = "mask") -> WeightedRandomSampler:
    """Build a WeightedRandomSampler that oversamples frames containing rare
    classes.

    Each frame is weighted by the inverse-sqrt frequency of the rarest class
    present in it, so frames with (e.g.) the cystic duct are drawn more often.
    """
    _, frequencies = compute_pixel_frequencies(dataset, num_classes, mask_key)
    with np.errstate(divide="ignore"):
        inv = 1.0 / np.sqrt(np.where(frequencies > 0, frequencies, np.inf))

    sample_weights = np.empty(len(dataset), dtype=np.float64)
    for i in range(len(dataset)):
        present = np.unique(_mask_of(dataset[i], mask_key).astype(np.int64))
        present = present[(present >= 0) & (present < num_classes)]
        sample_weights[i] = float(inv[present].max()) if present.size else 1.0

    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(dataset),
        replacement=True,
    )
