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

import os

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


def compute_class_stats(dataset, num_classes: int = NUM_CLASSES,
                        mask_key: str = "mask"):
    """Per-class frequencies AND per-frame sampler weights in a *single* pass.

    Returns ``(frequencies, sample_weights)``:
        frequencies   - float64 (num_classes,), normalized pixel frequencies.
        sample_weights - float64 (len(dataset),), each frame weighted by the
            inverse-sqrt frequency of the rarest class it contains (so frames
            with rare classes are oversampled).

    This fuses :func:`compute_pixel_frequencies` and the weighting loop of
    :func:`make_weighted_sampler` so the dataset (whose items are decoded images)
    is iterated once instead of twice -- the expensive part on large datasets.
    """
    counts = np.zeros(num_classes, dtype=np.int64)
    present_per_frame: list[np.ndarray] = []
    for i in range(len(dataset)):
        mask = _mask_of(dataset[i], mask_key).ravel().astype(np.int64)
        binc = np.bincount(mask, minlength=num_classes)[:num_classes]
        counts += binc
        present_per_frame.append(np.nonzero(binc)[0])

    total = int(counts.sum())
    frequencies = counts / total if total > 0 else counts.astype(np.float64)
    with np.errstate(divide="ignore"):
        inv = 1.0 / np.sqrt(np.where(frequencies > 0, frequencies, np.inf))

    sample_weights = np.empty(len(dataset), dtype=np.float64)
    for i, present in enumerate(present_per_frame):
        present = present[(present >= 0) & (present < num_classes)]
        sample_weights[i] = float(inv[present].max()) if present.size else 1.0
    return frequencies, sample_weights


def compute_or_load_class_stats(dataset, num_classes: int = NUM_CLASSES,
                                cache_path: str | None = None,
                                mask_key: str = "mask"):
    """:func:`compute_class_stats` with an on-disk ``.npz`` cache.

    The full pass over the train set (decoding thousands of masks) takes many
    minutes; this caches the result so re-runs and subsequent models start
    instantly. The cache is reused only when its ``sample_weights`` length and
    ``frequencies`` size match the current dataset -- the caller is responsible
    for keying ``cache_path`` on anything else that changes the split (e.g. the
    split seed).
    """
    if cache_path and os.path.exists(cache_path):
        try:
            cached = np.load(cache_path)
            freqs, weights = cached["frequencies"], cached["sample_weights"]
            if weights.shape[0] == len(dataset) and freqs.shape[0] == num_classes:
                return freqs, weights
        except Exception:
            pass  # corrupt/stale cache -> recompute below

    frequencies, sample_weights = compute_class_stats(dataset, num_classes, mask_key)
    if cache_path:
        directory = os.path.dirname(cache_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        np.savez(cache_path, frequencies=frequencies, sample_weights=sample_weights)
    return frequencies, sample_weights


def sampler_from_weights(sample_weights) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler from precomputed per-frame weights."""
    weights = torch.as_tensor(np.asarray(sample_weights), dtype=torch.double)
    return WeightedRandomSampler(weights=weights, num_samples=weights.numel(),
                                 replacement=True)
