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
import time

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

from src.data.cholecseg8k import NUM_CLASSES

# Bump when the *meaning* of the cached stats changes (e.g. the switch to
# counting native-resolution masks instead of transform-resized ones), so stale
# .npz caches recompute once instead of silently feeding old numbers.
_STATS_CACHE_VERSION = 2


def _mask_of(item, mask_key: str) -> np.ndarray:
    """Extract an integer label mask from a dataset item as a numpy array."""
    mask = item[mask_key] if isinstance(item, dict) else item
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()
    return np.asarray(mask)


def _frame_bincount(dataset, index: int, num_classes: int, mask_key: str,
                    loader) -> np.ndarray:
    """Per-class pixel counts for one frame, decoding the mask only if possible.

    When the dataset exposes ``load_mask(i)`` (CholecSeg8k / Endoscapes-Seg) it
    decodes *only* the mask at native resolution -- skipping the full RGB image
    decode and the resize the eval transform would apply. That is the bulk of the
    pre-train startup cost, since the stats pass never needs the image. Datasets
    without ``load_mask`` fall back to indexing (``dataset[i][mask_key]``).
    """
    mask = np.asarray(loader(index)) if loader is not None else _mask_of(
        dataset[index], mask_key)
    return np.bincount(mask.ravel().astype(np.int64),
                       minlength=num_classes)[:num_classes]


class _BincountDataset(torch.utils.data.Dataset):
    """torch Dataset that returns one frame's per-class pixel counts.

    Doing the decode+bincount inside ``__getitem__`` lets a DataLoader fan the
    work out across ``num_workers`` processes for the one-time stats pass, and
    the fixed-size ``(num_classes,)`` result batches with the default collate.
    """

    def __init__(self, dataset, num_classes: int, mask_key: str):
        self.dataset = dataset
        self.num_classes = num_classes
        self.mask_key = mask_key
        self.loader = getattr(dataset, "load_mask", None)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        return torch.from_numpy(_frame_bincount(
            self.dataset, index, self.num_classes, self.mask_key, self.loader))


def _iter_bincounts(dataset, num_classes: int, mask_key: str, num_workers: int):
    """Yield each frame's per-class bincount (in dataset order), with progress.

    ``num_workers > 0`` parallelizes the mask decode across worker processes via
    a DataLoader (order preserved, so sampler weights stay aligned); otherwise it
    runs a simple in-process loop.
    """
    n = len(dataset)
    start = time.time()
    log_every = max(1, n // 10)  # ~10 progress lines over the pass

    def tick(done: int) -> None:
        if done % log_every == 0 or done == n:
            print(f"  [class-stats] {done}/{n} masks "
                  f"({time.time() - start:.0f}s)", flush=True)

    if num_workers and num_workers > 0:
        from torch.utils.data import DataLoader

        loader = DataLoader(
            _BincountDataset(dataset, num_classes, mask_key),
            batch_size=64, num_workers=num_workers, shuffle=False)
        done = 0
        for batch in loader:  # (B, num_classes) int64
            for row in batch.numpy():
                yield row
                done += 1
                tick(done)
    else:
        load_mask = getattr(dataset, "load_mask", None)
        for i in range(n):
            yield _frame_bincount(dataset, i, num_classes, mask_key, load_mask)
            tick(i + 1)


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
                        mask_key: str = "mask", num_workers: int = 0):
    """Per-class frequencies AND per-frame sampler weights in a *single* pass.

    Returns ``(frequencies, sample_weights)``:
        frequencies   - float64 (num_classes,), normalized pixel frequencies.
        sample_weights - float64 (len(dataset),), each frame weighted by the
            inverse-sqrt frequency of the rarest class it contains (so frames
            with rare classes are oversampled).

    This fuses :func:`compute_pixel_frequencies` and the weighting loop of
    :func:`make_weighted_sampler` so the dataset (whose items are decoded images)
    is iterated once instead of twice -- the expensive part on large datasets.
    ``num_workers > 0`` fans the per-frame mask decode out across worker processes
    (order preserved), cutting the one-time pass roughly by that factor.
    """
    counts = np.zeros(num_classes, dtype=np.int64)
    present_per_frame: list[np.ndarray] = []
    for binc in _iter_bincounts(dataset, num_classes, mask_key, num_workers):
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
                                mask_key: str = "mask", num_workers: int = 0):
    """:func:`compute_class_stats` with an on-disk ``.npz`` cache.

    The full pass over the train set (decoding thousands of masks) takes many
    minutes; this caches the result so re-runs and subsequent models start
    instantly. The cache is reused only when its ``version``, ``sample_weights``
    length and ``frequencies`` size match the current dataset -- the caller is
    responsible for keying ``cache_path`` on anything else that changes the split
    (e.g. the split seed). Because the default split is deterministic, a generated
    ``.npz`` can be committed/shared so the *first* run skips the pass entirely.
    ``num_workers`` is forwarded to parallelize the one-time computation.
    """
    if cache_path and os.path.exists(cache_path):
        try:
            cached = np.load(cache_path)
            freqs, weights = cached["frequencies"], cached["sample_weights"]
            version = int(cached["version"]) if "version" in cached else 1
            if (version == _STATS_CACHE_VERSION
                    and weights.shape[0] == len(dataset)
                    and freqs.shape[0] == num_classes):
                print(f"[class-stats] loaded cache {cache_path}", flush=True)
                return freqs, weights
        except Exception:
            pass  # corrupt/stale cache -> recompute below

    print(f"[class-stats] computing over {len(dataset)} train frames "
          "(one-time; cached afterwards)...", flush=True)
    frequencies, sample_weights = compute_class_stats(
        dataset, num_classes, mask_key, num_workers=num_workers)
    if cache_path:
        directory = os.path.dirname(cache_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        np.savez(cache_path, frequencies=frequencies, sample_weights=sample_weights,
                 version=_STATS_CACHE_VERSION)
        print(f"[class-stats] cached -> {cache_path}", flush=True)
    return frequencies, sample_weights


def sampler_from_weights(sample_weights) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler from precomputed per-frame weights."""
    weights = torch.as_tensor(np.asarray(sample_weights), dtype=torch.double)
    return WeightedRandomSampler(weights=weights, num_samples=weights.numel(),
                                 replacement=True)


def window_sample_weights(windows, frame_sample_weights) -> np.ndarray:
    """Per-clip sampler weights for the temporal (window) path.

    The frame-level WeightedRandomSampler does not apply to clip-indexed
    temporal training, so the only model trained so far saw rare classes (e.g.
    ``cystic_duct``) at their natural <0.1% prevalence. This rebuilds the same
    oversampling at clip granularity: each window is weighted by its *target*
    (last) frame's weight, since that is the frame the temporal model predicts.
    Clips whose predicted frame contains a rare class are therefore drawn more
    often, exactly as the per-frame sampler oversamples rare-class frames.

    ``windows`` is the list of local-index tuples built by
    :class:`~src.data.cholecseg8k.CholecSeg8kWindowDataset` (its ``_windows``);
    its local indices line up with ``frame_sample_weights``, which is computed
    over the *same split and seed* in frame order. Returns a float64 array of
    one weight per window, ready for :func:`sampler_from_weights`.
    """
    weights = np.asarray(frame_sample_weights, dtype=np.float64)
    return np.array([weights[window[-1]] for window in windows],
                    dtype=np.float64)
