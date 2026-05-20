"""CholecSeg8k dataset loader and the 13 -> 6 class mapping.

CholecSeg8k contains 8,080 frames drawn from 17 Cholec80 video clips, each
annotated with 13 semantic classes. We merge them into 6 CVS-relevant target
classes.

Original 13 CholecSeg8k classes:
     1  Black Background          8  Blood
     2  Abdominal Wall            9  Cystic Duct
     3  Liver                    10  L-hook Electrocautery
     4  Gastrointestinal Tract   11  Gallbladder
     5  Fat                      12  Hepatic Vein
     6  Grasper                  13  Liver Ligament
     7  Connective Tissue

6-class target schema (index -> name):
     0  background      3  cystic_duct
     1  liver           4  cystic_artery
     2  gallbladder     5  tool

13 -> 6 mapping (decision: strict 6 classes, no fabricated labels):
    background    <- Black Background, Abdominal Wall, Gastrointestinal Tract,
                     Fat, Connective Tissue, Blood
    liver         <- Liver, Hepatic Vein, Liver Ligament
    gallbladder   <- Gallbladder
    cystic_duct   <- Cystic Duct
    cystic_artery <- (NONE) CholecSeg8k has no cystic-artery label. This class
                     receives 0 pixels from CholecSeg8k and is learned instead
                     from Endoscapes2023, which does annotate the cystic artery.
    tool          <- Grasper, L-hook Electrocautery

Splits are video-level (12 train / 2 val / 3 test of 17 videos) to prevent
frame leakage.

The HF dataset (``minwoosun/CholecSeg8k``) exposes
``image`` / ``color_mask`` / ``watershed_mask`` / ``annotation_mask`` and
**no video-id column**; the per-frame video id is recovered from the image
file path (:meth:`CholecSeg8kDataset._video_ids_from_image_paths`). The
:data:`CHOLECSEG8K_COLOR_MAP` palette below is still best-effort -- verify it
against the real masks in ``notebooks/01_data_exploration.ipynb`` and correct
any mismatches.
"""
from __future__ import annotations

import random
import re

import numpy as np
import torch
from torch.utils.data import Dataset

NUM_CLASSES: int = 6
CLASS_NAMES: tuple[str, ...] = (
    "background",
    "liver",
    "gallbladder",
    "cystic_duct",
    "cystic_artery",
    "tool",
)

# Canonical order of the 13 original CholecSeg8k classes.
CHOLECSEG8K_13_CLASSES: tuple[str, ...] = (
    "Black Background",
    "Abdominal Wall",
    "Liver",
    "Gastrointestinal Tract",
    "Fat",
    "Grasper",
    "Connective Tissue",
    "Blood",
    "Cystic Duct",
    "L-hook Electrocautery",
    "Gallbladder",
    "Hepatic Vein",
    "Liver Ligament",
)

# 13-class name -> 6-class target index (see module docstring for rationale).
NAME13_TO_IDX6: dict[str, int] = {
    "Black Background": 0,
    "Abdominal Wall": 0,
    "Gastrointestinal Tract": 0,
    "Fat": 0,
    "Connective Tissue": 0,
    "Blood": 0,
    "Liver": 1,
    "Hepatic Vein": 1,
    "Liver Ligament": 1,
    "Gallbladder": 2,
    "Cystic Duct": 3,
    # index 4 == "cystic_artery": no CholecSeg8k class maps here.
    "Grasper": 5,
    "L-hook Electrocautery": 5,
}

# CholecSeg8k color-mask RGB triple -> 13-class name.
# TODO: verify against the real masks. The pixel-distribution cell in
# notebooks/01_data_exploration.ipynb surfaces any mismatch -- correct this
# table if the colors differ.
CHOLECSEG8K_COLOR_MAP: dict[tuple[int, int, int], str] = {
    (127, 127, 127): "Black Background",
    (210, 140, 140): "Abdominal Wall",
    (255, 114, 114): "Liver",
    (231, 70, 156): "Gastrointestinal Tract",
    (186, 183, 75): "Fat",
    (170, 255, 0): "Grasper",
    (255, 85, 0): "Connective Tissue",
    (255, 0, 0): "Blood",
    (255, 255, 0): "Cystic Duct",
    (169, 255, 184): "L-hook Electrocautery",
    (255, 160, 165): "Gallbladder",
    (0, 50, 128): "Hepatic Vein",
    (111, 74, 0): "Liver Ligament",
}


def remap_color_mask(color_mask) -> np.ndarray:
    """Map an (H, W, 3) RGB CholecSeg8k color mask -> (H, W) 6-class indices.

    Colors absent from ``CHOLECSEG8K_COLOR_MAP`` fall back to background (0).
    """
    arr = np.asarray(color_mask)[..., :3].astype(np.int32)
    out = np.zeros(arr.shape[:2], dtype=np.uint8)
    for rgb, name13 in CHOLECSEG8K_COLOR_MAP.items():
        idx6 = NAME13_TO_IDX6[name13]
        if idx6 == 0:
            continue  # background is already the default
        match = np.all(arr == np.array(rgb, dtype=np.int32), axis=-1)
        out[match] = idx6
    return out


def remap_index_mask(index_mask) -> np.ndarray:
    """Map an (H, W) single-channel mask of 0..12 indices -> 6-class indices.

    Assumes indices follow ``CHOLECSEG8K_13_CLASSES`` order.  # TODO: verify.
    """
    arr = np.asarray(index_mask).astype(np.int64)
    lut_size = max(len(CHOLECSEG8K_13_CLASSES), int(arr.max(initial=0)) + 1)
    lut = np.zeros(lut_size, dtype=np.uint8)
    for i, name13 in enumerate(CHOLECSEG8K_13_CLASSES):
        lut[i] = NAME13_TO_IDX6[name13]
    return lut[np.clip(arr, 0, lut_size - 1)]


def remap_mask(raw_mask) -> np.ndarray:
    """Remap a raw CholecSeg8k mask to 6-class indices.

    Dispatches to :func:`remap_color_mask` for RGB masks and
    :func:`remap_index_mask` for single-channel index masks.
    """
    arr = np.asarray(raw_mask)
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        return remap_color_mask(arr)
    if arr.ndim == 2:
        return remap_index_mask(arr)
    raise ValueError(f"Unexpected mask shape {arr.shape}; expected (H,W) or (H,W,3).")


def _video_id_from_path(path) -> str:
    """Extract a 'videoNN' identifier from a path/filename string."""
    match = re.search(r"video[_-]?\d+", str(path), flags=re.IGNORECASE)
    return match.group(0).lower() if match else str(path)


def make_video_level_split(video_ids, n_train: int = 12, n_val: int = 2,
                           n_test: int = 3, seed: int = 42) -> dict[str, set]:
    """Partition unique video IDs into train/val/test sets (no frame leakage).

    With the expected 17 CholecSeg8k videos this yields a 12/2/3 split; for any
    other video count the sizes are scaled proportionally.
    """
    unique = sorted({str(v) for v in video_ids})
    total = n_train + n_val + n_test
    if len(unique) != total and len(unique) >= 3:
        n_train = max(1, round(len(unique) * n_train / total))
        n_val = max(1, round(len(unique) * n_val / total))
        n_test = max(1, len(unique) - n_train - n_val)

    shuffled = unique[:]
    random.Random(seed).shuffle(shuffled)
    return {
        "train": set(shuffled[:n_train]),
        "val": set(shuffled[n_train:n_train + n_val]),
        "test": set(shuffled[n_train + n_val:n_train + n_val + n_test]),
    }


def load_cholecseg8k(hf_repo: str = "minwoosun/CholecSeg8k",
                     cache_dir: str = "./data/cholecseg8k"):
    """Load CholecSeg8k via the HuggingFace ``datasets`` loader.

    Returns a ``datasets.Dataset`` / ``DatasetDict``. Requires network access
    to the HuggingFace Hub on first call; results are cached in ``cache_dir``.
    """
    from datasets import load_dataset  # lazy import: keeps this module offline-importable

    return load_dataset(hf_repo, trust_remote_code=True, cache_dir=cache_dir)


class CholecSeg8kDataset(Dataset):
    """torch Dataset over CholecSeg8k with the 13 -> 6 class remapping.

    Each item is a dict with ``image`` (float tensor, C x H x W) and ``mask``
    (long tensor, H x W of 6-class indices). When a ``transform`` is supplied it
    must be an albumentations pipeline (see ``src.data.transforms``).

    NOTE: the HF dataset column layout could not be verified offline. The
    ``image_column`` / ``mask_column`` / ``video_id_column`` defaults are
    best-effort — confirm them against the dataset card.  # TODO: verify schema.
    """

    def __init__(
        self,
        split: str = "train",
        image_size: int = 512,
        transform=None,
        hf_repo: str = "minwoosun/CholecSeg8k",
        cache_dir: str = "./data/cholecseg8k",
        hf_split: str = "train",
        image_column: str = "image",
        mask_column: str = "color_mask",
        video_id_column: str = "video_id",
        split_seed: int = 42,
    ):
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train'/'val'/'test', got {split!r}")

        self.split = split
        self.image_size = image_size
        self.transform = transform
        self.image_column = image_column
        self.mask_column = mask_column

        dataset = load_cholecseg8k(hf_repo, cache_dir)
        # Unwrap a DatasetDict to the requested HF split if necessary.
        if hasattr(dataset, "keys") and hf_split in dataset:
            dataset = dataset[hf_split]
        self._hf = dataset

        video_ids = self._extract_video_ids(video_id_column)
        keep = make_video_level_split(video_ids, seed=split_seed)[split]
        self._indices = [i for i, v in enumerate(video_ids) if str(v) in keep]

    def _extract_video_ids(self, video_id_column: str) -> list[str]:
        """Return a per-frame video id, used for the video-level split."""
        columns = list(getattr(self._hf, "column_names", []) or [])
        if video_id_column in columns:
            return [str(v) for v in self._hf[video_id_column]]
        for candidate in ("path", "file_name", "filename", "image_path"):
            if candidate in columns:
                return [_video_id_from_path(p) for p in self._hf[candidate]]
        # minwoosun/CholecSeg8k exposes no video-id column; recover the id
        # from the image feature's stored file path (.../videoNN/...).
        if self.image_column in columns:
            ids = self._video_ids_from_image_paths()
            if ids is not None:
                return ids
        raise KeyError(
            "No video-id column found for the video-level split, and no "
            f"'videoNN' marker in the image paths. Available columns: "
            f"{columns}. Pass video_id_column=... explicitly."
        )

    def _video_ids_from_image_paths(self) -> list[str] | None:
        """Recover per-frame video ids from the image feature's file paths.

        Returns one ``videoNN`` id per frame, or ``None`` if the paths carry
        no usable video marker.
        """
        try:
            import pyarrow.compute as pc
            from datasets import Image as HfImage

            undecoded = self._hf.cast_column(
                self.image_column, HfImage(decode=False))
            paths = pc.struct_field(
                undecoded.data.column(self.image_column), "path").to_pylist()
        except Exception:
            return None
        if not paths or any(not path for path in paths):
            return None
        ids = [_video_id_from_path(path) for path in paths]
        if all(re.fullmatch(r"video[_-]?\d+", vid) for vid in ids):
            return ids
        return None

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int) -> dict:
        record = self._hf[self._indices[index]]
        image = np.asarray(record[self.image_column].convert("RGB"), dtype=np.uint8)
        mask = remap_mask(np.asarray(record[self.mask_column]))

        if self.transform is not None:
            out = self.transform(image=image, mask=mask)
            image_t, mask_t = out["image"], out["mask"]
        else:
            image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask_t = torch.from_numpy(mask)

        return {"image": image_t, "mask": mask_t.long()}
