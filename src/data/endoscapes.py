"""Endoscapes2023 CVS frame-level loader.

Endoscapes2023 (201 videos, 11,090 CVS-annotated frames) is the primary CVS
benchmark, distributed by the CAMMA group. Download the official
``endoscapes.zip`` with ``scripts/download_endoscapes.sh`` (CAMMA public mirror)
or from PhysioNet (czwq-jh81).

Each CVS keyframe carries 3 Strasberg criteria, each averaged over 3 annotators
(values in {0, 1/3, 2/3, 1}); a binary label is the majority vote
(:func:`binarize_cvs_criteria`), and the CVS score is their sum in [0, 3]
(:func:`cvs_score`).

On-disk layout (the zip extracts to an ``endoscapes/`` directory):

    <root>/<split>/<vid>_<frame>.jpg   train / val / test frame folders
    <root>/all_metadata.csv            per-frame CVS labels, columns:
                                       vid, frame, avg_cvs, C1, C2, C3,
                                       is_ds_keyframe, ...

``is_ds_keyframe == True`` marks the frames that actually carry CVS
annotations. The train/val/test membership is the folder a frame lives in.
``Endoscapes2023Dataset`` descends into ``<root>/endoscapes/`` automatically.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

CVS_CRITERIA: tuple[str, ...] = (
    "c1_two_structures",
    "c2_triangle_cleared",
    "c3_cystic_plate_exposed",
)
NUM_CVS_CRITERIA: int = len(CVS_CRITERIA)


def binarize_cvs_criteria(values, threshold: float = 0.5) -> tuple[int, ...]:
    """Reduce per-criterion CVS agreement scores to binary {0, 1} labels.

    ``values`` holds one value per criterion: the fraction of annotators
    agreeing the criterion is met (or an already-binary {0, 1} label). A
    criterion counts as achieved when agreement is at least ``threshold``.
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    if values.size != NUM_CVS_CRITERIA:
        raise ValueError(
            f"expected {NUM_CVS_CRITERIA} criteria values, got {values.size}")
    return tuple(int(value >= threshold) for value in values)


def cvs_score(binary_criteria) -> int:
    """CVS score in [0, 3]: the number of criteria achieved."""
    return int(np.asarray(binary_criteria).ravel().sum())


def _resolve_root(root) -> Path:
    """Return the dir holding all_metadata.csv (descend into endoscapes/ if needed)."""
    root = Path(root)
    if (not (root / "all_metadata.csv").is_file()
            and (root / "endoscapes" / "all_metadata.csv").is_file()):
        return root / "endoscapes"
    return root


def _read_cvs_metadata(path, vid_column, frame_column, criteria_columns,
                       keyframes_only) -> dict[str, tuple[float, ...]]:
    """Parse all_metadata.csv into ``{"<vid>_<frame>": (c1, c2, c3)}``."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"CVS metadata not found: {path}. Download Endoscapes2023 with "
            "scripts/download_endoscapes.sh and point the config root at it.")
    labels: dict[str, tuple[float, ...]] = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            if keyframes_only and str(row.get("is_ds_keyframe", "")).strip() != "True":
                continue
            key = f"{int(float(row[vid_column]))}_{int(float(row[frame_column]))}"
            labels[key] = tuple(float(row[column]) for column in criteria_columns)
    return labels


def _load_rgb(path) -> np.ndarray:
    """Load an image file as an (H, W, 3) uint8 RGB array."""
    from PIL import Image  # lazy import: keeps this module offline-importable

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _resize_to_tensor(image: np.ndarray, image_size: int) -> torch.Tensor:
    """Resize an (H, W, 3) uint8 array to a (3, S, S) float tensor in [0, 1]."""
    from PIL import Image

    resized = Image.fromarray(image).resize((image_size, image_size))
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


class Endoscapes2023Dataset(Dataset):
    """torch Dataset over CVS-annotated Endoscapes2023 frames.

    Each item is a dict:
        ``image``      float tensor (3, image_size, image_size), RGB in [0, 1]
        ``criteria``   float tensor (3,) of binary CVS criteria labels
        ``cvs_score``  long scalar tensor, the criteria sum in [0, 3]

    Args:
        root: dataset root (the extracted ``endoscapes.zip`` directory, or its
            parent -- the loader descends into ``endoscapes/`` if needed).
        split: ``train`` / ``val`` / ``test`` (the official frame folders).
        image_size: output square size for the default (no-transform) path.
        transform: optional albumentations image pipeline (owns resizing).
        metadata_file: explicit metadata CSV (default ``<root>/all_metadata.csv``).
        vid_column / frame_column: metadata columns identifying each frame.
        criteria_columns: the 3 CVS criteria columns, in output order.
        keyframes_only: keep only frames with real CVS annotations
            (``is_ds_keyframe == True``).
        threshold: agreement threshold for :func:`binarize_cvs_criteria`.
    """

    def __init__(self, root: str = "./data/endoscapes2023", split: str = "train",
                 image_size: int = 512, transform=None,
                 metadata_file: str | None = None, vid_column: str = "vid",
                 frame_column: str = "frame",
                 criteria_columns=("C1", "C2", "C3"),
                 keyframes_only: bool = True, threshold: float = 0.5):
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train'/'val'/'test', got {split!r}")

        self.root = _resolve_root(root)
        self.split = split
        self.image_size = image_size
        self.transform = transform
        self.threshold = threshold

        self.frames_dir = self.root / split
        if not self.frames_dir.is_dir():
            raise FileNotFoundError(
                f"Endoscapes2023 frames not found: {self.frames_dir}. Download "
                "with scripts/download_endoscapes.sh.")

        metadata_path = (Path(metadata_file) if metadata_file
                         else self.root / "all_metadata.csv")
        labels = _read_cvs_metadata(metadata_path, vid_column, frame_column,
                                    criteria_columns, keyframes_only)

        # Frames present on disk for this split, intersected with CVS labels.
        self._samples: list[tuple[Path, tuple[float, ...]]] = [
            (jpg, labels[jpg.stem])
            for jpg in sorted(self.frames_dir.glob("*.jpg"))
            if jpg.stem in labels
        ]
        if not self._samples:
            raise RuntimeError(
                f"No CVS-labelled frames matched {self.frames_dir} against "
                f"{metadata_path.name}.")

    def __len__(self) -> int:
        return len(self._samples)

    def cvs_labels(self) -> np.ndarray:
        """All binarized CVS criteria as an (N, 3) int array (no image I/O)."""
        return np.array(
            [binarize_cvs_criteria(raw, self.threshold)
             for _, raw in self._samples],
            dtype=np.int64,
        )

    def __getitem__(self, index: int) -> dict:
        path, raw_criteria = self._samples[index]
        criteria = binarize_cvs_criteria(raw_criteria, self.threshold)

        image = _load_rgb(path)
        if self.transform is not None:
            image_t = self.transform(image=image)["image"]
        else:
            image_t = _resize_to_tensor(image, self.image_size)

        return {
            "image": image_t,
            "criteria": torch.tensor(criteria, dtype=torch.float32),
            "cvs_score": torch.tensor(cvs_score(criteria), dtype=torch.long),
        }
