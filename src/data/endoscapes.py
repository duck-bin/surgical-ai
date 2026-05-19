"""Endoscapes2023 CVS frame-level loader.

Endoscapes2023 (201 videos, 11,090 CVS-annotated frames) is the primary CVS
benchmark. It requires PhysioNet credentialed access (czwq-jh81); data is
expected to be downloaded manually to ``./data/endoscapes2023/``.

Each annotated frame carries 3 binary Strasberg "Critical View of Safety"
criteria, each rated by 3 annotators. Per-criterion agreement is reduced to a
binary label by majority vote (:func:`binarize_cvs_criteria`); the CVS score is
their sum in [0, 3] (:func:`cvs_score`).

VERIFICATION NEEDED — the dataset is credentialed-access only, so the on-disk
layout assumed below is best-effort and MUST be confirmed against the real
download (Mascagni et al., Scientific Data 2024). The frame directory and the
CVS annotation CSV (path / column names) are configurable on
``Endoscapes2023Dataset`` so the defaults can be corrected without code changes:

    <root>/<split>/*.jpg          official video-level train/val/test frames
    <root>/<split>_cvs.csv        per-frame CVS annotations: a `frame` column
                                  plus one column per criterion (fraction of
                                  annotators agreeing, or a {0,1} label)
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


def _read_cvs_annotations(path, frame_column: str,
                          criteria_columns) -> dict[str, tuple[float, ...]]:
    """Parse the CVS annotation CSV into ``{frame_filename: raw criteria}``."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"CVS annotation file not found: {path}. Pass "
            "cvs_annotation_file=... if the dataset layout differs.")
    annotations: dict[str, tuple[float, ...]] = {}
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            annotations[row[frame_column]] = tuple(
                float(row[column]) for column in criteria_columns)
    return annotations


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
        root: dataset root directory.
        split: ``train`` / ``val`` / ``test`` (the official video-level splits,
            which Endoscapes ships as separate frame directories).
        image_size: output square size for the default (no-transform) path.
        transform: optional albumentations image pipeline; when given it
            receives the native-resolution RGB array and owns the resizing.
        cvs_annotation_file: explicit CVS CSV path (default ``<root>/<split>_cvs.csv``).
        frame_column: CSV column holding the frame filename.
        criteria_columns: CSV columns holding the 3 criteria, in output order.
        threshold: agreement threshold for :func:`binarize_cvs_criteria`.
    """

    def __init__(self, root: str = "./data/endoscapes2023", split: str = "train",
                 image_size: int = 512, transform=None,
                 cvs_annotation_file: str | None = None,
                 frame_column: str = "frame",
                 criteria_columns=CVS_CRITERIA, threshold: float = 0.5):
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train'/'val'/'test', got {split!r}")

        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.transform = transform
        self.threshold = threshold

        self.frames_dir = self.root / split
        if not self.frames_dir.is_dir():
            raise FileNotFoundError(
                f"Endoscapes2023 frames not found: {self.frames_dir}. Download "
                "the dataset from PhysioNet (czwq-jh81) to the configured root.")

        annotation_path = (Path(cvs_annotation_file) if cvs_annotation_file
                           else self.root / f"{split}_cvs.csv")
        annotations = _read_cvs_annotations(
            annotation_path, frame_column, criteria_columns)

        # Index only the annotated frames that are actually present on disk.
        self._samples: list[tuple[Path, tuple[float, ...]]] = [
            (self.frames_dir / name, raw)
            for name, raw in sorted(annotations.items())
            if (self.frames_dir / name).is_file()
        ]
        if not self._samples:
            raise RuntimeError(
                f"No annotated frames from {annotation_path.name} matched "
                f"files under {self.frames_dir}.")

    def __len__(self) -> int:
        return len(self._samples)

    def cvs_labels(self) -> np.ndarray:
        """All binarized CVS criteria as an (N, 3) int array (no image I/O).

        Useful for class-balance statistics (e.g. BCE ``pos_weight``).
        """
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
