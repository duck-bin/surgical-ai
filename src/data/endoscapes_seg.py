"""Endoscapes2023 semantic-segmentation loader — the source of cystic_duct/artery.

CholecSeg8k effectively does not label the cystic duct (<=3 px across the entire
train split, confirmed in both its color_mask and watershed_mask) and has no
cystic-artery label at all. The two CVS-critical tubular structures are therefore
learned from Endoscapes2023, which annotates both with semantic masks
(Endoscapes-Seg, 1933 frames over the official video-level splits).

On-disk layout (endoscapes.zip extracts to an ``endoscapes/`` dir, descended
into automatically):

    <root>/<split>_seg/<vid>_<frame>.jpg   segmentation frames (Endoscapes-Seg)
    <root>/semseg/<vid>_<frame>.png        480x854x3 mask, each pixel = class_id
                                           (the 3 channels are identical)

Endoscapes semseg class ids (background + the 6 classes, in CAMMA's
``seg_label_map`` / SurgLatentGraph order):

    0 background   1 cystic_plate   2 calot_triangle (hepatocystic triangle)
    3 cystic_artery   4 cystic_duct   5 gallbladder   6 tool

These are remapped into this project's shared 6-class schema (see
``src.data.cholecseg8k.CLASS_NAMES``: background/liver/gallbladder/cystic_duct/
cystic_artery/tool). ``cystic_plate`` and ``calot_triangle`` are not target
classes here and fold into background; ``liver`` is absent from Endoscapes (it
is supplied by CholecSeg8k under joint training).

VERIFY BEFORE TRAINING. The id->name table below is from CAMMA's public docs,
not the bytes on your disk. Run ``semseg_id_histogram`` (or notebook 09) on the
downloaded masks to confirm cystic_duct/cystic_artery are present and land on
the expected ids — the CholecSeg8k color-map mistake is exactly what this check
prevents.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.cholecseg8k import CLASS_NAMES, NUM_CLASSES

# Endoscapes semseg pixel id -> Endoscapes class name.
ENDOSCAPES_SEMSEG_ID_TO_NAME: dict[int, str] = {
    0: "background",
    1: "cystic_plate",
    2: "calot_triangle",
    3: "cystic_artery",
    4: "cystic_duct",
    5: "gallbladder",
    6: "tool",
}

# Endoscapes class name -> this project's 6-class index (CLASS_NAMES order).
# cystic_plate/calot_triangle are not target classes here -> background; liver
# does not exist in Endoscapes.
ENDOSCAPES_NAME_TO_IDX6: dict[str, int] = {
    "background": CLASS_NAMES.index("background"),
    "cystic_plate": CLASS_NAMES.index("background"),
    "calot_triangle": CLASS_NAMES.index("background"),
    "cystic_artery": CLASS_NAMES.index("cystic_artery"),
    "cystic_duct": CLASS_NAMES.index("cystic_duct"),
    "gallbladder": CLASS_NAMES.index("gallbladder"),
    "tool": CLASS_NAMES.index("tool"),
}


def semseg_lut(max_id: int = 6) -> np.ndarray:
    """LUT mapping an Endoscapes semseg id -> this project's 6-class index.

    Sized to ``max(max_id, highest known id)``; ids without a known mapping
    fall to background (0), so an unexpected value never crashes or mislabels
    into a real class -- it is visibly absorbed into background instead.
    """
    size = max(int(max_id), max(ENDOSCAPES_SEMSEG_ID_TO_NAME)) + 1
    lut = np.zeros(size, dtype=np.uint8)
    for sid, name in ENDOSCAPES_SEMSEG_ID_TO_NAME.items():
        lut[sid] = ENDOSCAPES_NAME_TO_IDX6[name]
    return lut


def remap_semseg(mask) -> np.ndarray:
    """Map an Endoscapes semseg mask -> (H, W) uint8 of 6-class indices.

    Accepts the 480x854x3 PNG (3 identical channels) or a single-channel id map.
    """
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[..., 0]  # the 3 channels are identical
    arr = arr.astype(np.int64)
    lut = semseg_lut(int(arr.max(initial=0)))
    return lut[arr]


def _resolve_root(root, masks_subdir: str) -> Path:
    """Return the dir holding ``masks_subdir`` (descend into ``endoscapes/``)."""
    root = Path(root)
    if not (root / masks_subdir).is_dir() and (root / "endoscapes" / masks_subdir).is_dir():
        return root / "endoscapes"
    return root


class EndoscapesSegDataset(Dataset):
    """torch Dataset over Endoscapes2023 semantic-segmentation frames.

    Each item is ``{"image": float (3,S,S), "mask": long (S,S) in [0,5]}`` --
    the same contract as :class:`~src.data.cholecseg8k.CholecSeg8kDataset`, so it
    is a drop-in for ``SegmentationModule`` / the benchmark / copy-paste.

    Args:
        root: dataset root (the extracted ``endoscapes.zip`` dir or its parent).
        split: ``train`` / ``val`` / ``test``.
        image_size: square size for the default (no-transform) path.
        transform: optional albumentations pipeline (owns resize/normalize).
        frames_subdir: frame folder; defaults to ``<split>_seg`` then ``<split>``.
        masks_subdir: semantic-mask folder (default ``semseg``).
        copy_paste: optional rare-class copy-paste (train only), applied to the
            raw frame before ``transform`` (same as the CholecSeg8k path).
    """

    def __init__(self, root: str = "./data/endoscapes2023", split: str = "train",
                 image_size: int = 512, transform=None,
                 frames_subdir: str | None = None, masks_subdir: str = "semseg",
                 copy_paste=None):
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train'/'val'/'test', got {split!r}")
        self.split = split
        self.image_size = image_size
        self.transform = transform
        self.copy_paste = copy_paste

        self.root = _resolve_root(root, masks_subdir)
        self.masks_dir = self.root / masks_subdir
        self.frames_dir = self._resolve_frames_dir(frames_subdir, split)

        if not self.masks_dir.is_dir():
            raise FileNotFoundError(
                f"Endoscapes semseg masks not found: {self.masks_dir}. Download "
                "with scripts/download_endoscapes.sh.")
        # Pair each frame with its mask by the <vid>_<frame> stem.
        self._samples: list[tuple[Path, Path]] = [
            (jpg, self.masks_dir / f"{jpg.stem}.png")
            for jpg in sorted(self.frames_dir.glob("*.jpg"))
            if (self.masks_dir / f"{jpg.stem}.png").is_file()
        ]
        if not self._samples:
            raise RuntimeError(
                f"No frame/mask pairs matched {self.frames_dir} against "
                f"{self.masks_dir}.")

    def _resolve_frames_dir(self, frames_subdir, split) -> Path:
        if frames_subdir is not None:
            return self.root / frames_subdir
        seg_dir = self.root / f"{split}_seg"
        return seg_dir if seg_dir.is_dir() else self.root / split

    def __len__(self) -> int:
        return len(self._samples)

    def _load_raw(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Load one frame as raw ``(image_uint8_HWC, mask_HW)`` (no transform)."""
        from PIL import Image  # lazy: keeps this module offline-importable

        image_path, mask_path = self._samples[index]
        with Image.open(image_path) as image:
            image_np = np.asarray(image.convert("RGB"), dtype=np.uint8)
        with Image.open(mask_path) as mask:
            mask_np = remap_semseg(np.asarray(mask))
        return image_np, mask_np

    def load_mask(self, index: int) -> np.ndarray:
        """Decode *only* the mask PNG for one frame, remapped to 6-class indices.

        Skips the RGB image decode so the class-stats pass (which only counts
        mask pixels) is roughly twice as fast on first run.
        """
        from PIL import Image  # lazy: keeps this module offline-importable

        _, mask_path = self._samples[index]
        with Image.open(mask_path) as mask:
            return remap_semseg(np.asarray(mask))

    def __getitem__(self, index: int) -> dict:
        image, mask = self._load_raw(index)
        if self.copy_paste is not None:
            image, mask = self.copy_paste(image, mask)

        if self.transform is not None:
            out = self.transform(image=image, mask=mask)
            image_t, mask_t = out["image"], out["mask"]
        else:
            image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask_t = torch.from_numpy(mask)
        return {"image": image_t, "mask": mask_t.long()}


def semseg_id_histogram(masks_dir, sample: int | None = 200) -> dict:
    """Count raw Endoscapes semseg ids over the mask PNGs (a verification aid).

    Returns ``{raw_id: pixel_count}``. Cross-check the keys against
    :data:`ENDOSCAPES_SEMSEG_ID_TO_NAME` and confirm cystic_duct (4) and
    cystic_artery (3) are present BEFORE training -- this is the empirical check
    that the id mapping matches the bytes on disk.
    """
    from PIL import Image

    paths = sorted(Path(masks_dir).glob("*.png"))
    if sample is not None:
        paths = paths[:: max(1, len(paths) // sample)] if paths else paths
    counts: Counter = Counter()
    for path in paths:
        with Image.open(path) as mask:
            arr = np.asarray(mask)
        arr = arr[..., 0] if arr.ndim == 3 else arr
        ids, n = np.unique(arr, return_counts=True)
        for i, c in zip(ids, n):
            counts[int(i)] += int(c)
    return dict(sorted(counts.items()))
