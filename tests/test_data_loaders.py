"""Tests for dataset loaders and the data layer.

The CholecSeg8k mask remapping, video-level split, and augmentation pipelines
are tested with synthetic inputs (no dataset download required). End-to-end
loader I/O is exercised in the EDA notebook against the real dataset.
"""
import numpy as np
import pytest
import torch

from src.data.cholecseg8k import (
    CHOLECSEG8K_13_CLASSES,
    CHOLECSEG8K_COLOR_MAP,
    NAME13_TO_IDX6,
    make_video_level_split,
    remap_color_mask,
    remap_index_mask,
    remap_mask,
)


def test_remap_color_mask_known_colors():
    """Each known RGB color maps to its expected 6-class index; unknowns -> 0."""
    colors = list(CHOLECSEG8K_COLOR_MAP.keys()) + [(1, 2, 3)]  # last = unknown
    mask_rgb = np.array([colors], dtype=np.uint8)  # shape (1, N, 3)

    out = remap_color_mask(mask_rgb)

    assert out.shape == (1, len(colors))
    assert out.dtype == np.uint8
    assert out.min() >= 0 and out.max() <= 5
    for j, rgb in enumerate(colors[:-1]):
        assert out[0, j] == NAME13_TO_IDX6[CHOLECSEG8K_COLOR_MAP[rgb]]
    assert out[0, -1] == 0  # unrecognized color falls back to background


def test_remap_index_mask_follows_canonical_order():
    """Single-channel indices 0..12 map per CHOLECSEG8K_13_CLASSES order."""
    idx = np.arange(len(CHOLECSEG8K_13_CLASSES), dtype=np.uint8).reshape(1, -1)

    out = remap_index_mask(idx)

    assert out.shape == idx.shape
    for i, name in enumerate(CHOLECSEG8K_13_CLASSES):
        assert out[0, i] == NAME13_TO_IDX6[name]


def test_remap_mask_dispatch_and_range():
    """remap_mask handles both RGB and single-channel masks; output in [0, 5]."""
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    idx = np.zeros((8, 8), dtype=np.uint8)
    for remapped in (remap_mask(rgb), remap_mask(idx)):
        assert remapped.shape == (8, 8)
        assert remapped.min() >= 0 and remapped.max() <= 5


def test_video_level_split_no_leakage():
    """17 videos split 12/2/3 into disjoint sets covering every video."""
    video_ids = [f"video{i:02d}" for i in range(1, 18)]
    split = make_video_level_split(video_ids, seed=42)

    assert len(split["train"]) == 12
    assert len(split["val"]) == 2
    assert len(split["test"]) == 3
    assert not (split["train"] & split["val"])
    assert not (split["train"] & split["test"])
    assert not (split["val"] & split["test"])
    assert split["train"] | split["val"] | split["test"] == set(video_ids)


def test_video_level_split_is_deterministic():
    """The same seed produces the same partition."""
    ids = [f"video{i:02d}" for i in range(1, 18)]
    assert make_video_level_split(ids, seed=42) == make_video_level_split(ids, seed=42)


def test_train_transforms_output_shapes():
    """Train pipeline yields a (3,512,512) float image and (512,512) mask."""
    pytest.importorskip("albumentations")
    from src.data.transforms import build_train_transforms

    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(480, 854, 3), dtype=np.uint8)
    mask = rng.integers(0, 6, size=(480, 854), dtype=np.uint8)

    out = build_train_transforms(image_size=512)(image=image, mask=mask)
    img_t, mask_t = out["image"], out["mask"]

    assert tuple(img_t.shape) == (3, 512, 512)
    assert tuple(mask_t.shape) == (512, 512)
    assert torch.is_floating_point(img_t)
    assert not torch.isnan(img_t).any()
    assert int(mask_t.min()) >= 0 and int(mask_t.max()) <= 5


def test_eval_transforms_are_deterministic():
    """Eval pipeline has no randomness and produces fixed-size tensors."""
    pytest.importorskip("albumentations")
    from src.data.transforms import build_eval_transforms

    rng = np.random.default_rng(1)
    image = rng.integers(0, 256, size=(480, 854, 3), dtype=np.uint8)
    mask = rng.integers(0, 6, size=(480, 854), dtype=np.uint8)

    transform = build_eval_transforms(image_size=512)
    first = transform(image=image, mask=mask)
    second = transform(image=image, mask=mask)

    assert torch.equal(first["image"], second["image"])
    assert tuple(first["image"].shape) == (3, 512, 512)
    assert tuple(first["mask"].shape) == (512, 512)


def test_endoscapes_cvs_labels_binary():
    """CVS criteria binarize to {0, 1}; the CVS score lies in [0, 3]."""
    from src.data.endoscapes import binarize_cvs_criteria, cvs_score

    # Fractional 3-annotator agreement -> majority-vote binary labels.
    criteria = binarize_cvs_criteria([1.0, 0.33, 0.67], threshold=0.5)
    assert criteria == (1, 0, 1)
    assert all(c in (0, 1) for c in criteria)
    assert cvs_score(criteria) == 2

    assert cvs_score(binarize_cvs_criteria([0.0, 0.0, 0.0])) == 0
    assert cvs_score(binarize_cvs_criteria([1.0, 1.0, 1.0])) == 3
    for raw in ([0.2, 0.8, 0.5], [1, 0, 1], [0.49, 0.51, 1.0]):
        assert 0 <= cvs_score(binarize_cvs_criteria(raw)) <= 3


def test_endoscapes_dataset_reads_frames_and_labels(tmp_path):
    """The dataset pairs on-disk frames with their CVS CSV annotations."""
    from PIL import Image

    from src.data.endoscapes import Endoscapes2023Dataset

    frames_dir = tmp_path / "train"
    frames_dir.mkdir()
    for name in ("v1_1.jpg", "v1_2.jpg"):
        Image.new("RGB", (64, 48), color=(120, 30, 30)).save(frames_dir / name)
    (tmp_path / "train_cvs.csv").write_text(
        "frame,c1_two_structures,c2_triangle_cleared,c3_cystic_plate_exposed\n"
        "v1_1.jpg,1.0,0.0,1.0\n"
        "v1_2.jpg,0.33,0.67,1.0\n"
    )

    dataset = Endoscapes2023Dataset(root=tmp_path, split="train", image_size=32)

    assert len(dataset) == 2
    item = dataset[0]
    assert tuple(item["image"].shape) == (3, 32, 32)
    assert torch.is_floating_point(item["image"])
    assert tuple(item["criteria"].shape) == (3,)
    assert set(item["criteria"].tolist()) <= {0.0, 1.0}
    assert 0 <= int(item["cvs_score"]) <= 3
    assert int(item["cvs_score"]) == int(item["criteria"].sum())


def test_video_id_from_path_extracts_video_number():
    """_video_id_from_path pulls 'videoNN' from a CholecSeg8k-style file path."""
    from src.data.cholecseg8k import _video_id_from_path

    assert _video_id_from_path(
        "/cache/datasets/extracted/x/video01/frame_0.png") == "video01"
    assert _video_id_from_path("/x/y/video_12/frame.png") == "video_12"
    # No videoNN marker -> the loader treats this as an unusable result.
    assert _video_id_from_path("/no/marker/here.png") == "/no/marker/here.png"
