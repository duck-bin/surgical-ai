"""Tests for the segmentation dataset factory (loader selection).

Only the branching that does NOT touch disk is exercised: the temporal-support
flag, the class-stats cache key, and the two guard ValueErrors. Building a real
dataset needs the downloaded data, so it is left to integration/CI on the data.

``train_segmentation`` imports hydra at module load, so the whole file is skipped
where hydra/omegaconf are absent (e.g. a minimal CPU container).
"""
import pytest

pytest.importorskip("hydra")
pytest.importorskip("omegaconf")

from omegaconf import OmegaConf  # noqa: E402

from src.train.train_segmentation import _make_dataset_fn  # noqa: E402


def test_endoscapes_branch_is_frame_only():
    """Endoscapes-Seg is frame-level: no temporal support, its own stats key."""
    make, supports_temporal, stats_key = _make_dataset_fn(OmegaConf.create(
        {"data": {"name": "endoscapes2023_seg", "loader": "endoscapes2023_seg",
                  "root": "./x", "image_size": 512}}))
    assert supports_temporal is False
    assert stats_key == "endoscapes2023_seg"
    with pytest.raises(ValueError):           # temporal window rejected, no disk I/O
        make("train", None, window=3)


def test_cholecseg8k_branch_supports_temporal_and_keys_by_seed():
    make, supports_temporal, stats_key = _make_dataset_fn(OmegaConf.create(
        {"data": {"name": "cholecseg8k", "hf_repo": "r", "cache_dir": "./c",
                  "image_size": 512, "split": {"seed": 7}}}))
    assert supports_temporal is True
    assert stats_key == "cholecseg8k_seed7"


def test_joint_branch_is_frame_only_and_keys_by_seed():
    """Joint CholecSeg8k+Endoscapes-Seg: frame-level, stats key namespaced by seed."""
    make, supports_temporal, stats_key = _make_dataset_fn(OmegaConf.create(
        {"data": {"name": "joint_seg", "loader": "joint_seg", "image_size": 512,
                  "cholecseg8k": {"hf_repo": "r", "cache_dir": "./c",
                                  "split": {"seed": 42}},
                  "endoscapes": {"root": "./e"}}}))
    assert supports_temporal is False
    assert stats_key == "joint_seg_seed42"
    with pytest.raises(ValueError):           # temporal window rejected, no disk I/O
        make("train", None, window=3)


def test_unknown_loader_raises():
    with pytest.raises(ValueError):
        _make_dataset_fn(OmegaConf.create({"data": {"name": "mystery"}}))
