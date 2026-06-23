"""Tests for the Endoscapes2023 semantic-segmentation id remapping.

The mapping is the failure-prone part (the CholecSeg8k color-map lesson), so it
is covered with numpy only -- no PIL, no dataset download. Frame/mask file I/O
is a thin PIL wrapper exercised in CI / notebook 09 against the real data.
"""
import numpy as np

from src.data.cholecseg8k import CLASS_NAMES
from src.data.endoscapes_seg import (
    ENDOSCAPES_NAME_TO_IDX6,
    ENDOSCAPES_SEMSEG_ID_TO_NAME,
    remap_semseg,
    semseg_lut,
)

DUCT = CLASS_NAMES.index("cystic_duct")
ARTERY = CLASS_NAMES.index("cystic_artery")
GB = CLASS_NAMES.index("gallbladder")
TOOL = CLASS_NAMES.index("tool")
BG = CLASS_NAMES.index("background")


def test_lut_maps_each_endoscapes_id_to_expected_class():
    """Every documented Endoscapes id lands on the right 6-class index."""
    lut = semseg_lut()
    # 0 bg, 1 cystic_plate, 2 calot_triangle, 3 artery, 4 duct, 5 gb, 6 tool
    assert lut[0] == BG
    assert lut[1] == BG        # cystic_plate -> background (not a target class)
    assert lut[2] == BG        # calot_triangle -> background
    assert lut[3] == ARTERY
    assert lut[4] == DUCT
    assert lut[5] == GB
    assert lut[6] == TOOL


def test_duct_and_artery_are_distinct_real_classes():
    """The two CVS-critical structures map to their own (non-background) ids."""
    assert DUCT != BG and ARTERY != BG and DUCT != ARTERY
    assert ENDOSCAPES_NAME_TO_IDX6["cystic_duct"] == DUCT
    assert ENDOSCAPES_NAME_TO_IDX6["cystic_artery"] == ARTERY


def test_remap_semseg_single_channel():
    """A single-channel id map is remapped element-wise into 6-class indices."""
    raw = np.array([[0, 1, 2], [3, 4, 5], [6, 6, 0]], dtype=np.uint8)
    out = remap_semseg(raw)
    expected = np.array([[BG, BG, BG], [ARTERY, DUCT, GB], [TOOL, TOOL, BG]])
    assert out.shape == raw.shape
    assert np.array_equal(out, expected)
    assert out.dtype == np.uint8


def test_remap_semseg_three_identical_channels():
    """The 480x854x3 PNG form (identical channels) collapses correctly."""
    single = np.array([[4, 3], [5, 6]], dtype=np.uint8)        # duct, artery, gb, tool
    rgb = np.stack([single, single, single], axis=-1)          # (H, W, 3)
    out = remap_semseg(rgb)
    assert out.shape == (2, 2)
    assert np.array_equal(out, np.array([[DUCT, ARTERY], [GB, TOOL]]))


def test_remap_semseg_unknown_id_falls_to_background():
    """An id with no known mapping is absorbed into background, never a real class."""
    raw = np.array([[4, 99], [200, 0]], dtype=np.uint8)
    out = remap_semseg(raw)
    assert out[0, 0] == DUCT
    assert out[0, 1] == BG and out[1, 0] == BG   # 99, 200 -> background
    assert int(out.max()) <= len(CLASS_NAMES) - 1


def test_id_to_name_and_name_to_idx_are_consistent():
    """Every semseg id name is a key in the name->index map."""
    for name in ENDOSCAPES_SEMSEG_ID_TO_NAME.values():
        assert name in ENDOSCAPES_NAME_TO_IDX6
