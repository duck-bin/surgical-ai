"""Tests for dataset loaders: shape, dtype, value range, no NaN, class sanity.

Implemented alongside the loaders (Step 2: CholecSeg8k, Step 6: Endoscapes2023).
"""
import pytest


@pytest.mark.skip(reason="CholecSeg8k loader implemented in Step 2")
def test_cholecseg8k_sample_shapes():
    """Frame is (3, 512, 512) float; mask is (512, 512) int in [0, 5]; no NaN."""
    raise NotImplementedError


@pytest.mark.skip(reason="CholecSeg8k loader implemented in Step 2")
def test_cholecseg8k_class_distribution_sane():
    """No out-of-range label indices; expected classes appear in the dataset."""
    raise NotImplementedError


@pytest.mark.skip(reason="Endoscapes2023 loader implemented in Step 6")
def test_endoscapes_cvs_labels_binary():
    """CVS criteria labels are binary {0, 1}; the CVS score lies in [0, 3]."""
    raise NotImplementedError
