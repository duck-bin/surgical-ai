"""Metric tests plus a real reproducibility check for utils/seeds.

The known-input metric checks (IoU, Dice, AUROC) are implemented in Step 3 / 7.
The seeds tests run now so CI exercises real code rather than only skips.
"""
import numpy as np
import pytest
import torch

from src.utils.seeds import seed_everything


def test_seed_everything_is_reproducible():
    """Identical seeds produce identical NumPy and torch random draws."""
    seed_everything(123)
    np_a, torch_a = np.random.rand(8), torch.rand(8)

    seed_everything(123)
    np_b, torch_b = np.random.rand(8), torch.rand(8)

    assert np.allclose(np_a, np_b)
    assert torch.allclose(torch_a, torch_b)


def test_seed_everything_returns_seed():
    """seed_everything echoes back the seed it applied."""
    assert seed_everything(7) == 7


@pytest.mark.skip(reason="IoU / Dice metrics implemented in Step 3")
def test_iou_known_input():
    """IoU on a hand-computed mask pair matches the expected value."""
    raise NotImplementedError


@pytest.mark.skip(reason="CVS metrics implemented in Step 7")
def test_auroc_known_input():
    """AUROC on a hand-computed score set matches the expected value."""
    raise NotImplementedError
