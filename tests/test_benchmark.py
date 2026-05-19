"""Tests for the segmentation benchmark runner."""
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.eval.benchmark_runner import (
    evaluate_model,
    format_table,
    load_segmentation_checkpoint,
    summarize,
)


class _ConstantSegmenter(nn.Module):
    """Emits one-hot logits of a fixed mask, so its argmax is that mask."""

    def __init__(self, mask):
        super().__init__()
        self.register_buffer("mask", mask)

    def forward(self, x):
        batch = x.shape[0]
        onehot = torch.stack([(self.mask == c).float() for c in range(6)])
        return onehot.unsqueeze(0).expand(batch, -1, -1, -1)


def test_evaluate_model_perfect_prediction():
    """A model whose output equals the target scores per-image mIoU 1.0."""
    mask = torch.randint(0, 6, (16, 16))
    mask[0, 0] = 3  # ensure cystic_duct (class 3) is present -> Dice not NaN
    data = [{"image": torch.rand(3, 16, 16), "mask": mask} for _ in range(5)]
    loader = DataLoader(data, batch_size=2)

    metrics = evaluate_model(_ConstantSegmenter(mask), loader, num_classes=6)

    assert len(metrics["miou"]) == 5
    assert len(metrics["cystic_duct_dice"]) == 5
    assert all(m == pytest.approx(1.0) for m in metrics["miou"])
    assert all(d == pytest.approx(1.0) for d in metrics["cystic_duct_dice"])


def test_summarize_returns_ci_triples():
    """summarize wraps each metric list in a (mean, lo, hi) bootstrap triple."""
    summary = summarize({"miou": [0.5] * 20, "cystic_duct_dice": [0.3] * 20})

    assert set(summary) == {"miou", "cystic_duct_dice"}
    assert len(summary["miou"]) == 3
    assert summary["miou"][0] == pytest.approx(0.5)
    assert summary["cystic_duct_dice"][0] == pytest.approx(0.3)


def test_format_table_renders_values_and_tbd():
    """Numeric summaries render as 'mean (lo-hi)'; NaN summaries render 'TBD'."""
    results = {
        "U-Net": {"miou": (0.7, 0.65, 0.75),
                  "cystic_duct_dice": (0.4, 0.3, 0.5)},
        "SAM2": {"miou": (float("nan"),) * 3,
                 "cystic_duct_dice": (float("nan"),) * 3},
    }
    table = format_table(results)

    assert table.startswith("| Method | mIoU | Cystic Duct Dice | CVS mAP |")
    assert "| U-Net | 0.700 (0.650-0.750) | 0.400 (0.300-0.500) | TBD |" in table
    assert "| SAM2 | TBD | TBD | TBD |" in table


def test_load_segmentation_checkpoint_round_trip(tmp_path):
    """Weights saved under a 'model.' prefix load back into a bare model."""
    trained = nn.Conv2d(3, 6, kernel_size=1)
    checkpoint = {"state_dict": {f"model.{k}": v
                                 for k, v in trained.state_dict().items()}}
    path = tmp_path / "segmentation.ckpt"
    torch.save(checkpoint, path)

    fresh = nn.Conv2d(3, 6, kernel_size=1)
    load_segmentation_checkpoint(fresh, path)

    for key, value in trained.state_dict().items():
        assert torch.equal(fresh.state_dict()[key], value)
