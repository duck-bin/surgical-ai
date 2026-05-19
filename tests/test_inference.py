"""Tests for the inference pipeline, visualization helpers, and the demo."""
import numpy as np
import pytest
import torch
import torch.nn as nn

from src.inference.pipeline import CVSPipeline
from src.utils.viz import CLASS_COLORS, cvs_panel, overlay_mask


def test_overlay_mask_keeps_background_colors_foreground():
    """Background pixels are untouched; non-background pixels are recolored."""
    image = np.full((8, 8, 3), 40, dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.int64)
    mask[0, 0] = 3  # cystic_duct

    out = overlay_mask(image, mask, alpha=0.5)

    assert out.shape == (8, 8, 3)
    assert out.dtype == np.uint8
    assert np.array_equal(out[4, 4], image[4, 4])   # background unchanged
    assert not np.array_equal(out[0, 0], image[0, 0])  # foreground recolored


def test_cvs_panel_returns_matplotlib_figure():
    """cvs_panel renders a 3-panel matplotlib Figure."""
    from matplotlib.figure import Figure

    figure = cvs_panel(
        np.zeros((16, 16, 3), dtype=np.uint8),
        np.zeros((16, 16), dtype=np.int64),
        criteria_probs=[0.9, 0.2, 0.6],
    )
    assert isinstance(figure, Figure)
    assert len(figure.axes) == 3


def test_cvs_pipeline_outputs():
    """The pipeline maps an RGB frame to (mask, 3 probs, score in [0, 3])."""
    segmentation_model = nn.Conv2d(3, 6, kernel_size=1)
    cvs_classifier = nn.Sequential(
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(9, 3))
    pipeline = CVSPipeline(segmentation_model, cvs_classifier, device="cpu",
                           classifier_image_size=64)

    image = np.random.randint(0, 256, (48, 64, 3), dtype=np.uint8)
    mask, probs, score = pipeline(image)

    assert mask.shape == (48, 64)
    assert set(np.unique(mask)).issubset(set(range(len(CLASS_COLORS))))
    assert probs.shape == (3,)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    assert 0 <= score <= 3
    assert score == int((probs >= 0.5).sum())


def test_build_demo_constructs_blocks():
    """build_demo wires a CVSPipeline-like callable into a Gradio Blocks app."""
    gr = pytest.importorskip("gradio")
    from app.gradio_demo import build_demo

    def fake_pipeline(image):
        return np.zeros((4, 4), dtype=np.int64), np.array([0.9, 0.1, 0.5]), 2

    demo = build_demo(fake_pipeline)
    assert isinstance(demo, gr.Blocks)
