"""Visualization helpers: segmentation overlays and CVS-criteria panels."""
from __future__ import annotations

import numpy as np

# RGB colors for the 6-class schema (index 0 = background, never overlaid).
CLASS_COLORS: np.ndarray = np.array(
    [
        [0, 0, 0],        # background
        [220, 60, 60],    # liver
        [70, 180, 220],   # gallbladder
        [250, 200, 50],   # cystic_duct
        [120, 220, 100],  # cystic_artery
        [200, 200, 200],  # tool
    ],
    dtype=np.uint8,
)


def overlay_mask(image, mask, alpha: float = 0.5) -> np.ndarray:
    """Blend a 6-class segmentation mask onto an RGB frame.

    Args:
        image: (H, W, 3) uint8 RGB frame.
        mask: (H, W) integer label map with values in [0, 5].
        alpha: overlay opacity for non-background pixels.

    Returns:
        (H, W, 3) uint8 RGB array; background pixels keep the original frame.
    """
    image = np.asarray(image, dtype=np.uint8)
    mask = np.asarray(mask)
    color = CLASS_COLORS[np.clip(mask, 0, len(CLASS_COLORS) - 1)]
    blended = (1.0 - alpha) * image + alpha * color
    out = np.where((mask == 0)[..., None], image, blended)
    return out.astype(np.uint8)


def cvs_panel(image, mask, criteria_probs):
    """Render the frame, segmentation overlay and 3 CVS criteria bars.

    Returns a matplotlib ``Figure`` (Agg backend -- safe for headless use).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.data.endoscapes import CVS_CRITERIA

    probs = np.asarray(criteria_probs, dtype=float).ravel()
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(np.asarray(image, dtype=np.uint8))
    axes[0].set_title("frame")
    axes[1].imshow(overlay_mask(image, mask))
    axes[1].set_title("segmentation")
    for ax in axes[:2]:
        ax.axis("off")

    axes[2].barh(range(len(probs)), probs, color="steelblue")
    axes[2].set_yticks(range(len(probs)))
    axes[2].set_yticklabels(list(CVS_CRITERIA))
    axes[2].set_xlim(0.0, 1.0)
    axes[2].invert_yaxis()
    axes[2].set_title(f"CVS score: {int((probs >= 0.5).sum())}/3")

    figure.tight_layout()
    return figure
