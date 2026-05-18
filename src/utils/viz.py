"""Visualization helpers: segmentation overlays and CVS-criteria panels.

Implemented in Step 3 (segmentation overlays) and Step 8 (demo panels).
"""
from __future__ import annotations


def overlay_mask(image, mask, alpha: float = 0.5):
    """Blend a 6-class segmentation mask onto an RGB frame."""
    raise NotImplementedError("Implemented in Step 3.")


def cvs_panel(image, mask, criteria_probs):
    """Render the original frame, segmentation overlay and 3 CVS criteria bars."""
    raise NotImplementedError("Implemented in Step 8.")
