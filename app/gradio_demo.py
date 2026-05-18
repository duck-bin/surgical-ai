"""Gradio demo for the CVS assessment pipeline.

Displays, for an uploaded or sample frame: the original image, the segmentation
overlay, the 3 CVS criteria with confidence bars, and the composite 0-3 CVS
score. Measured FPS is reported on the deploying hardware. Implemented in Step 8.
"""
from __future__ import annotations

DISCLAIMER = "Research prototype, not for clinical use."


def build_demo():
    """Construct the Gradio Blocks app. Implemented in Step 8."""
    raise NotImplementedError("Gradio demo is implemented in Step 8.")


if __name__ == "__main__":
    build_demo().launch()
