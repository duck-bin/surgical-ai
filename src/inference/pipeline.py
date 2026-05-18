"""End-to-end inference pipeline: frame -> segmentation -> CVS assessment.

Chains the segmentation model and the CVS classifier into a single callable
that returns a 6-class mask, 3 criterion probabilities and the 0-3 CVS score.
Used by the Gradio demo. Implemented in Step 8.
"""
from __future__ import annotations


class CVSPipeline:
    """Segmentation -> CVS classification inference pipeline.

    Implemented in Step 8.
    """

    def __init__(self, seg_ckpt: str, cvs_ckpt: str, device: str = "cuda"):
        raise NotImplementedError("Inference pipeline is implemented in Step 8.")

    def __call__(self, image):
        """Return (segmentation mask, 3 criterion probs, CVS score)."""
        raise NotImplementedError("Implemented in Step 8.")
