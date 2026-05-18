"""Forward-pass shape tests with dummy input.

Implemented alongside the models (Step 3: U-Net, Step 4: SAM2 + LoRA,
Step 7: CVS classifier).
"""
import pytest


@pytest.mark.skip(reason="U-Net baseline implemented in Step 3")
def test_unet_forward_shape():
    """(B, 3, 512, 512) input -> (B, 6, 512, 512) segmentation logits."""
    raise NotImplementedError


@pytest.mark.skip(reason="SAM2 + LoRA implemented in Step 4")
def test_sam2_lora_forward_shape():
    """(B, 3, 512, 512) input -> 6-class mask logits."""
    raise NotImplementedError


@pytest.mark.skip(reason="CVS classifier implemented in Step 7")
def test_cvs_classifier_forward_shape():
    """(B, 9, 224, 224) input -> (B, 3) criterion logits."""
    raise NotImplementedError
