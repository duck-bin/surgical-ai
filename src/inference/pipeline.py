"""End-to-end inference pipeline: frame -> segmentation -> CVS assessment.

Chains the segmentation model and the CVS classifier into a single callable
that returns a 6-class mask, 3 criterion probabilities and the 0-3 CVS score.
Used by the Gradio demo.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _load_state_with_prefix(module, checkpoint_path, prefix: str):
    """Load weights stored under ``prefix`` in a Lightning checkpoint."""
    # weights_only=False: the user's own checkpoints, with non-tensor metadata.
    checkpoint = torch.load(checkpoint_path, map_location="cpu",
                            weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    sub_state = {key[len(prefix):]: value
                 for key, value in state_dict.items()
                 if key.startswith(prefix)}
    module.load_state_dict(sub_state, strict=True)
    return module


class CVSPipeline:
    """Segmentation -> CVS classification inference pipeline.

    Args:
        segmentation_model: produces (1, 6, H, W) logits from an RGB tensor.
        cvs_classifier: produces (1, 3) criterion logits from a 9-channel tensor.
        device: torch device the models run on.
        classifier_image_size: spatial size of the CVS classifier input.
    """

    def __init__(self, segmentation_model, cvs_classifier, device: str = "cpu",
                 classifier_image_size: int = 224):
        self.device = device
        self.classifier_image_size = classifier_image_size
        self.segmentation_model = segmentation_model.to(device).eval()
        self.cvs_classifier = cvs_classifier.to(device).eval()

    @classmethod
    def from_checkpoints(cls, seg_model_config: str, seg_checkpoint: str,
                         cvs_checkpoint: str,
                         cvs_backbone: str = "vit_small_patch16_224",
                         device: str = "cpu",
                         classifier_image_size: int = 224) -> "CVSPipeline":
        """Build a pipeline from a segmentation checkpoint + a CVSModule one."""
        from omegaconf import OmegaConf

        from src.models.cvs_classifier import CVSClassifier
        from src.train.train_segmentation import build_model

        seg_cfg = OmegaConf.load(seg_model_config)
        segmentation_model = _load_state_with_prefix(
            build_model(seg_cfg, pretrained=False), seg_checkpoint, "model.")

        cvs_classifier = _load_state_with_prefix(
            CVSClassifier(backbone=cvs_backbone, in_channels=9, num_criteria=3,
                          pretrained=False),
            cvs_checkpoint, "classifier.")
        return cls(segmentation_model, cvs_classifier, device,
                   classifier_image_size)

    @torch.no_grad()
    def __call__(self, image):
        """Run a frame through segmentation and CVS classification.

        Args:
            image: (H, W, 3) uint8 RGB array (or anything ``np.asarray`` takes).

        Returns:
            mask: (H, W) int numpy array of 6-class indices.
            criteria_probs: (3,) float numpy array of sigmoid probabilities.
            cvs_score: int CVS score in [0, 3].
        """
        rgb = np.asarray(image, dtype=np.float32) / 255.0
        tensor = (torch.from_numpy(rgb).permute(2, 0, 1)
                  .unsqueeze(0).to(self.device))

        seg_logits = self.segmentation_model(tensor)          # (1, 6, H, W)
        mask = seg_logits.argmax(dim=1)[0]                    # (H, W)

        size = (self.classifier_image_size, self.classifier_image_size)
        seg_probs = F.interpolate(seg_logits.softmax(dim=1), size=size,
                                  mode="bilinear", align_corners=False)
        rgb_resized = F.interpolate(tensor, size=size, mode="bilinear",
                                    align_corners=False)
        cvs_input = torch.cat([seg_probs, rgb_resized], dim=1)  # (1, 9, S, S)
        probs = torch.sigmoid(self.cvs_classifier(cvs_input))[0]
        score = int((probs >= 0.5).sum())

        return mask.cpu().numpy(), probs.cpu().numpy(), score
