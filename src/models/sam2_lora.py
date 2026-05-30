"""SAM2 with LoRA adapters for surgical anatomical segmentation (primary model).

Base: ``facebook/sam2-hiera-base-plus``, loaded via ``transformers.Sam2Model``
(the native ``sam2`` package is avoided -- it brings its own ``hydra-core``
config system). LoRA adapters (PEFT) are added to the Hiera vision encoder's
attention projections; the small mask decoder is fully fine-tuned and
repurposed to emit a dense, prompt-free ``num_classes`` logit map -- its
``num_multimask_outputs`` is set to the number of target classes. This keeps
the model a drop-in for the shared ``SegmentationModule``: ``forward`` returns
(B, num_classes, H, W) logits, exactly like the U-Net baseline.

Motivated by SurgiSAM2 (Kamtam et al., arXiv:2503.03942, 2025).
"""
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import Sam2Config, Sam2Model


class SAM2LoRASegmenter(nn.Module):
    """SAM2 + LoRA producing a dense ``num_classes`` segmentation map.

    Args:
        base_checkpoint: HuggingFace SAM2 checkpoint id.
        num_classes: number of target segmentation classes.
        lora_rank: LoRA rank for the vision-encoder attention adapters.
        lora_alpha: LoRA alpha (scaling) for those adapters.
        lora_dropout: dropout applied inside the LoRA adapters.
        lora_target_modules: attention sub-module names to adapt.
        sam2_image_size: spatial size the SAM2 Hiera backbone expects; inputs
            are resized to this internally and predictions resized back.
        pretrained: load pretrained SAM2 weights (downloads the large weight
            file). When False the weights are randomly initialised; the
            architecture still comes from ``base_checkpoint``'s config (a small
            cached download) so a trained checkpoint loads back correctly.
    """

    def __init__(self, base_checkpoint: str = "facebook/sam2-hiera-base-plus",
                 num_classes: int = 6, lora_rank: int = 8, lora_alpha: int = 16,
                 lora_dropout: float = 0.05, lora_target_modules=("qkv",),
                 sam2_image_size: int = 1024, pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.sam2_image_size = sam2_image_size

        if pretrained:
            config = Sam2Config.from_pretrained(base_checkpoint)
            config.mask_decoder_config.num_multimask_outputs = num_classes
            sam = Sam2Model.from_pretrained(
                base_checkpoint, config=config, ignore_mismatched_sizes=True,
            )
        else:
            # The architecture must still come from base_checkpoint's config so
            # that a checkpoint trained with pretrained=True reloads into a
            # pretrained=False model (benchmark / CVS). A default Sam2Config()
            # builds a different (smaller) Hiera and fails the strict load.
            # Only the pretrained *weights* are skipped here.
            config = Sam2Config.from_pretrained(base_checkpoint)
            config.mask_decoder_config.num_multimask_outputs = num_classes
            sam = Sam2Model(config)

        lora = LoraConfig(
            r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=list(lora_target_modules), bias="none",
        )
        self.sam = get_peft_model(sam, lora)
        # The mask decoder is small and repurposed (new class tokens), so it is
        # fully fine-tuned rather than LoRA-adapted.
        for param in self.sam.get_base_model().mask_decoder.parameters():
            param.requires_grad_(True)

    def forward(self, x):
        """(B, 3, H, W) RGB input -> (B, num_classes, H, W) segmentation logits."""
        height, width = x.shape[-2:]
        if (height, width) != (self.sam2_image_size, self.sam2_image_size):
            x = F.interpolate(x, size=(self.sam2_image_size, self.sam2_image_size),
                              mode="bilinear", align_corners=False)
        outputs = self.sam(pixel_values=x, multimask_output=True)
        logits = outputs.pred_masks.squeeze(1)  # (B, 1, C, h, w) -> (B, C, h, w)
        return F.interpolate(logits, size=(height, width), mode="bilinear",
                             align_corners=False)

    def param_groups(self, lr_encoder: float, lr_decoder: float) -> list[dict]:
        """Optimizer parameter groups -- encoder LoRA adapters at ``lr_encoder``,
        the fully fine-tuned mask decoder at ``lr_decoder``.
        """
        encoder, decoder = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            (encoder if "lora_" in name else decoder).append(param)
        return [
            {"params": encoder, "lr": lr_encoder},
            {"params": decoder, "lr": lr_decoder},
        ]
