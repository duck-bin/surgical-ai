"""Lightweight temporal head over the per-frame SAM2 + LoRA segmenter.

Motivation
----------
The frame-level :class:`~src.models.sam2_lora.SAM2LoRASegmenter` segments each
frame independently, so thin, intermittently visible structures (notably the
``cystic_duct``) flicker between frames and are easily missed on hard frames.
This module adds short-range temporal context: a window of ``T`` consecutive
frames is segmented per frame, and a small **ConvGRU** aggregates the per-frame
class-evidence maps to refine the prediction for the *target* (last) frame.

Why a head, not ``Sam2VideoModel``
----------------------------------
transformers (>=5.8) does ship a SAM2 *video* model (see
:func:`sam2_video_available`), but it is built around a prompt-driven,
streaming inference-session memory bank (point/box prompts, per-object state).
Repurposing it *prompt-free* for a dense multi-class + LoRA training target is
heavy and version-fragile. A small ConvGRU head is far simpler to get working
and keeps the per-frame SAM2 encoder/decoder (and its LoRA) exactly as-is.

Where the fusion happens
------------------------
Fusion is applied to the per-frame **mask-decoder outputs** at SAM2's native
mask resolution (``SAM2LoRASegmenter.native_logits``), not to the encoder
embeddings. This keeps the head fully decoupled from SAM2's internal
encoder/decoder split (no reconstruction of high-res FPN features or no-prompt
prompt embeddings). A **zero-initialised residual** means that at init the
temporal model reproduces the frame-level baseline exactly on the target frame;
training only ever *adds* a learned temporal correction.

Output is (B, num_classes, H, W) for the target frame -- a drop-in for the
shared :class:`~src.train.lightning_modules.SegmentationModule`.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.sam2_lora import SAM2LoRASegmenter


def sam2_video_available() -> bool:
    """True if the installed ``transformers`` exposes a SAM2 *video* model.

    Used to document the Step-1 capability check explicitly in code. The
    temporal head is used regardless (simpler to get working); this probe lets a
    future revision branch on a trainable propagation module if desired.
    """
    try:
        import transformers
    except Exception:
        return False
    return any(hasattr(transformers, name)
               for name in ("Sam2VideoModel", "Sam2VideoModelWithMemory"))


class ConvGRUCell(nn.Module):
    """A single convolutional GRU cell (spatial state, channel-wise gating)."""

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        gate_in = in_channels + hidden_channels
        self.conv_zr = nn.Conv2d(gate_in, 2 * hidden_channels, kernel_size,
                                 padding=padding)
        self.conv_n = nn.Conv2d(gate_in, hidden_channels, kernel_size,
                                padding=padding)
        self.hidden_channels = hidden_channels

    def forward(self, x, h):
        z, r = torch.sigmoid(self.conv_zr(torch.cat([x, h], dim=1))).chunk(2, dim=1)
        n = torch.tanh(self.conv_n(torch.cat([x, r * h], dim=1)))
        return (1.0 - z) * h + z * n


class TemporalRefinementHead(nn.Module):
    """ConvGRU over a window of per-frame logit maps -> a residual correction.

    Input  : (B, T, C, h, w) per-frame class-evidence (logit) maps.
    Output : (B, C, h, w) additive correction for the *target* (last) frame.
    The final projection is zero-initialised, so an untrained head outputs
    zeros and the temporal model equals the frame-level baseline at init.
    """

    def __init__(self, num_classes: int, hidden_channels: int = 64,
                 kernel_size: int = 3):
        super().__init__()
        self.cell = ConvGRUCell(num_classes, hidden_channels, kernel_size)
        self.project = nn.Conv2d(hidden_channels, num_classes, kernel_size=1)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)
        self.hidden_channels = hidden_channels

    def forward(self, sequence):
        batch, _, _, height, width = sequence.shape
        h = sequence.new_zeros(batch, self.hidden_channels, height, width)
        for t in range(sequence.shape[1]):
            h = self.cell(sequence[:, t], h)
        return self.project(h)


class TemporalSAM2LoRASegmenter(nn.Module):
    """SAM2 + LoRA with a ConvGRU temporal head over a window of ``T`` frames.

    Args mirror :class:`~src.models.sam2_lora.SAM2LoRASegmenter`, plus:
        window: number of consecutive frames per clip (T); the last is the target.
        temporal_hidden: ConvGRU hidden channels.
        temporal_lr: LR for the temporal head's own optimizer group. ``None``
            falls back to the decoder LR passed to :meth:`param_groups`.

    ``forward`` accepts either a clip ``(B, T, 3, H, W)`` or a single frame
    ``(B, 3, H, W)`` (treated as T=1); it always returns ``(B, num_classes, H, W)``
    logits for the target frame -- a drop-in for ``SegmentationModule``.
    """

    def __init__(self, base_checkpoint: str = "facebook/sam2-hiera-base-plus",
                 num_classes: int = 6, lora_rank: int = 8, lora_alpha: int = 16,
                 lora_dropout: float = 0.05, lora_target_modules=("qkv",),
                 sam2_image_size: int = 1024, pretrained: bool = True,
                 window: int = 3, temporal_hidden: int = 64,
                 temporal_lr: float | None = None):
        super().__init__()
        self.num_classes = num_classes
        self.window = window
        self.temporal_lr = temporal_lr
        self.frame_model = SAM2LoRASegmenter(
            base_checkpoint=base_checkpoint, num_classes=num_classes,
            lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
            sam2_image_size=sam2_image_size, pretrained=pretrained,
        )
        self.temporal = TemporalRefinementHead(num_classes, temporal_hidden)

    def forward(self, x):
        if x.dim() == 4:  # (B, 3, H, W) -> single-frame window
            x = x.unsqueeze(1)
        batch, time, _, height, width = x.shape
        # Per-frame native-resolution logits, stacked into (B, T, C, h, w).
        per_frame = [self.frame_model.native_logits(x[:, t]) for t in range(time)]
        sequence = torch.stack(per_frame, dim=1)
        correction = self.temporal(sequence)            # (B, C, h, w)
        refined = sequence[:, -1] + correction          # zero-init residual
        return F.interpolate(refined, size=(height, width), mode="bilinear",
                             align_corners=False)

    def param_groups(self, lr_encoder: float, lr_decoder: float) -> list[dict]:
        """Encoder LoRA + mask decoder (from the frame model) + temporal head.

        The temporal head is its own group at ``temporal_lr`` (or ``lr_decoder``
        when unset); the frame model's encoder/decoder groups are unchanged.
        """
        groups = self.frame_model.param_groups(lr_encoder, lr_decoder)
        temporal_lr = lr_decoder if self.temporal_lr is None else self.temporal_lr
        groups.append({"params": list(self.temporal.parameters()),
                       "lr": temporal_lr})
        return groups
