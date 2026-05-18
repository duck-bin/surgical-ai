"""Entry point: train an anatomical segmentation model on CholecSeg8k.

Hydra-configured (see configs/train/segmentation.yaml). Supports U-Net baseline
and SAM2 + LoRA via the `model=` override, plus a `low_memory=true` flag for
16GB GPUs. Implemented in Step 3 (U-Net) and Step 5 (SAM2 + LoRA).

Usage (once implemented):
    python -m src.train.train_segmentation model=unet_baseline
    python -m src.train.train_segmentation model=sam2_lora low_memory=true
"""
from __future__ import annotations


def main() -> None:
    """Hydra-decorated training entry point. Implemented in Step 3."""
    raise NotImplementedError("Segmentation training is implemented in Step 3.")


if __name__ == "__main__":
    main()
