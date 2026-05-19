"""Entry point: train an anatomical segmentation model on CholecSeg8k.

Hydra-configured (see configs/segmentation.yaml). Supports the U-Net
baseline and SAM2 + LoRA via the ``model=`` override, plus a ``low_memory=true``
flag for 16GB GPUs (smaller per-device batch + gradient accumulation).

Usage:
    python -m src.train.train_segmentation model=unet_baseline
    python -m src.train.train_segmentation model=sam2_lora low_memory=true
"""
from __future__ import annotations

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

from src.data.cholecseg8k import NUM_CLASSES, CholecSeg8kDataset
from src.data.class_balance import (
    class_loss_weights,
    compute_pixel_frequencies,
    make_weighted_sampler,
)
from src.data.transforms import build_eval_transforms, build_train_transforms
from src.models.sam2_lora import SAM2LoRASegmenter
from src.models.unet_baseline import UNetBaseline
from src.train.lightning_modules import SegmentationModule
from src.utils.seeds import seed_everything


def build_model(model_cfg: DictConfig, pretrained: bool = True):
    """Construct a segmentation model from the ``model`` config group.

    ``pretrained`` loads pretrained backbone weights; pass ``False`` when the
    weights will be overwritten anyway (e.g. the benchmark runner loading a
    trained checkpoint) to skip the download.
    """
    name = model_cfg.name
    if name == "unet_baseline":
        return UNetBaseline(
            encoder=model_cfg.encoder,
            encoder_weights=model_cfg.encoder_weights if pretrained else None,
            num_classes=model_cfg.num_classes,
        )
    if name == "sam2_lora":
        return SAM2LoRASegmenter(
            base_checkpoint=model_cfg.base_checkpoint,
            num_classes=model_cfg.num_classes,
            lora_rank=model_cfg.lora.rank,
            lora_alpha=model_cfg.lora.alpha,
            lora_dropout=model_cfg.lora.dropout,
            lora_target_modules=list(model_cfg.lora.target_modules),
            sam2_image_size=model_cfg.sam2_image_size,
            pretrained=pretrained,
        )
    raise ValueError(f"Unknown model '{name}'.")


def _batch_and_accumulation(effective_bs: int, low_memory: bool) -> tuple[int, int]:
    """Split an effective batch size into per-device batch + grad accumulation.

    ``low_memory`` targets a 16GB T4: per-device batch 1 (SAM2 @ 1024 fits only
    one frame), so the effective batch is reached purely by accumulation.
    """
    per_device = 1 if low_memory else 4
    per_device = min(per_device, effective_bs)
    accumulate = max(1, effective_bs // per_device)
    return per_device, accumulate


def _resolve_precision(requested: str) -> str:
    """Fall back from bf16 to fp16 to fp32 depending on hardware support."""
    if not torch.cuda.is_available():
        return "32-true"
    if requested.startswith("bf16") and not torch.cuda.is_bf16_supported():
        return "16-mixed"
    return requested


@hydra.main(version_base=None, config_path="../../configs",
            config_name="segmentation")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed, cfg.deterministic)

    train_tf = build_train_transforms(cfg.data.image_size)
    eval_tf = build_eval_transforms(cfg.data.image_size)
    common = dict(
        hf_repo=cfg.data.hf_repo,
        cache_dir=cfg.data.cache_dir,
        image_size=cfg.data.image_size,
        split_seed=cfg.data.split.seed,
    )
    train_ds = CholecSeg8kDataset(split="train", transform=train_tf, **common)
    val_ds = CholecSeg8kDataset(split="val", transform=eval_tf, **common)
    test_ds = CholecSeg8kDataset(split="test", transform=eval_tf, **common)

    # Per-class loss weights and the sampler are derived from the train split
    # under the deterministic eval transform (same frame order as train_ds).
    stats_ds = CholecSeg8kDataset(split="train", transform=eval_tf, **common)
    _, frequencies = compute_pixel_frequencies(stats_ds, NUM_CLASSES)
    class_weights = class_loss_weights(frequencies)

    per_device_bs, accumulate = _batch_and_accumulation(cfg.batch_size, cfg.low_memory)
    sampler = (make_weighted_sampler(stats_ds)
               if cfg.sampler == "weighted_random" else None)
    train_loader = DataLoader(
        train_ds, batch_size=per_device_bs, sampler=sampler,
        shuffle=sampler is None, num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=per_device_bs, num_workers=4,
                            pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=per_device_bs, num_workers=4,
                             pin_memory=True)

    module = SegmentationModule(
        build_model(cfg.model),
        num_classes=cfg.model.num_classes,
        class_weights=class_weights,
        lr_encoder=cfg.optimizer.lr_encoder,
        lr_decoder=cfg.optimizer.lr_decoder,
        weight_decay=cfg.optimizer.weight_decay,
        dice_weight=cfg.loss.dice_weight,
        focal_weight=cfg.loss.focal_weight,
        focal_gamma=cfg.loss.focal_gamma,
        max_epochs=cfg.epochs,
        warmup_epochs=cfg.scheduler.warmup_epochs,
    )

    callbacks = [
        EarlyStopping(monitor=cfg.early_stopping.monitor,
                      mode=cfg.early_stopping.mode,
                      patience=cfg.early_stopping.patience),
        ModelCheckpoint(monitor=cfg.early_stopping.monitor,
                        mode=cfg.early_stopping.mode, save_top_k=1,
                        dirpath=f"outputs/{cfg.model.name}", filename="best"),
    ]
    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        precision=_resolve_precision(cfg.precision),
        accumulate_grad_batches=accumulate,
        callbacks=callbacks,
        log_every_n_steps=10,
    )
    trainer.fit(module, train_loader, val_loader)
    trainer.test(module, test_loader, ckpt_path="best")


if __name__ == "__main__":
    main()
