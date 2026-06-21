"""Entry point: train an anatomical segmentation model on CholecSeg8k.

Hydra-configured (see configs/segmentation.yaml). Supports the U-Net
baseline and SAM2 + LoRA via the ``model=`` override, plus a ``low_memory=true``
flag for 16GB GPUs (smaller per-device batch + gradient accumulation).

Usage:
    python -m src.train.train_segmentation model=unet_baseline
    python -m src.train.train_segmentation model=sam2_lora low_memory=true
"""
from __future__ import annotations

import os

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

from src.data.cholecseg8k import (
    CLASS_NAMES,
    NUM_CLASSES,
    CholecSeg8kDataset,
    CholecSeg8kWindowDataset,
)
from src.data.class_balance import (
    class_loss_weights,
    compute_or_load_class_stats,
    sampler_from_weights,
    window_sample_weights,
)
from src.data.transforms import build_eval_transforms, build_train_transforms
from src.models.sam2_lora import SAM2LoRASegmenter
from src.models.temporal import TemporalSAM2LoRASegmenter
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
    if name == "sam2_temporal":
        return TemporalSAM2LoRASegmenter(
            base_checkpoint=model_cfg.base_checkpoint,
            num_classes=model_cfg.num_classes,
            lora_rank=model_cfg.lora.rank,
            lora_alpha=model_cfg.lora.alpha,
            lora_dropout=model_cfg.lora.dropout,
            lora_target_modules=list(model_cfg.lora.target_modules),
            sam2_image_size=model_cfg.sam2_image_size,
            pretrained=pretrained,
            window=model_cfg.window,
            temporal_hidden=model_cfg.temporal.hidden,
            temporal_lr=model_cfg.temporal.get("lr", None),
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

    common = dict(
        hf_repo=cfg.data.hf_repo,
        cache_dir=cfg.data.cache_dir,
        image_size=cfg.data.image_size,
        split_seed=cfg.data.split.seed,
    )
    # The temporal model consumes clips of `window` consecutive frames; its
    # train augmentation must be replay-consistent across the clip (ConvGRU
    # needs spatial correspondence), so the train transform is a ReplayCompose.
    window = cfg.model.get("window", None)
    is_temporal = window is not None
    eval_tf = build_eval_transforms(cfg.data.image_size)
    train_tf = build_train_transforms(cfg.data.image_size, replay=is_temporal)
    if is_temporal:
        train_ds = CholecSeg8kWindowDataset(split="train", window=window,
                                            transform=train_tf, **common)
        val_ds = CholecSeg8kWindowDataset(split="val", window=window,
                                          transform=eval_tf, **common)
        test_ds = CholecSeg8kWindowDataset(split="test", window=window,
                                           transform=eval_tf, **common)
    else:
        train_ds = CholecSeg8kDataset(split="train", transform=train_tf, **common)
        val_ds = CholecSeg8kDataset(split="val", transform=eval_tf, **common)
        test_ds = CholecSeg8kDataset(split="test", transform=eval_tf, **common)

    # Per-class loss weights and the sampler are derived from the train split
    # under the deterministic eval transform (same frame order as train_ds).
    # Both require full passes over the train set, so skip them when disabled
    # (e.g. a smoke test with sampler=null loss.class_weighting=none).
    # The frame-level sampler weights also drive the temporal path: clips are
    # oversampled by their target frame's weight (see window_sample_weights),
    # so the temporal model no longer sees rare classes only at natural
    # prevalence -- the cause of its cystic_duct Dice never leaving zero.
    need_sampler = cfg.sampler == "weighted_random"
    need_class_weights = str(cfg.loss.class_weighting).lower() not in (
        "none", "null", "off", "false", "")
    class_weights = None
    sample_weights = None
    if need_class_weights or need_sampler:
        stats_ds = CholecSeg8kDataset(split="train", transform=eval_tf, **common)
        # One pass computes BOTH frequencies and sampler weights, cached to disk
        # keyed by split seed -- re-runs and later models then start instantly
        # instead of repeating the multi-minute decode of the whole train set.
        cache_path = os.path.join(
            cfg.data.cache_dir, f"class_stats_seed{cfg.data.split.seed}.npz")
        frequencies, sample_weights = compute_or_load_class_stats(
            stats_ds, NUM_CLASSES, cache_path=cache_path)
        if need_class_weights:
            clip = tuple(cfg.loss.get("class_weight_clip", (0.5, 10.0)))
            class_weights = class_loss_weights(frequencies, clip=clip)

    per_device_bs, accumulate = _batch_and_accumulation(cfg.batch_size, cfg.low_memory)
    if need_sampler:
        # Map per-frame weights onto clips for the temporal model; frame-level
        # models use the per-frame weights directly. Both index the same split.
        sampler_weights = (window_sample_weights(train_ds._windows, sample_weights)
                           if is_temporal else sample_weights)
        sampler = sampler_from_weights(sampler_weights)
    else:
        sampler = None
    train_loader = DataLoader(
        train_ds, batch_size=per_device_bs, sampler=sampler,
        shuffle=sampler is None, num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=per_device_bs, num_workers=cfg.num_workers,
                            pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=per_device_bs, num_workers=cfg.num_workers,
                             pin_memory=True)

    module = SegmentationModule(
        build_model(cfg.model),
        num_classes=cfg.model.num_classes,
        class_weights=class_weights,
        class_names=CLASS_NAMES,
        lr_encoder=cfg.optimizer.lr_encoder,
        lr_decoder=cfg.optimizer.lr_decoder,
        weight_decay=cfg.optimizer.weight_decay,
        dice_weight=cfg.loss.dice_weight,
        focal_weight=cfg.loss.focal_weight,
        focal_gamma=cfg.loss.focal_gamma,
        max_epochs=cfg.epochs,
        warmup_epochs=cfg.scheduler.warmup_epochs,
    )

    checkpoint_dir = f"outputs/{cfg.model.name}"
    # Single ModelCheckpoint with save_top_k=1: ``best.ckpt`` is unambiguously
    # the top-1 by the monitor (the path benchmark_runner loads), and
    # ``last.ckpt`` (save_last) supports resume after an interruption.
    # NOTE: an earlier design added a SECOND ModelCheckpoint for top-k, but
    # Lightning rejects two stateful ModelCheckpoints with the same
    # monitor/mode ("more than one stateful callback ... identical state_key").
    # With save_top_k>1 on a fixed filename, ``best.ckpt`` is also no longer
    # guaranteed to be rank-1, so top-1 is the correct, benchmark-safe choice.
    callbacks = [
        EarlyStopping(monitor=cfg.early_stopping.monitor,
                      mode=cfg.early_stopping.mode,
                      patience=cfg.early_stopping.patience),
        ModelCheckpoint(monitor=cfg.early_stopping.monitor,
                        mode=cfg.early_stopping.mode, save_top_k=1,
                        dirpath=checkpoint_dir, filename="best",
                        save_last=True),
    ]
    # wandb logger: disabled by default (offline-friendly). Set
    # ``wandb.mode=online`` to stream live training curves to the wandb web UI;
    # set ``wandb.mode=offline`` to log to disk only.
    wandb_mode = str(cfg.wandb.get("mode", "disabled")).lower()
    if wandb_mode in ("online", "offline"):
        from pytorch_lightning.loggers import WandbLogger

        logger = WandbLogger(project=cfg.wandb.project,
                             name=cfg.model.name, mode=wandb_mode)
    else:
        logger = False

    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        # Floor on training length so early-stopping can't fire during the rare
        # class's zero-Dice warmup (cystic_duct needs ~30+ epochs before its
        # monitored metric leaves zero); the first run early-stopped at epoch 11.
        min_epochs=cfg.get("min_epochs", 1),
        precision=_resolve_precision(cfg.precision),
        accumulate_grad_batches=accumulate,
        limit_train_batches=cfg.limit_batches,
        limit_val_batches=cfg.limit_batches,
        limit_test_batches=cfg.limit_batches,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
    )
    # Resume from the last checkpoint when a previous run was interrupted
    # (e.g. a Colab disconnect) -- re-running the script then continues.
    last_ckpt = os.path.join(checkpoint_dir, "last.ckpt")
    trainer.fit(module, train_loader, val_loader,
                ckpt_path=last_ckpt if os.path.exists(last_ckpt) else None)
    trainer.test(module, test_loader, ckpt_path="best")


if __name__ == "__main__":
    main()
