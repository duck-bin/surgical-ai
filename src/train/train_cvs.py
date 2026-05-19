"""Entry point: train the CVS classifier on Endoscapes2023.

Hydra-configured (see configs/cvs_classifier.yaml). A frozen segmentation
model supplies the 6 mask channels; the 9-channel (6 mask + 3 RGB) input feeds
a ViT-Small classifier with 3 binary criterion heads.

Usage:
    python -m src.train.train_cvs
"""
from __future__ import annotations

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

from src.data.endoscapes import Endoscapes2023Dataset
from src.eval.benchmark_runner import load_segmentation_checkpoint
from src.models.cvs_classifier import CVSClassifier
from src.train.lightning_modules import CVSModule
from src.train.train_segmentation import (
    _batch_and_accumulation,
    _resolve_precision,
    build_model,
)
from src.utils.seeds import seed_everything


def _pos_weight(dataset: Endoscapes2023Dataset) -> torch.Tensor:
    """BCE ``pos_weight`` per criterion: #negative / #positive over the split."""
    labels = dataset.cvs_labels()
    positives = labels.sum(axis=0)
    negatives = len(labels) - positives
    weight = negatives / np.clip(positives, 1, None)
    return torch.tensor(weight, dtype=torch.float32)


@hydra.main(version_base=None, config_path="../../configs",
            config_name="cvs_classifier")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed, cfg.deterministic)

    common = dict(root=cfg.data.root, image_size=cfg.data.image_size,
                  threshold=cfg.data.cvs_threshold)
    train_ds = Endoscapes2023Dataset(split="train", **common)
    val_ds = Endoscapes2023Dataset(split="val", **common)
    test_ds = Endoscapes2023Dataset(split="test", **common)

    # Frozen segmentation model providing the 6 mask channels.
    seg_cfg = OmegaConf.load(cfg.segmentation.model_config)
    seg_model = load_segmentation_checkpoint(
        build_model(seg_cfg, pretrained=False), cfg.segmentation.checkpoint)

    classifier = CVSClassifier(backbone=cfg.backbone,
                               in_channels=cfg.in_channels,
                               num_criteria=cfg.num_criteria)

    per_device_bs, accumulate = _batch_and_accumulation(cfg.batch_size,
                                                        cfg.low_memory)
    train_loader = DataLoader(train_ds, batch_size=per_device_bs, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=per_device_bs, num_workers=4,
                            pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=per_device_bs, num_workers=4,
                             pin_memory=True)

    pos_weight = (_pos_weight(train_ds)
                  if cfg.loss.class_weights == "from_train_set" else None)

    module = CVSModule(
        classifier, seg_model,
        num_criteria=cfg.num_criteria,
        pos_weight=pos_weight,
        lr_backbone=cfg.optimizer.lr_backbone,
        lr_heads=cfg.optimizer.lr_heads,
        weight_decay=cfg.optimizer.weight_decay,
        max_epochs=cfg.epochs,
        warmup_epochs=cfg.scheduler.warmup_epochs,
        classifier_image_size=cfg.classifier_image_size,
    )

    callbacks = [
        EarlyStopping(monitor=cfg.early_stopping.monitor,
                      mode=cfg.early_stopping.mode,
                      patience=cfg.early_stopping.patience),
        ModelCheckpoint(monitor=cfg.early_stopping.monitor,
                        mode=cfg.early_stopping.mode, save_top_k=1,
                        dirpath="outputs/cvs_classifier", filename="best"),
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
