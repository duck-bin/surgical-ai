"""Entry point: train the CVS classifier on Endoscapes2023.

Hydra-configured (see configs/cvs_classifier.yaml). A frozen segmentation
model supplies the 6 mask channels; the 9-channel (6 mask + 3 RGB) input feeds
a ViT-Small classifier with 3 binary criterion heads.

Usage:
    python -m src.train.train_cvs
"""
from __future__ import annotations

import json
import os
from pathlib import Path

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
from src.train.callbacks import EpochProgress
from src.train.lightning_modules import CVSModule
from src.train.train_segmentation import (
    _batch_and_accumulation,
    _resolve_precision,
    build_model,
)
from src.utils.seeds import seed_everything


def _report_cvs_results(test_results: list[dict], seg_model_name: str) -> None:
    """Surface the CVS test metrics so they're easy to spot after a long run.

    Prints a clearly delimited summary block and writes the numbers to
    ``results/cvs_metrics.md`` (human-readable) and ``results/cvs_metrics.json``
    (machine-readable). The benchmark runner reads the JSON -- keyed by the
    ``seg_model_name`` that fed the classifier -- to fill that model's CVS mAP
    cell. Persisting also keeps the values across a Colab scroll-back/disconnect.
    """
    metrics = test_results[0] if test_results else {}
    test_map = metrics.get("test_map")
    test_qwk = metrics.get("test_qwk")
    test_loss = metrics.get("test_loss")
    rows = [
        ("CVS mAP (mean AP over 3 criteria)", test_map),
        ("CVS QWK (score 0-3 agreement)", test_qwk),
        ("test_loss", test_loss),
    ]

    def num(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def fmt(value) -> str:
        number = num(value)
        return "n/a" if number is None else f"{number:.4f}"

    bar = "=" * 60
    print(f"\n{bar}\n  CVS classifier -- test results\n{bar}")
    for label, value in rows:
        print(f"  {label:34s}: {fmt(value)}")
    print(bar)

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    table = "# CVS classifier -- test results\n\n| Metric | Value |\n|---|---|\n"
    table += "".join(f"| {label} | {fmt(value)} |\n" for label, value in rows)
    (results_dir / "cvs_metrics.md").write_text(table)
    (results_dir / "cvs_metrics.json").write_text(json.dumps({
        "seg_model": seg_model_name,
        "test_map": num(test_map),
        "test_qwk": num(test_qwk),
        "test_loss": num(test_loss),
    }, indent=2))
    print(f"  saved -> {results_dir / 'cvs_metrics.md'}, "
          f"{results_dir / 'cvs_metrics.json'}\n")


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
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=per_device_bs, num_workers=cfg.num_workers,
                            pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=per_device_bs, num_workers=cfg.num_workers,
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

    checkpoint_dir = "outputs/cvs_classifier"
    callbacks = [
        EarlyStopping(monitor=cfg.early_stopping.monitor,
                      mode=cfg.early_stopping.mode,
                      patience=cfg.early_stopping.patience),
        ModelCheckpoint(monitor=cfg.early_stopping.monitor,
                        mode=cfg.early_stopping.mode, save_top_k=1,
                        dirpath=checkpoint_dir, filename="best",
                        save_last=True),
    ]
    # Per-epoch terminal summary (mAP/QWK for this run) so progress is visible in
    # a plain terminal without wandb.
    progress_cfg = cfg.get("progress", {})
    if progress_cfg.get("per_epoch", True):
        callbacks.append(EpochProgress(
            metrics=["train_loss", "val_loss", "val_map", "val_qwk"],
            max_epochs=cfg.epochs))
    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        precision=_resolve_precision(cfg.precision),
        accumulate_grad_batches=accumulate,
        limit_train_batches=cfg.limit_batches,
        limit_val_batches=cfg.limit_batches,
        limit_test_batches=cfg.limit_batches,
        callbacks=callbacks,
        log_every_n_steps=10,
        enable_progress_bar=progress_cfg.get("bar", True),
    )
    # Resume from the last checkpoint when a previous run was interrupted.
    last_ckpt = os.path.join(checkpoint_dir, "last.ckpt")
    trainer.fit(module, train_loader, val_loader,
                ckpt_path=last_ckpt if os.path.exists(last_ckpt) else None)
    test_results = trainer.test(module, test_loader, ckpt_path="best")
    _report_cvs_results(test_results, seg_cfg.name)


if __name__ == "__main__":
    main()
