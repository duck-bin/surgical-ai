"""PyTorch Lightning modules for segmentation and CVS classification.

Wraps the models, losses, optimizers (AdamW with separate encoder/decoder
learning rates), the cosine schedule with linear warmup, and metric logging.
"""
from __future__ import annotations

import math

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from src.eval.metrics import (
    cvs_metrics,
    dice_score,
    iou_score,
    quadratic_weighted_kappa,
)
from src.losses.dice import DiceLoss
from src.losses.focal import FocalLoss
from src.losses.partial_label import PartialLabelSegLoss
from src.losses.tversky import TverskyLoss


def _warmup_cosine(warmup_epochs: int, max_epochs: int):
    """LR multiplier schedule: linear warmup, then cosine decay to 0."""

    def multiplier(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return multiplier


class SegmentationModule(pl.LightningModule):
    """Lightning module for anatomical segmentation (U-Net or SAM2 + LoRA).

    ``model`` must produce (B, num_classes, H, W) logits and expose
    ``param_groups(lr_encoder, lr_decoder)``. The loss is a weighted sum of
    focal loss and Dice loss.
    """

    def __init__(self, model, *, num_classes: int = 6, class_weights=None,
                 class_names=None, lr_encoder: float = 1e-4,
                 lr_decoder: float = 1e-3,
                 weight_decay: float = 0.01, dice_weight: float = 1.0,
                 focal_weight: float = 1.0, focal_gamma: float = 2.0,
                 region_loss: str = "dice", tversky_alpha: float = 0.3,
                 tversky_beta: float = 0.7, tversky_gamma: float = 1.0,
                 max_epochs: int = 100, warmup_epochs: int = 5):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "class_weights"])
        self.model = model
        self.num_classes = num_classes
        self.class_names = tuple(class_names) if class_names is not None else None
        self.lr_encoder = lr_encoder
        self.lr_decoder = lr_decoder
        self.weight_decay = weight_decay
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.focal = FocalLoss(gamma=focal_gamma, class_weights=class_weights)
        # Region term: plain Dice, or Tversky / Focal-Tversky for small, thin
        # classes (beta > alpha penalizes misses harder than false positives;
        # gamma > 1 focuses on the still-poorly-segmented rare classes). The
        # macro average already balances classes, so class_weights stays on the
        # focal (pixel-level) term only and is not duplicated here.
        if str(region_loss).lower() in ("tversky", "focal_tversky"):
            self.region = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta,
                                      gamma=tversky_gamma)
        else:
            self.region = DiceLoss()
        # Partial-label (marginal) variant of the same focal + region loss, used
        # only when a batch carries a per-sample ``labeled`` mask (joint
        # CholecSeg8k+Endoscapes-Seg training). With every class labelled it is
        # numerically the same loss, so single-dataset runs are unaffected.
        self.partial_loss = PartialLabelSegLoss(
            focal_weight=focal_weight, region_weight=dice_weight,
            focal_gamma=focal_gamma, class_weights=class_weights,
            region_loss=region_loss, tversky_alpha=tversky_alpha,
            tversky_beta=tversky_beta, tversky_gamma=tversky_gamma)
        # Per-class Dice is aggregated at epoch end (not per-step): a class
        # absent from a batch yields NaN, which would poison a running mean and
        # break the monitored ``val_<class>_dice`` checkpoint/early-stop metric.
        self._dice_buffer: dict[str, list] = {"val": [], "test": []}

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, stage: str):
        logits = self(batch["image"])
        target = batch["mask"]
        # Joint partially-labelled batches carry a per-sample ``labeled`` mask;
        # supervise through the marginal loss so each source's unlabelled classes
        # are folded into background instead of penalised. Single-dataset batches
        # (no ``labeled``) take the ordinary focal + region loss unchanged.
        if "labeled" in batch:
            loss = self.partial_loss(logits, target, batch["labeled"])
        else:
            loss = (self.focal_weight * self.focal(logits, target)
                    + self.dice_weight * self.region(logits, target))

        preds = logits.argmax(dim=1)
        _, miou = iou_score(preds, target, self.num_classes)
        per_class_dice, mdice = dice_score(preds, target, self.num_classes)

        batch_size = batch["image"].shape[0]
        self.log(f"{stage}_loss", loss, prog_bar=True, batch_size=batch_size)
        self.log(f"{stage}_miou", miou, prog_bar=True, batch_size=batch_size)
        self.log(f"{stage}_mdice", mdice, batch_size=batch_size)
        # Stash per-class Dice for epoch-end aggregation (val/test only).
        if stage in self._dice_buffer:
            self._dice_buffer[stage].append(per_class_dice)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def on_validation_epoch_end(self):
        self._log_per_class_dice("val")

    def on_test_epoch_end(self):
        self._log_per_class_dice("test")

    def _log_per_class_dice(self, stage: str):
        """Log epoch-level per-class Dice (NaN-ignoring mean over batches).

        Emits ``{stage}_{name}_dice`` for every class, so ``cystic_duct`` Dice
        becomes available for model selection (configs/segmentation.yaml monitors
        ``val_cystic_duct_dice``) and the rest are visible for post-hoc analysis.
        A class never present in the split logs 0.0 (rather than NaN) so the
        monitored metric stays well-defined.
        """
        buffer = self._dice_buffer.get(stage, [])
        if not buffer:
            return
        stacked = np.stack(buffer, axis=0)  # (n_batches, num_classes)
        self._dice_buffer[stage] = []
        for c in range(self.num_classes):
            column = stacked[:, c]
            column = column[~np.isnan(column)]
            value = float(column.mean()) if column.size else 0.0
            name = self.class_names[c] if self.class_names else f"class{c}"
            self.log(f"{stage}_{name}_dice", value,
                     prog_bar=(name == "cystic_duct"))


    def configure_optimizers(self):
        groups = self.model.param_groups(self.lr_encoder, self.lr_decoder)
        optimizer = AdamW(groups, weight_decay=self.weight_decay)
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=_warmup_cosine(self.warmup_epochs, self.max_epochs),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


class CVSModule(pl.LightningModule):
    """Lightning module for the 3-criteria CVS classifier.

    Composes a 9-channel input -- the frozen segmentation model's 6 softmax
    channels plus the 3 RGB channels, both resized to the classifier's input
    size -- and trains the CVS classifier with BCE-with-logits over the 3
    Strasberg criteria. Validation/test report mAP and the quadratic-weighted
    kappa of the derived CVS score (0-3).
    """

    def __init__(self, classifier, segmentation_model, *, num_criteria: int = 3,
                 pos_weight=None, lr_backbone: float = 1e-4,
                 lr_heads: float = 1e-3, weight_decay: float = 0.01,
                 max_epochs: int = 60, warmup_epochs: int = 5,
                 classifier_image_size: int = 224):
        super().__init__()
        self.save_hyperparameters(ignore=["classifier", "segmentation_model",
                                          "pos_weight"])
        self.classifier = classifier
        self.segmentation_model = segmentation_model
        for param in self.segmentation_model.parameters():
            param.requires_grad_(False)
        self.segmentation_model.eval()

        self.num_criteria = num_criteria
        self.lr_backbone = lr_backbone
        self.lr_heads = lr_heads
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.classifier_image_size = classifier_image_size
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self._eval_outputs: list[tuple] = []

    def train(self, mode: bool = True):
        """Keep the frozen segmentation model in eval mode at all times."""
        super().train(mode)
        self.segmentation_model.eval()
        return self

    def compose_input(self, rgb):
        """RGB (B,3,H,W) -> 9-channel (B,9,S,S): 6 segmentation softmax + RGB."""
        with torch.no_grad():
            seg_probs = self.segmentation_model(rgb).softmax(dim=1)
        size = (self.classifier_image_size, self.classifier_image_size)
        seg_probs = F.interpolate(seg_probs, size=size, mode="bilinear",
                                  align_corners=False)
        rgb = F.interpolate(rgb, size=size, mode="bilinear",
                            align_corners=False)
        return torch.cat([seg_probs, rgb], dim=1)

    def forward(self, rgb):
        return self.classifier(self.compose_input(rgb))

    def _shared_step(self, batch, stage: str):
        logits = self(batch["image"])
        target = batch["criteria"]
        loss = self.loss_fn(logits, target)
        self.log(f"{stage}_loss", loss, prog_bar=True,
                 batch_size=target.shape[0])
        return logits, target, loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")[2]

    def validation_step(self, batch, batch_idx):
        logits, target, _ = self._shared_step(batch, "val")
        self._eval_outputs.append(
            (logits.detach().cpu(), target.detach().cpu()))

    def test_step(self, batch, batch_idx):
        logits, target, _ = self._shared_step(batch, "test")
        self._eval_outputs.append(
            (logits.detach().cpu(), target.detach().cpu()))

    def on_validation_epoch_end(self):
        self._epoch_metrics("val")

    def on_test_epoch_end(self):
        self._epoch_metrics("test")

    def _epoch_metrics(self, stage: str):
        """Aggregate the epoch's predictions into dataset-level mAP and QWK."""
        if not self._eval_outputs:
            return
        logits = torch.cat([out[0] for out in self._eval_outputs])
        target = torch.cat([out[1] for out in self._eval_outputs])
        self._eval_outputs.clear()

        self.log(f"{stage}_map", cvs_metrics(logits, target)["map"],
                 prog_bar=True)
        pred_score = (torch.sigmoid(logits) >= 0.5).sum(dim=1)
        true_score = target.sum(dim=1)
        self.log(f"{stage}_qwk",
                 quadratic_weighted_kappa(pred_score, true_score))

    def configure_optimizers(self):
        groups = self.classifier.param_groups(self.lr_backbone, self.lr_heads)
        optimizer = AdamW(groups, weight_decay=self.weight_decay)
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=_warmup_cosine(self.warmup_epochs, self.max_epochs),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
