"""PyTorch Lightning modules for segmentation and CVS classification.

Wraps the models, losses, optimizers (AdamW with separate encoder/decoder
learning rates), the cosine schedule with linear warmup, and metric logging.
"""
from __future__ import annotations

import math

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
                 lr_encoder: float = 1e-4, lr_decoder: float = 1e-3,
                 weight_decay: float = 0.01, dice_weight: float = 1.0,
                 focal_weight: float = 1.0, focal_gamma: float = 2.0,
                 max_epochs: int = 100, warmup_epochs: int = 5):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "class_weights"])
        self.model = model
        self.num_classes = num_classes
        self.lr_encoder = lr_encoder
        self.lr_decoder = lr_decoder
        self.weight_decay = weight_decay
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.focal = FocalLoss(gamma=focal_gamma, class_weights=class_weights)
        self.dice = DiceLoss()

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, stage: str):
        logits = self(batch["image"])
        target = batch["mask"]
        loss = (self.focal_weight * self.focal(logits, target)
                + self.dice_weight * self.dice(logits, target))

        preds = logits.argmax(dim=1)
        _, miou = iou_score(preds, target, self.num_classes)
        _, mdice = dice_score(preds, target, self.num_classes)

        batch_size = batch["image"].shape[0]
        self.log(f"{stage}_loss", loss, prog_bar=True, batch_size=batch_size)
        self.log(f"{stage}_miou", miou, prog_bar=True, batch_size=batch_size)
        self.log(f"{stage}_mdice", mdice, batch_size=batch_size)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

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
