"""PyTorch Lightning modules for segmentation and CVS classification.

Wraps the models, losses, optimizers (AdamW with separate encoder/decoder
learning rates), the cosine schedule with linear warmup, and metric logging.
The segmentation module is implemented here; the CVS module is added in Step 7.
"""
from __future__ import annotations

import math

import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from src.eval.metrics import dice_score, iou_score
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

    Implemented in Step 7.
    """

    def __init__(self, cfg):
        super().__init__()
        raise NotImplementedError("Implemented in Step 7.")
