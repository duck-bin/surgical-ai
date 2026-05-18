"""PyTorch Lightning modules for segmentation and CVS classification.

Wraps the models, losses, optimizers (AdamW with separate encoder/decoder
learning rates), the cosine schedule with 5-epoch warmup, and metric logging.
Implemented in Step 3 (segmentation) and Step 7 (CVS classifier).
"""
from __future__ import annotations

import pytorch_lightning as pl


class SegmentationModule(pl.LightningModule):
    """Lightning module for anatomical segmentation (U-Net or SAM2 + LoRA).

    Implemented in Step 3.
    """

    def __init__(self, cfg):
        super().__init__()
        raise NotImplementedError("Implemented in Step 3.")


class CVSModule(pl.LightningModule):
    """Lightning module for the 3-criteria CVS classifier.

    Implemented in Step 7.
    """

    def __init__(self, cfg):
        super().__init__()
        raise NotImplementedError("Implemented in Step 7.")
