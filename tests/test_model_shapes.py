"""Model, loss, and training-pipeline tests.

The U-Net baseline, the focal/Dice losses, and an end-to-end Lightning smoke
test run with synthetic input. SAM2 and the CVS classifier are added later.
"""
import pytest
import torch


def test_unet_forward_shape():
    """(B, 3, 128, 128) input -> (B, 6, 128, 128) segmentation logits."""
    from src.models.unet_baseline import UNetBaseline

    model = UNetBaseline(encoder_weights=None, num_classes=6)
    out = model(torch.randn(2, 3, 128, 128))
    assert tuple(out.shape) == (2, 6, 128, 128)


def test_unet_param_groups_split():
    """param_groups yields encoder + decoder groups at the requested LRs."""
    from src.models.unet_baseline import UNetBaseline

    model = UNetBaseline(encoder_weights=None, num_classes=6)
    groups = model.param_groups(lr_encoder=1e-4, lr_decoder=1e-3)
    assert len(groups) == 2
    assert groups[0]["lr"] == 1e-4 and groups[1]["lr"] == 1e-3
    assert all(len(g["params"]) > 0 for g in groups)


def test_dice_loss_perfect_beats_uniform():
    """Dice loss is near 0 for a perfect prediction and higher for a uniform one."""
    from src.losses.dice import DiceLoss

    loss_fn = DiceLoss()
    target = torch.randint(0, 6, (2, 16, 16))
    perfect = torch.full((2, 6, 16, 16), -10.0).scatter_(1, target.unsqueeze(1), 10.0)

    good = loss_fn(perfect, target)
    uniform = loss_fn(torch.zeros(2, 6, 16, 16), target)
    assert 0.0 <= good.item() < 0.05
    assert good.item() < uniform.item()
    assert torch.isfinite(good)


def test_focal_loss_perfect_and_gradient():
    """Focal loss is near 0 for a perfect prediction and propagates gradients."""
    from src.losses.focal import FocalLoss

    loss_fn = FocalLoss(gamma=2.0)
    target = torch.randint(0, 6, (2, 16, 16))
    perfect = torch.full((2, 6, 16, 16), -10.0).scatter_(1, target.unsqueeze(1), 10.0)
    assert loss_fn(perfect, target).item() < 0.01

    logits = torch.randn(2, 6, 16, 16, requires_grad=True)
    loss_fn(logits, target).backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_focal_loss_with_class_weights():
    """Focal loss accepts per-class weights and stays finite and non-negative."""
    from src.losses.focal import FocalLoss

    weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 10.0, 1.0])
    loss_fn = FocalLoss(gamma=2.0, class_weights=weights)
    out = loss_fn(torch.randn(2, 6, 8, 8), torch.randint(0, 6, (2, 8, 8)))
    assert torch.isfinite(out) and out.item() >= 0.0


def test_segmentation_module_smoke():
    """End-to-end Lightning pipeline runs a train + val step (fast_dev_run)."""
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader, Dataset

    from src.models.unet_baseline import UNetBaseline
    from src.train.lightning_modules import SegmentationModule

    class _Synthetic(Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, index):
            return {"image": torch.randn(3, 64, 64),
                    "mask": torch.randint(0, 6, (64, 64))}

    module = SegmentationModule(
        UNetBaseline(encoder_weights=None, num_classes=6),
        num_classes=6, max_epochs=1, warmup_epochs=0,
    )
    loader = DataLoader(_Synthetic(), batch_size=2)
    trainer = pl.Trainer(fast_dev_run=True, accelerator="cpu", logger=False,
                         enable_checkpointing=False)
    trainer.fit(module, loader, loader)


def test_sam2_lora_forward_shape():
    """(B, 3, 512, 512) input -> (B, 6, 512, 512) logits, with gradient flow."""
    from src.models.sam2_lora import SAM2LoRASegmenter

    model = SAM2LoRASegmenter(num_classes=6, pretrained=False)
    out = model(torch.randn(1, 3, 512, 512))
    assert tuple(out.shape) == (1, 6, 512, 512)

    out.sum().backward()
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert any(p.grad is not None for p in trainable)


def test_sam2_lora_param_groups_split():
    """param_groups yields encoder-LoRA + mask-decoder groups at the given LRs."""
    from src.models.sam2_lora import SAM2LoRASegmenter

    model = SAM2LoRASegmenter(num_classes=6, pretrained=False)
    groups = model.param_groups(lr_encoder=1e-4, lr_decoder=1e-3)
    assert len(groups) == 2
    assert groups[0]["lr"] == 1e-4 and groups[1]["lr"] == 1e-3
    assert all(len(g["params"]) > 0 for g in groups)


def test_cvs_classifier_forward_shape():
    """(B, 9, 224, 224) input -> (B, 3) criterion logits, with gradient flow."""
    from src.models.cvs_classifier import CVSClassifier

    model = CVSClassifier(in_channels=9, num_criteria=3, pretrained=False)
    out = model(torch.randn(2, 9, 224, 224))
    assert tuple(out.shape) == (2, 3)

    out.sum().backward()
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_cvs_classifier_param_groups_split():
    """param_groups yields backbone + heads groups at the requested LRs."""
    from src.models.cvs_classifier import CVSClassifier

    model = CVSClassifier(in_channels=9, num_criteria=3, pretrained=False)
    groups = model.param_groups(lr_backbone=1e-4, lr_heads=1e-3)
    assert len(groups) == 2
    assert groups[0]["lr"] == 1e-4 and groups[1]["lr"] == 1e-3
    assert all(len(g["params"]) > 0 for g in groups)


def test_cvs_module_smoke():
    """End-to-end CVS Lightning pipeline runs a train + val step."""
    import pytorch_lightning as pl
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    from src.models.cvs_classifier import CVSClassifier
    from src.train.lightning_modules import CVSModule

    class _Synthetic(Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, index):
            return {"image": torch.rand(3, 64, 64),
                    "criteria": torch.randint(0, 2, (3,)).float(),
                    "cvs_score": torch.randint(0, 4, ()).long()}

    # A trivial 3->6 conv stands in for the frozen segmentation model.
    module = CVSModule(
        CVSClassifier(in_channels=9, num_criteria=3, pretrained=False),
        nn.Conv2d(3, 6, kernel_size=1), max_epochs=1, warmup_epochs=0,
    )
    loader = DataLoader(_Synthetic(), batch_size=2)
    trainer = pl.Trainer(fast_dev_run=True, accelerator="cpu", logger=False,
                         enable_checkpointing=False)
    trainer.fit(module, loader, loader)
