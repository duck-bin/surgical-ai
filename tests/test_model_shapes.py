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


def test_segmentation_module_logs_finite_cystic_duct_dice_when_absent():
    """val_cystic_duct_dice is logged and finite even when the class is absent.

    Per-class Dice is NaN for a class missing from a batch; if that NaN reached
    the logged ``val_cystic_duct_dice`` it would poison the epoch mean and break
    the checkpoint/early-stop monitor. The masks here never contain cystic_duct
    (index 3), so the metric must aggregate to a finite 0.0, not NaN.
    """
    import pytorch_lightning as pl
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    from src.data.cholecseg8k import CLASS_NAMES
    from src.train.lightning_modules import SegmentationModule

    class _Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 6, kernel_size=1)

        def forward(self, x):
            return self.conv(x)

        def param_groups(self, lr_encoder, lr_decoder):
            return [{"params": list(self.parameters()), "lr": lr_decoder}]

    class _NoCysticDuct(Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, index):
            # Classes 0..2 only -> cystic_duct (3), cystic_artery (4), tool (5)
            # never appear, so their per-class Dice is NaN every batch.
            return {"image": torch.randn(3, 16, 16),
                    "mask": torch.randint(0, 3, (16, 16))}

    module = SegmentationModule(_Tiny(), num_classes=6, class_names=CLASS_NAMES,
                                max_epochs=1, warmup_epochs=0)
    loader = DataLoader(_NoCysticDuct(), batch_size=2)
    trainer = pl.Trainer(max_epochs=1, limit_train_batches=1, limit_val_batches=2,
                         num_sanity_val_steps=0, accelerator="cpu", logger=False,
                         enable_checkpointing=False)
    trainer.fit(module, loader, loader)

    metric = trainer.callback_metrics["val_cystic_duct_dice"]
    assert torch.isfinite(metric).all()
    assert float(metric) == 0.0


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


def test_temporal_head_zero_init_residual_and_shape():
    """ConvGRU head: (B,T,C,h,w) -> (B,C,h,w); zero-init -> no-op correction."""
    from src.models.temporal import TemporalRefinementHead

    head = TemporalRefinementHead(num_classes=6, hidden_channels=8)
    sequence = torch.randn(2, 3, 6, 16, 16)
    correction = head(sequence)
    assert tuple(correction.shape) == (2, 6, 16, 16)
    # The output projection is zero-initialised, so an untrained head is a no-op.
    assert torch.allclose(correction, torch.zeros_like(correction))


def test_temporal_head_learns_nonzero_after_step():
    """A single optimizer step makes the residual correction non-zero."""
    from src.models.temporal import TemporalRefinementHead

    head = TemporalRefinementHead(num_classes=4, hidden_channels=8)
    sequence = torch.randn(1, 3, 4, 8, 8)
    optimizer = torch.optim.SGD(head.parameters(), lr=1.0)
    loss = (head(sequence) - 1.0).pow(2).mean()  # push correction toward 1
    loss.backward()
    optimizer.step()
    assert not torch.allclose(head(sequence), torch.zeros(1, 4, 8, 8))


def test_temporal_sam2_forward_shape():
    """Clip (B,T,3,H,W) -> (B,6,H,W) target-frame logits, with gradient flow."""
    from src.models.temporal import TemporalSAM2LoRASegmenter

    model = TemporalSAM2LoRASegmenter(num_classes=6, pretrained=False, window=2)
    out = model(torch.randn(1, 2, 3, 128, 128))
    assert tuple(out.shape) == (1, 6, 128, 128)

    out.sum().backward()
    temporal_params = list(model.temporal.parameters())
    assert any(p.grad is not None for p in temporal_params)


def test_temporal_sam2_accepts_single_frame():
    """A bare (B,3,H,W) frame is treated as a T=1 clip (drop-in compatibility)."""
    from src.models.temporal import TemporalSAM2LoRASegmenter

    model = TemporalSAM2LoRASegmenter(num_classes=6, pretrained=False, window=3)
    out = model(torch.randn(1, 3, 128, 128))
    assert tuple(out.shape) == (1, 6, 128, 128)


def test_temporal_sam2_param_groups_split():
    """param_groups yields encoder-LoRA + mask-decoder + temporal groups."""
    from src.models.temporal import TemporalSAM2LoRASegmenter

    model = TemporalSAM2LoRASegmenter(num_classes=6, pretrained=False, window=2,
                                      temporal_lr=5e-4)
    groups = model.param_groups(lr_encoder=1e-4, lr_decoder=1e-3)
    assert len(groups) == 3
    assert groups[0]["lr"] == 1e-4 and groups[1]["lr"] == 1e-3
    assert groups[2]["lr"] == 5e-4  # temporal head's own LR group
    assert all(len(g["params"]) > 0 for g in groups)


def test_temporal_sam2_default_temporal_lr_is_decoder_lr():
    """When temporal_lr is unset, the temporal group uses the decoder LR."""
    from src.models.temporal import TemporalSAM2LoRASegmenter

    model = TemporalSAM2LoRASegmenter(num_classes=6, pretrained=False, window=2)
    groups = model.param_groups(lr_encoder=1e-4, lr_decoder=1e-3)
    assert groups[2]["lr"] == 1e-3


def test_temporal_segmentation_module_smoke():
    """End-to-end Lightning train + val step on a synthetic clip (no download).

    The temporal analogue of ``test_segmentation_module_smoke``: the synthetic
    dataset yields a (T, 3, H, W) clip per item plus the target frame's (H, W)
    mask -- exactly the layout ``CholecSeg8kWindowDataset`` produces. Runs on
    CPU, no network, no checkpoint. Asserts the forward output is
    (B, num_classes, H, W) and the focal+Dice loss is finite, then runs one
    fast_dev_run train + val step to confirm the pipeline connects end-to-end.

    Memory-shaped for CI: T=2 (smallest meaningful temporal window) so the
    backward keeps activations for two SAM2 forwards rather than three, the
    shape/finite check runs under ``no_grad`` and its graph is ``del``-released
    before ``trainer.fit``, and ``gc.collect`` between the bare-forward and the
    Lightning step releases the autograd buffers from earlier tests. Hosted CI
    runners are ~7 GB and SAM2 activations at 1024 are large.
    """
    import gc

    import pytorch_lightning as pl
    from torch.utils.data import DataLoader, Dataset

    from src.models.temporal import TemporalSAM2LoRASegmenter
    from src.train.lightning_modules import SegmentationModule

    # Release any SAM2 instances left dangling by earlier temporal tests in
    # this module before allocating the heaviest one.
    gc.collect()

    window = 2

    class _SyntheticClips(Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, index):
            return {"image": torch.randn(window, 3, 64, 64),
                    "mask": torch.randint(0, 6, (64, 64))}

    model = TemporalSAM2LoRASegmenter(num_classes=6, pretrained=False,
                                      window=window)
    module = SegmentationModule(model, num_classes=6, max_epochs=1,
                                warmup_epochs=0)

    loader = DataLoader(_SyntheticClips(), batch_size=1)
    batch = next(iter(loader))
    with torch.no_grad():
        logits = model(batch["image"])
        assert tuple(logits.shape) == (1, 6, 64, 64)
        loss = (module.focal_weight * module.focal(logits, batch["mask"])
                + module.dice_weight * module.dice(logits, batch["mask"]))
        assert torch.isfinite(loss)
    del logits, loss
    gc.collect()

    trainer = pl.Trainer(fast_dev_run=True, accelerator="cpu", logger=False,
                         enable_checkpointing=False)
    trainer.fit(module, loader, loader)


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
