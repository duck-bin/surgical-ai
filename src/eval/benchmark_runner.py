"""Benchmark runner: reproduces the Endoscapes-paper comparison table.

Evaluates the trained segmentation models on the CholecSeg8k test split and
emits the README comparison table (mIoU, Cystic Duct Dice) with 95% bootstrap
confidence intervals over the test frames. The CVS mAP column is filled in
Step 9.

Usage:
    python -m src.eval.benchmark_runner

``hydra`` / ``omegaconf`` are imported lazily (only ``run_benchmark`` and the
CLI need them) so the metric helpers stay importable without them.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.cholecseg8k import CLASS_NAMES, NUM_CLASSES, CholecSeg8kDataset
from src.data.transforms import build_eval_transforms
from src.eval.metrics import bootstrap_ci, dice_score, iou_score

CYSTIC_DUCT_INDEX = CLASS_NAMES.index("cystic_duct")
_NAN_CI = (float("nan"), float("nan"), float("nan"))


@torch.no_grad()
def evaluate_model(model, dataloader, num_classes: int = NUM_CLASSES,
                   device: str = "cpu") -> dict[str, list[float]]:
    """Run a segmentation model over a dataloader, returning per-image metrics.

    Returns ``{"miou": [...], "cystic_duct_dice": [...]}`` with one value per
    test image. A value is NaN when the class is absent from both the
    prediction and the target (handled downstream by :func:`bootstrap_ci`).
    """
    model = model.to(device).eval()
    miou: list[float] = []
    cystic_duct_dice: list[float] = []
    for batch in dataloader:
        preds = model(batch["image"].to(device)).argmax(dim=1).cpu()
        for pred, target in zip(preds, batch["mask"]):
            _, image_miou = iou_score(pred, target, num_classes)
            per_class_dice, _ = dice_score(pred, target, num_classes)
            miou.append(float(image_miou))
            cystic_duct_dice.append(float(per_class_dice[CYSTIC_DUCT_INDEX]))
    return {"miou": miou, "cystic_duct_dice": cystic_duct_dice}


def summarize(per_image_metrics: dict, seed: int = 42) -> dict:
    """Reduce per-image metric lists to ``(mean, ci_low, ci_high)`` triples."""
    return {name: bootstrap_ci(values, seed=seed)
            for name, values in per_image_metrics.items()}


def _format_cell(ci: tuple) -> str:
    """Format a ``(mean, lo, hi)`` triple as ``mean (lo-hi)``; ``TBD`` if NaN."""
    mean, lo, hi = ci
    if mean != mean:  # NaN -> not evaluated
        return "TBD"
    return f"{mean:.3f} ({lo:.3f}-{hi:.3f})"


def format_table(results: dict) -> str:
    """Render the markdown comparison table from ``{label: summary}``."""
    lines = ["| Method | mIoU | Cystic Duct Dice | CVS mAP |",
             "|---|---|---|---|"]
    for label, summary in results.items():
        lines.append(
            f"| {label} | {_format_cell(summary['miou'])} "
            f"| {_format_cell(summary['cystic_duct_dice'])} | TBD |"
        )
    return "\n".join(lines) + "\n"


def load_segmentation_checkpoint(model, checkpoint_path):
    """Load trained weights into ``model`` from a SegmentationModule checkpoint.

    The Lightning checkpoint stores the wrapped model under a ``model.`` prefix;
    this strips it and loads the remaining state into ``model``.
    """
    # weights_only=False: these are the user's own trained checkpoints, and a
    # Lightning checkpoint carries non-tensor metadata alongside the weights.
    checkpoint = torch.load(checkpoint_path, map_location="cpu",
                            weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    prefix = "model."
    model_state = {key[len(prefix):]: value
                   for key, value in state_dict.items()
                   if key.startswith(prefix)}
    model.load_state_dict(model_state, strict=True)
    return model


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def run_benchmark(cfg) -> None:
    """Evaluate every model in ``cfg.models`` and write the comparison table.

    Models whose checkpoint file is missing are reported as ``TBD`` rather than
    failing the run, so the benchmark can be run incrementally as models finish
    training.
    """
    from omegaconf import OmegaConf

    from src.train.train_segmentation import build_model

    device = _resolve_device(str(cfg.get("device", "auto")))
    test_ds = CholecSeg8kDataset(
        split="test",
        transform=build_eval_transforms(cfg.data.image_size),
        hf_repo=cfg.data.hf_repo,
        cache_dir=cfg.data.cache_dir,
        image_size=cfg.data.image_size,
        split_seed=cfg.data.split.seed,
    )
    loader = DataLoader(test_ds, batch_size=cfg.batch_size,
                        num_workers=cfg.num_workers, pin_memory=True)

    results: dict = {}
    for entry in cfg.models:
        checkpoint = Path(entry.checkpoint)
        if not checkpoint.is_file():
            print(f"[skip] {entry.label}: checkpoint not found ({checkpoint})")
            results[entry.label] = {"miou": _NAN_CI,
                                    "cystic_duct_dice": _NAN_CI}
            continue
        model_cfg = OmegaConf.load(
            Path(cfg.config_dir) / f"{entry.model_config}.yaml")
        model = load_segmentation_checkpoint(
            build_model(model_cfg, pretrained=False), checkpoint)
        print(f"[eval] {entry.label} on {len(test_ds)} test frames ...")
        results[entry.label] = summarize(
            evaluate_model(model, loader, NUM_CLASSES, device), seed=cfg.seed)

    table = format_table(results)
    output_path = Path(cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(table)
    print(f"\nwrote {output_path}\n\n{table}")


if __name__ == "__main__":
    import hydra

    hydra.main(version_base=None, config_path="../../configs",
               config_name="eval/benchmark")(run_benchmark)()
