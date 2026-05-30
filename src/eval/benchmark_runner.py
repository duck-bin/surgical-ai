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

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.cholecseg8k import (
    CLASS_NAMES,
    NUM_CLASSES,
    CholecSeg8kDataset,
    CholecSeg8kWindowDataset,
)
from src.data.transforms import build_eval_transforms
from src.eval.metrics import bootstrap_ci, dice_score, iou_score

CYSTIC_DUCT_INDEX = CLASS_NAMES.index("cystic_duct")
_NAN_CI = (float("nan"), float("nan"), float("nan"))


@torch.no_grad()
def evaluate_model(model, dataloader, num_classes: int = NUM_CLASSES,
                   device: str = "cpu",
                   max_batches: int | None = None) -> dict[str, list[float]]:
    """Run a segmentation model over a dataloader, returning per-image metrics.

    Returns a flat dict of per-image value lists: ``miou``, ``cystic_duct_dice``
    (the headline metrics), plus ``iou_<class>`` and ``dice_<class>`` for each
    of the 6 classes. A value is NaN when the class is absent from both the
    prediction and the target (handled downstream by :func:`bootstrap_ci`).

    ``max_batches`` caps the number of batches evaluated (a smoke-test knob);
    ``None`` evaluates the whole dataloader.
    """
    model = model.to(device).eval()
    metrics: dict[str, list[float]] = {"miou": [], "cystic_duct_dice": []}
    for name in CLASS_NAMES:
        metrics[f"iou_{name}"] = []
        metrics[f"dice_{name}"] = []
    for index, batch in enumerate(dataloader):
        if max_batches is not None and index >= max_batches:
            break
        preds = model(batch["image"].to(device)).argmax(dim=1).cpu()
        for pred, target in zip(preds, batch["mask"]):
            per_class_iou, image_miou = iou_score(pred, target, num_classes)
            per_class_dice, _ = dice_score(pred, target, num_classes)
            metrics["miou"].append(float(image_miou))
            metrics["cystic_duct_dice"].append(
                float(per_class_dice[CYSTIC_DUCT_INDEX]))
            for class_index, name in enumerate(CLASS_NAMES):
                metrics[f"iou_{name}"].append(float(per_class_iou[class_index]))
                metrics[f"dice_{name}"].append(float(per_class_dice[class_index]))
    return metrics


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


def _format_scalar(value) -> str:
    """Format a single number as ``0.123``; ``TBD`` for ``None`` or NaN."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "TBD"
    return "TBD" if number != number else f"{number:.3f}"


def format_table(results: dict, cvs_map: dict | None = None) -> str:
    """Render the headline comparison table from ``{label: summary}``.

    ``cvs_map`` optionally maps a model label to its CVS mAP. The CVS classifier
    is trained on a single segmentation model, so usually only that model's row
    carries a value and the rest stay ``TBD``.
    """
    cvs_map = cvs_map or {}
    lines = ["| Method | mIoU | Cystic Duct Dice | CVS mAP |",
             "|---|---|---|---|"]
    for label, summary in results.items():
        lines.append(
            f"| {label} | {_format_cell(summary['miou'])} "
            f"| {_format_cell(summary['cystic_duct_dice'])} "
            f"| {_format_scalar(cvs_map.get(label))} |"
        )
    return "\n".join(lines) + "\n"


def format_per_class_table(results: dict, kind: str) -> str:
    """Render a per-class breakdown table; ``kind`` is ``"iou"`` or ``"dice"``.

    Rows are the 6 anatomical classes, columns the evaluated models. A class
    absent from the test split (e.g. ``cystic_artery`` in CholecSeg8k) renders
    as ``TBD``.
    """
    labels = list(results)
    lines = ["| Class | " + " | ".join(labels) + " |",
             "|" + "---|" * (len(labels) + 1)]
    for name in CLASS_NAMES:
        cells = [_format_cell(results[label].get(f"{kind}_{name}", _NAN_CI))
                 for label in labels]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
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


def _load_cvs_map(cfg) -> dict:
    """Map each model label to its CVS mAP, read from the CVS metrics JSON.

    ``train_cvs`` writes ``results/cvs_metrics.json`` recording which
    segmentation model fed the classifier; only that model's row gets a value.
    Returns ``{}`` when the file is absent (CVS not trained yet).
    """
    path = Path(str(cfg.get("cvs_metrics_path", "results/cvs_metrics.json")))
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except (ValueError, OSError):
        return {}
    seg_model = data.get("seg_model")
    test_map = data.get("test_map")
    return {entry.label: test_map
            for entry in cfg.models if entry.model_config == seg_model}


def run_benchmark(cfg) -> None:
    """Evaluate every model in ``cfg.models`` and write the comparison table.

    Models whose checkpoint file is missing are reported as ``TBD`` rather than
    failing the run, so the benchmark can be run incrementally as models finish
    training.
    """
    from omegaconf import OmegaConf

    from src.train.train_segmentation import build_model

    device = _resolve_device(str(cfg.get("device", "auto")))
    eval_tf = build_eval_transforms(cfg.data.image_size)
    test_kwargs = dict(
        split="test",
        hf_repo=cfg.data.hf_repo,
        cache_dir=cfg.data.cache_dir,
        image_size=cfg.data.image_size,
        split_seed=cfg.data.split.seed,
    )

    def make_test_loader(model_cfg):
        """Frame loader by default; a clip loader for temporal models so they
        are evaluated *with* temporal context (not collapsed to T=1)."""
        window = model_cfg.get("window", None)
        if window is not None:
            dataset = CholecSeg8kWindowDataset(window=window, transform=eval_tf,
                                               **test_kwargs)
        else:
            dataset = CholecSeg8kDataset(transform=eval_tf, **test_kwargs)
        loader = DataLoader(dataset, batch_size=cfg.batch_size,
                            num_workers=cfg.num_workers, pin_memory=True)
        return loader, len(dataset)

    max_batches = cfg.get("max_eval_batches", None)
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
        loader, num_items = make_test_loader(model_cfg)
        scope = (f"{max_batches} batches" if max_batches is not None
                 else f"{num_items} test items")
        print(f"[eval] {entry.label} on {scope} ...")
        results[entry.label] = summarize(
            evaluate_model(model, loader, NUM_CLASSES, device,
                           max_batches=max_batches), seed=cfg.seed)

    table = format_table(results, _load_cvs_map(cfg))
    table += "\n## Per-class IoU\n\n" + format_per_class_table(results, "iou")
    table += "\n## Per-class Dice\n\n" + format_per_class_table(results, "dice")
    output_path = Path(cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(table)
    print(f"\nwrote {output_path}\n\n{table}")


if __name__ == "__main__":
    import hydra

    hydra.main(version_base=None, config_path="../../configs",
               config_name="benchmark")(run_benchmark)()
