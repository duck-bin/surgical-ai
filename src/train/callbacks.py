"""Lightning callbacks for terminal-friendly training progress.

RunPod / Colab web terminals (and any redirected log file) don't render tqdm's
carriage-return progress bar well, so the previous workflow leaned on wandb just
to watch epochs tick by. :class:`EpochProgress` prints one flushed plain-text
line per epoch to stdout, so a run can be followed directly in the terminal --
no wandb, no TTY required.

Example line::

    [epoch  12/100]   87.3s  train_loss=0.2341  val_loss=0.1980  val_miou=0.6121  val_cystic_duct_dice=0.0412
"""
from __future__ import annotations

import time

import pytorch_lightning as pl


def _fmt(value) -> str:
    """Format a scalar metric (tensor/float) to 4 decimals, robust to junk."""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


class EpochProgress(pl.Callback):
    """Print a clean one-line summary at the end of every epoch.

    ``metrics`` lists which logged metrics to surface, in order; any that are
    absent for a given epoch are skipped (so the same callback works for both
    the segmentation and CVS runs). The line also reports the wall-clock time of
    the epoch (train + validation), which is the number most users actually want
    when sizing a RunPod run.
    """

    def __init__(self, metrics=None, max_epochs: int | None = None):
        super().__init__()
        self.metrics = list(metrics) if metrics else [
            "train_loss", "val_loss", "val_miou", "val_cystic_duct_dice",
        ]
        self.max_epochs = max_epochs
        self._epoch_start: float | None = None

    def _total_epochs(self, trainer) -> int:
        return self.max_epochs or trainer.max_epochs or 0

    def on_train_start(self, trainer, pl_module) -> None:
        total = self._total_epochs(trainer)
        batches = trainer.num_training_batches
        print(f"[train] starting -- up to {total} epochs, "
              f"{batches} train batches/epoch", flush=True)

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self._epoch_start = time.time()

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        # Validation runs right after each train epoch, so callback_metrics is
        # fully populated here. Skip the pre-train sanity-check pass.
        if trainer.sanity_checking:
            return
        self._print(trainer)

    def _print(self, trainer) -> None:
        elapsed = time.time() - self._epoch_start if self._epoch_start else 0.0
        total = self._total_epochs(trainer)
        metrics = trainer.callback_metrics
        parts = [f"{name}={_fmt(metrics[name])}"
                 for name in self.metrics if name in metrics]
        body = "  ".join(parts) if parts else "(metrics pending)"
        print(f"[epoch {trainer.current_epoch + 1:3d}/{total}] "
              f"{elapsed:6.1f}s  {body}", flush=True)
