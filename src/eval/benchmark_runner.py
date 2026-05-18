"""Benchmark runner: reproduces the Endoscapes-paper comparison table.

Runs the trained models over the test split and emits the README comparison
table (mIoU, Cystic Duct Dice, CVS mAP) with 95% bootstrap CIs.
Implemented in Step 5 (segmentation rows) and Step 9 (final table).
"""
from __future__ import annotations


def run_benchmark(cfg) -> None:
    """Evaluate all trained models and write the comparison table. Step 5/9."""
    raise NotImplementedError("Benchmark runner is implemented in Step 5.")
