"""Reproducibility utilities: seed all RNGs and enforce deterministic ops."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic: bool = True) -> int:
    """Seed Python, NumPy and PyTorch RNGs for reproducible runs.

    Args:
        seed: integer seed applied to every RNG.
        deterministic: if True, force deterministic CuDNN/cuBLAS kernels via
            ``torch.use_deterministic_algorithms``. Disable for a speed boost
            when exact reproducibility is not required.

    Returns:
        The seed that was applied.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Required for deterministic cuBLAS matmul on CUDA >= 10.2.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # warn_only: ops without a deterministic implementation warn instead
        # of raising, so training never hard-crashes on an exotic kernel.
        torch.use_deterministic_algorithms(True, warn_only=True)

    return seed
