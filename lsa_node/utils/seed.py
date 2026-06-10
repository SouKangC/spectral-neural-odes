"""Seed utilities for reproducibility."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and Torch (CPU + CUDA) RNGs.

    If ``deterministic`` is True we also enable PyTorch's deterministic
    algorithms (cudnn + ``use_deterministic_algorithms``), which trades
    some speed for bit-exact reproducibility — appropriate for unit tests
    and small smoke runs. Disable for full training runs to recover the
    autotuner speedup; results across seeds are then statistically (not
    bitwise) reproducible.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # CUBLAS workspace must be set for full determinism with CUDA >= 10.2.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:  # noqa: BLE001 — older torch versions
            pass
