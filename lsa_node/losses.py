"""Regression loss + Hoyer sparsity regularizer for attention scores.

Classification was dropped from project scope (see ``docs/changes.md``).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def mse_loss(y_pred: Tensor, y_true: Tensor) -> Tensor:
    return F.mse_loss(y_pred, y_true)


def hoyer_sparsity(p: Tensor, eps: float = 1e-12) -> Tensor:
    """Hoyer sparsity in [0, 1]; higher = sparser. Operates on the last dim.

    ``Hoyer(p) = (sqrt(N) - ||p||_1 / ||p||_2) / (sqrt(N) - 1)``.
    """
    n = p.shape[-1]
    l1 = p.abs().sum(dim=-1)
    l2 = p.pow(2).sum(dim=-1).sqrt().clamp_min(eps)
    return (math.sqrt(n) - l1 / l2) / (math.sqrt(n) - 1.0)


def hoyer_penalty(p: Tensor, weight: float) -> Tensor:
    """Negative Hoyer (so minimizing this *increases* sparsity)."""
    return -weight * hoyer_sparsity(p).mean()
