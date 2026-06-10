"""Tests for losses.hoyer_sparsity (the only non-trivial loss)."""

from __future__ import annotations

import math

import torch

from lsa_node.losses import hoyer_sparsity


def test_hoyer_one_hot_is_one() -> None:
    p = torch.zeros(8)
    p[3] = 1.0
    assert math.isclose(hoyer_sparsity(p).item(), 1.0, abs_tol=1e-6)


def test_hoyer_uniform_is_zero() -> None:
    p = torch.full((8,), 1.0 / 8.0)
    assert math.isclose(hoyer_sparsity(p).item(), 0.0, abs_tol=1e-6)
