"""Visualization helpers for attention maps and trajectory plots."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_attention_heatmap(
    attn: np.ndarray,
    out_path: Path | str,
    title: str = "",
) -> None:
    """Save a (τ, ω) attention heatmap. ``attn`` is (L, W) or (L, T*L)."""
    raise NotImplementedError("matplotlib imshow + colorbar")


def save_trajectory_plot(
    t: np.ndarray,
    truth: np.ndarray,
    pred: np.ndarray,
    out_path: Path | str,
) -> None:
    raise NotImplementedError("scaffold only")
