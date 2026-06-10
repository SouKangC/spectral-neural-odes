"""Decoder MLP ``MLP_D : h(t) → x̂``.

Regression-only: classification was dropped from scope (see
``docs/changes.md`` and ``docs/ideas.md`` §1).

``h(t)`` is the post-filter ODE hidden state — a length-``N`` time-domain
patch (per ``docs/qa_design_choices.md`` Q2). The decoder maps each
``N``-vector to ``out_dim``-vector — typically the dimension of the
observed signal.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class RegressionDecoder(nn.Module):
    """MLP ``R^N → R^{out_dim}`` with one hidden layer."""

    def __init__(self, hidden_dim: int, out_dim: int, mlp_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.SiLU(),
            nn.Linear(mlp_dim, out_dim),
        )

    def forward(self, h: Tensor) -> Tensor:
        """
        Args:
            h: ``(..., hidden_dim)`` — any leading batch dims.

        Returns:
            ``(..., out_dim)``.
        """
        return self.net(h)
