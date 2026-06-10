"""FODE-style learnable element-wise output filter.

``\\hat{x}(t) = K \\odot h(t)``, where ``K`` is a learnable parameter of the
same shape as ``h(t)``. Initialized uniformly per FODE Sec. II-D.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class FilterK(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        init: str = "uniform",
    ) -> None:
        super().__init__()
        if init == "uniform":
            init_w = torch.empty(hidden_dim).uniform_(-1.0, 1.0)
        elif init == "ones":
            init_w = torch.ones(hidden_dim)
        elif init == "zeros":
            init_w = torch.zeros(hidden_dim)
        elif init == "xavier":
            init_w = torch.empty(hidden_dim)
            nn.init.xavier_uniform_(init_w.unsqueeze(0))
            init_w = init_w.squeeze(0)
        else:
            raise ValueError(f"unknown init scheme: {init}")
        self.K = nn.Parameter(init_w)

    def forward(self, h: Tensor) -> Tensor:
        return self.K * h
