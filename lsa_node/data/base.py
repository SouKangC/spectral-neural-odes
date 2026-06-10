"""Common interfaces for time-series datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor


@dataclass
class TSItem:
    """A single time-series example (possibly irregularly sampled).

    For irregular variants ``x_obs`` is forward-filled onto the dense
    in-window grid, ``t_obs`` is dense, and ``mask_obs`` marks the
    positions of the *real* (non-filled) observations.
    """

    x_obs: Tensor                       # (T, in_dim) observed/filled values
    t_obs: Tensor                       # (T,)        dense observation timestamps
    x_query: Tensor                     # (Q, in_dim) targets at query times
    t_query: Tensor                     # (Q,)        query timestamps
    mask_obs: Tensor | None = None      # (T,) bool — True at real obs, False at fill


class TimeSeriesDataset(Protocol):
    """Minimal protocol every dataset must implement."""

    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> TSItem: ...
