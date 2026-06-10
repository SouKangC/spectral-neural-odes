"""Torch ``Dataset`` wrappers that produce LSA-NODE training tuples.

Every dataset yields a :class:`~lsa_node.data.base.TSItem` (``x_obs``,
``t_obs``, ``x_query``, ``t_query``) — the input contract every model in
``code/lsa_node/models/`` agrees on.

The training tuple is a forecasting **sliding window**: given a length-``T``
input segment ending at time ``t_T``, predict the next ``Q`` values at
times ``t_{T+1} … t_{T+Q}``.
"""

from __future__ import annotations

from typing import Callable, Sequence

import torch
from torch.utils.data import Dataset, default_collate

from .base import TSItem
from .synthetic import poisson_subsample


class SlidingWindowDataset(Dataset[TSItem]):
    """Forecasting windows over a single dense trajectory.

    Each item is one ``(in_len, out_len)`` window. Adjacent items overlap
    by ``in_len + out_len - stride`` samples; with ``stride=in_len+out_len``
    the windows tile the trajectory disjointly.

    For the **irregular** variant, the *input* segment is Poisson-
    subsampled per item (one fresh subsample per window, controlled by
    ``irregular_seed`` + the window index). Query timestamps stay at the
    full grid so we can compute MSE on the same comparison points.
    """

    def __init__(
        self,
        traj: dict[str, torch.Tensor],
        in_len: int,
        out_len: int,
        stride: int | None = None,
        irregular: bool = False,
        irregular_keep_rate: float = 0.3,
        irregular_seed_offset: int = 0,
    ) -> None:
        t = traj["t"]
        x = traj["x"] if traj["x"].dim() == 2 else traj["x"].unsqueeze(-1)
        if t.shape[0] != x.shape[0]:
            raise ValueError(f"t and x must agree on length, got {t.shape}, {x.shape}")
        N = t.shape[0]
        window = in_len + out_len
        if N < window:
            raise ValueError(f"trajectory too short: {N} < in_len+out_len={window}")
        self.t = t.float()
        self.x = x.float()
        self.in_len = int(in_len)
        self.out_len = int(out_len)
        self.stride = int(stride or (in_len + out_len))
        self.irregular = bool(irregular)
        self.keep_rate = float(irregular_keep_rate)
        self.seed_offset = int(irregular_seed_offset)
        self._starts = list(range(0, N - window + 1, self.stride))

    def __len__(self) -> int:
        return len(self._starts)

    def __getitem__(self, idx: int) -> TSItem:
        start = self._starts[idx]
        i_end = start + self.in_len
        q_end = i_end + self.out_len

        t_in = self.t[start:i_end].clone()
        x_in = self.x[start:i_end].clone()
        t_q = self.t[i_end:q_end]
        x_q = self.x[i_end:q_end]
        mask = torch.ones(self.in_len, dtype=torch.bool)

        if self.irregular:
            g = torch.Generator().manual_seed(self.seed_offset + idx)
            keep = torch.rand(self.in_len, generator=g) < self.keep_rate
            # Make sure at least the first and last positions are kept so
            # forward-fill works and the model gets the most recent obs.
            keep[0] = True
            keep[-1] = True
            # Forward-fill: at every non-kept position copy the previous
            # value (the loop is cheap; in_len is small).
            mask = keep
            for j in range(1, self.in_len):
                if not keep[j]:
                    x_in[j] = x_in[j - 1]

        return TSItem(
            x_obs=x_in, t_obs=t_in, x_query=x_q, t_query=t_q,
            mask_obs=mask,
        )

    @classmethod
    def from_generator(
        cls,
        gen: Callable[..., dict[str, torch.Tensor]],
        *,
        in_len: int,
        out_len: int,
        irregular: bool = False,
        irregular_keep_rate: float = 0.3,
        irregular_seed_offset: int = 0,
        stride: int | None = None,
        **gen_kwargs,
    ) -> "SlidingWindowDataset":
        traj = gen(**gen_kwargs)
        return cls(
            traj,
            in_len=in_len,
            out_len=out_len,
            stride=stride,
            irregular=irregular,
            irregular_keep_rate=irregular_keep_rate,
            irregular_seed_offset=irregular_seed_offset,
        )


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------
def collate_ts_items(items: Sequence[TSItem]):
    """Stack a batch of :class:`TSItem`.

    All items share the dense ``t_obs`` grid by construction (the
    dataset forward-fills irregular observations onto that grid before
    returning the item), so collation is a plain stack. ``mask_obs``
    marks the real (non-filled) positions per item.
    """
    x_obs = default_collate([it.x_obs for it in items])              # (B, T, in_dim)
    t_obs = default_collate([it.t_obs for it in items])              # (B, T)
    mask = default_collate([
        it.mask_obs if it.mask_obs is not None
        else torch.ones(it.x_obs.shape[0], dtype=torch.bool)
        for it in items
    ])                                                                 # (B, T)
    x_query = default_collate([it.x_query for it in items])           # (B, Q, in_dim)
    t_query = items[0].t_query                                         # (Q,) — shared

    return {
        "x_obs": x_obs,
        "t_obs": t_obs,
        "mask_obs": mask,
        "x_query": x_query,
        "t_query": t_query,
    }
