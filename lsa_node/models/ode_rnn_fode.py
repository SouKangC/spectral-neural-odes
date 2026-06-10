"""ODE-RNN-FODE — fusion of the two strongest single ideas from E005.

E005 showed:
* ``ODE-RNN`` (Rubanova et al. 2019) is the strongest "consume the full
  irregular window" encoder — wins 3 / 8 cells with a plain MLP vector
  field between observations.
* ``FODE`` (Guo & Weng 2025) is the strongest spectral-inductive-bias
  vector field — wins 1 / 8 (Lorenz96) by replacing the MLP with
  IFFT(MLP(FFT(h))).

This module plugs FODE's spectral vector field into ODE-RNN's
window-encoding scaffold:

* Encoder: GRU updates at each observation, FODE field integrates
  the hidden state continuously between observations.
* Forecast: same FODE field rolls the encoded state forward to the
  query times.

If this fusion beats both parents on any cell, we have a positive
result. Otherwise it confirms the negative finding from E005.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .filter_k import FilterK
from .fode import _FODEField


class ODE_RNN_FODE(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        fode_mlp_dim: int = 16,
        fode_depth: int = 3,
        solver: str = "dopri5",
        rtol: float = 1.0e-3,
        atol: float = 1.0e-4,
        use_adjoint: bool = False,
        **_ignored,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.use_adjoint = use_adjoint

        self.gru_cell = nn.GRUCell(in_dim, hidden_dim)
        self.field = _FODEField(
            n=hidden_dim, mlp_dim=fode_mlp_dim, depth=fode_depth
        )
        self.filter_k = FilterK(hidden_dim=hidden_dim, init="uniform")
        self.decoder = nn.Linear(hidden_dim, out_dim)

    def forward(self, x_obs: Tensor, t_obs: Tensor, t_query: Tensor) -> Tensor:
        from torchdiffeq import odeint, odeint_adjoint
        solver_fn = odeint_adjoint if self.use_adjoint else odeint
        B, T, _ = x_obs.shape
        t_grid = t_obs[0].to(x_obs.dtype)

        # ----- Stage 1: encode the irregular observation window.
        h = torch.zeros(B, self.hidden_dim, dtype=x_obs.dtype, device=x_obs.device)
        h = self.gru_cell(x_obs[:, 0], h)
        for i in range(1, T):
            dt = (t_grid[i] - t_grid[i - 1]).item()
            if dt > 1e-9:
                t_step = torch.stack([t_grid[i - 1], t_grid[i]])
                h = solver_fn(
                    self.field, h, t_step,
                    method=self.solver, rtol=self.rtol, atol=self.atol,
                )[-1]
            h = self.gru_cell(x_obs[:, i], h)

        # ----- Stage 2: roll forward via FODE field.
        t_query = t_query.to(device=h.device, dtype=h.dtype)
        t_end = t_grid[-1]
        eps = 1e-6
        if t_query[0].item() > t_end.item() + eps:
            t_full = torch.cat([t_end[None], t_query])
            drop = 1
        else:
            t_full = t_query
            drop = 0
        z = solver_fn(
            self.field, h, t_full,
            method=self.solver, rtol=self.rtol, atol=self.atol,
        )                                                            # (≥Q, B, hidden_dim)
        z = z[drop:]                                                  # (Q, B, hidden_dim)
        z_filt = self.filter_k(z)
        return self.decoder(z_filt)
