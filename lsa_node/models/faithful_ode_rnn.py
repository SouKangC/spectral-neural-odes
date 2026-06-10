"""Faithful ODE-RNN (Rubanova et al. 2019).

Only difference from ``ode_rnn.py``: the vector field receives the time
t (non-autonomous). Paper's f_θ(h, t) — our v1 had f_θ(h) only.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class _TimeAwareODEField(nn.Module):
    """f_θ(h, t) = MLP([h; t])."""

    def __init__(self, hidden_dim: int, mlp_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + 1, mlp_dim),
            nn.SiLU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.SiLU(),
            nn.Linear(mlp_dim, hidden_dim),
        )

    def forward(self, t: Tensor, h: Tensor) -> Tensor:
        B = h.shape[0]
        t_b = t.expand(B, 1) if t.dim() == 0 else t.view(1, 1).expand(B, 1)
        return self.net(torch.cat([h, t_b], dim=-1))


class FaithfulODE_RNN(nn.Module):
    """ODE-RNN with time-aware vector field."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        mlp_dim: int = 64,
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
        self.field = _TimeAwareODEField(hidden_dim, mlp_dim)
        self.decoder = nn.Linear(hidden_dim, out_dim)

    def forward(self, x_obs: Tensor, t_obs: Tensor, t_query: Tensor) -> Tensor:
        from torchdiffeq import odeint, odeint_adjoint
        solver_fn = odeint_adjoint if self.use_adjoint else odeint
        B, T, _ = x_obs.shape

        t_grid = t_obs[0].to(x_obs.dtype)
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
        )
        z = z[drop:]
        return self.decoder(z)
