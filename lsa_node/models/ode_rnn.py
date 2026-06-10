"""ODE-RNN — strong NODE-style baseline that consumes the full window.

Rubanova et al. 2019 ("Latent ODEs for Irregularly-Sampled Time Series",
NeurIPS) introduced the ODE-RNN architecture: between consecutive
observations the hidden state evolves continuously via a Neural ODE;
at each observation a GRU cell updates the state with the new value.

This replaces the over-simplified ``NODEBaseline`` (which only used
``x_obs[:, -1]``) and is the standard reference for "NODE that handles
irregular sampling natively". Once trained, the encoded hidden state at
``t_obs[-1]`` is rolled forward by the same Neural ODE to produce
forecasts at ``t_query``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class _ODEField(nn.Module):
    def __init__(self, hidden_dim: int, mlp_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.SiLU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.SiLU(),
            nn.Linear(mlp_dim, hidden_dim),
        )

    def forward(self, t: Tensor, h: Tensor) -> Tensor:
        del t
        return self.net(h)


class ODE_RNN(nn.Module):
    """ODE-RNN encoder + ODE rollout for forecasting.

    ``forward`` walks the observation window left-to-right. Between two
    consecutive observation times ``t_{i-1}`` and ``t_i`` it integrates
    the hidden state via the Neural ODE; at ``t_i`` it applies a GRU
    cell update with the observed value. After the last observation it
    rolls forward to every query time using the same ODE.
    """

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
        # ignored kwargs (so the same YAML config can be reused)
        **_ignored,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.use_adjoint = use_adjoint

        self.gru_cell = nn.GRUCell(in_dim, hidden_dim)
        self.field = _ODEField(hidden_dim, mlp_dim)
        self.decoder = nn.Linear(hidden_dim, out_dim)

    def forward(self, x_obs: Tensor, t_obs: Tensor, t_query: Tensor) -> Tensor:
        """
        Args:
            x_obs:   (B, T, in_dim)
            t_obs:   (B, T) — assumed to be the same across batch (or at
                     least with the same Δt's); we use t_obs[0] for the
                     integration grid.
            t_query: (Q,)

        Returns:
            (Q, B, out_dim)
        """
        from torchdiffeq import odeint, odeint_adjoint
        solver_fn = odeint_adjoint if self.use_adjoint else odeint
        B, T, _ = x_obs.shape

        # Use the (shared) time grid of item 0. The dataset's forward-fill
        # protocol guarantees all items share the dense grid.
        t_grid = t_obs[0].to(x_obs.dtype)

        # ----- Stage 1: encode the window.
        h = torch.zeros(B, self.hidden_dim, dtype=x_obs.dtype, device=x_obs.device)
        # Initial GRU update at the first observation.
        h = self.gru_cell(x_obs[:, 0], h)
        for i in range(1, T):
            dt = (t_grid[i] - t_grid[i - 1]).item()
            if dt > 1e-9:
                # Integrate h from t_{i-1} to t_i.
                t_step = torch.stack([t_grid[i - 1], t_grid[i]])
                h = solver_fn(
                    self.field, h, t_step,
                    method=self.solver, rtol=self.rtol, atol=self.atol,
                )[-1]
            h = self.gru_cell(x_obs[:, i], h)

        # ----- Stage 2: roll forward to the query times.
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
        z = z[drop:]                                                    # (Q, B, hidden_dim)
        return self.decoder(z)
