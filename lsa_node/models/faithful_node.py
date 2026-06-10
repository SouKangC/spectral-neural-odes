"""Faithful Neural ODE for time series — Latent-ODE-style (deterministic).

Chen et al. 2018's Neural ODE for time series is paired with a recurrent
encoder that processes the entire observation window into a single
initial latent z(t_0). Then dz/dt = f_θ(z, t) integrates forward to
query times. Our ``NODEBaseline`` skipped the encoder entirely and used
only ``x_obs[:, -1]`` — making the regular/irregular distinction
unfalsifiable.

This faithful version:

1. **Reverse-time GRU encoder** reads ``x_obs[1:T]`` backwards (Rubanova
   2019 §3.1 convention) into the hidden state ``h0 = encoder(x_obs)``.
2. **Time-aware vector field** ``f_θ(h, t) = MLP([h; t])``.
3. **Linear decoder** at query times.

This is the *deterministic* Latent-ODE. The variational version adds
a (mu, sigma) split + reparameterised sampling; for our deterministic
forecasting setup the extra noise just hurts.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class _TimeAwareField(nn.Module):
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


class FaithfulNODE(nn.Module):
    """Latent-ODE-style faithful NODE for forecasting."""

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

        # Reverse-time GRU encoder (one layer).
        self.encoder_rnn = nn.GRU(
            input_size=in_dim, hidden_size=hidden_dim, batch_first=True
        )
        self.encoder_proj = nn.Linear(hidden_dim, hidden_dim)
        self.field = _TimeAwareField(hidden_dim, mlp_dim)
        self.decoder = nn.Linear(hidden_dim, out_dim)

    def forward(self, x_obs: Tensor, t_obs: Tensor, t_query: Tensor) -> Tensor:
        del t_obs
        # Reverse-in-time encoding per Latent ODE convention.
        x_rev = x_obs.flip(dims=[1])                              # (B, T, in)
        _, h_T = self.encoder_rnn(x_rev)                          # h_T: (1, B, hidden)
        h0 = self.encoder_proj(h_T.squeeze(0))                    # (B, hidden)

        t_query = t_query.to(device=h0.device, dtype=h0.dtype)
        from torchdiffeq import odeint, odeint_adjoint
        solver_fn = odeint_adjoint if self.use_adjoint else odeint
        h = solver_fn(
            self.field, h0, t_query,
            method=self.solver, rtol=self.rtol, atol=self.atol,
        )                                                          # (Q, B, hidden)
        return self.decoder(h)
