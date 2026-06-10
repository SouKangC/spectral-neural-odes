"""Faithful DIFFODE (Zhang et al. 2025) baseline.

DIFFODE introduces a **Differentiable Hidden State** (DHS) computed as
attention over latent observations (paper Eq. 5):

    a_t = z_t Zᵀ / √d ,   p_t = softmax(a_t) ,   S_t = p_t · Z

where ``Z = [z_{t_1}, …, z_{t_n}]`` is the encoder's latent for each
observation and ``z_t`` is a continuous-time latent that we *integrate*
via a small Neural ODE ``dz_t/dt = φ(z_t, t)``. The hidden state at any
query time falls out by evaluating the attention with the integrated
``z_t``.

The paper derives ``dS_t/dt`` analytically via Moore-Penrose inverses
(Eq. 12). For practical purposes we integrate ``z_t`` and compute
``S_t`` at each query time directly — same modelling assumption, cleaner
implementation that still consumes the full irregular observation grid.

We use the existing :class:`~lsa_node.models.encoder.IrregularEncoder`
to produce ``z_t_i`` from observations.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from .encoder import IrregularEncoder


class _LatentField(nn.Module):
    def __init__(self, d_latent: int, mlp_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_latent, mlp_dim),
            nn.SiLU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.SiLU(),
            nn.Linear(mlp_dim, d_latent),
        )

    def forward(self, t: Tensor, z: Tensor) -> Tensor:
        del t
        return self.net(z)


class DIFFODE(nn.Module):
    """DIFFODE forecaster.

    Forward pass:
    1. Encode each irregularly-sampled observation: ``z_i = ψ(x_i, t_i)``.
    2. Stack ``Z ∈ R^{B, T, d}``.
    3. Initialise the continuous latent ``z(t_obs[-1]) = z_T``.
    4. Integrate ``dz/dt = φ(z, t)`` over the query times.
    5. At each query time ``t``: ``S_t = softmax(z_t Zᵀ / √d) Z``.
    6. Decode ``ŷ_t = MLP_D(S_t)``.

    The ODE step (4) only needs the latent dimension ``d`` worth of
    hidden state; attention (5) re-aggregates global context at every
    query without requiring further integration.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,           # ≡ d_latent in the paper
        mlp_dim: int = 64,
        time_emb_dim: int = 16,
        encoder_hidden: int = 64,
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

        self.encoder = IrregularEncoder(
            in_dim=in_dim,
            time_emb_dim=time_emb_dim,
            d_latent=hidden_dim,
            hidden_dim=encoder_hidden,
        )
        self.field = _LatentField(hidden_dim, mlp_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.SiLU(),
            nn.Linear(mlp_dim, out_dim),
        )

    def _attention(self, z_t: Tensor, Z: Tensor) -> Tensor:
        """``S_t = softmax(z_t Zᵀ / √d) Z``.

        Args:
            z_t: (B, d)
            Z:   (B, T, d)

        Returns:
            S_t: (B, d)
        """
        d = z_t.shape[-1]
        # (B, 1, T)
        logits = torch.einsum("bd,btd->bt", z_t, Z) / math.sqrt(d)
        p = torch.softmax(logits, dim=-1)                                # (B, T)
        return torch.einsum("bt,btd->bd", p, Z)                          # (B, d)

    def forward(self, x_obs: Tensor, t_obs: Tensor, t_query: Tensor) -> Tensor:
        from torchdiffeq import odeint, odeint_adjoint
        Z = self.encoder(x_obs, t_obs)                                   # (B, T, d)
        z0 = Z[:, -1, :]                                                  # (B, d)
        t_grid = t_obs[0]
        t_end = t_grid[-1].to(z0)

        t_query = t_query.to(device=z0.device, dtype=z0.dtype)
        eps = 1e-6
        if t_query[0].item() > t_end.item() + eps:
            t_full = torch.cat([t_end[None], t_query])
            drop = 1
        else:
            t_full = t_query
            drop = 0

        solver_fn = odeint_adjoint if self.use_adjoint else odeint
        z_traj = solver_fn(
            self.field, z0, t_full,
            method=self.solver, rtol=self.rtol, atol=self.atol,
        )                                                                # (≥Q, B, d)
        z_traj = z_traj[drop:]                                            # (Q, B, d)

        # Per query time, compute S_t via attention on Z and decode.
        Q = z_traj.shape[0]
        S = torch.stack([self._attention(z_traj[q], Z) for q in range(Q)], dim=0)  # (Q, B, d)
        return self.decoder(S)                                            # (Q, B, out_dim)
