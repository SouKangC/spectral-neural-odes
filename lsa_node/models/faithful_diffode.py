"""Faithful DIFFODE re-implementation (Zhang et al. 2025, ICDE).

Differences from ``diffode.py`` (our first attempt):

1. **State integrated.** v1 integrates only ``z_t``; this version
   integrates the joint state ``(z_t, S_t)`` so the dynamics of the
   Differentiable Hidden State ``S_t`` follow the paper's Eq. 12
   exactly:

       dz_t/dt = phi(z_t, t)                                        (11)
       dS_t/dt = phi(z_t, t) * Z^T (P_diag - p_t^T p_t) Z / sqrt(d) (12)

   where ``P_diag = diag(p_t)`` and ``p_t = softmax(z_t Z^T / sqrt(d))``.
   The bracket ``(P_diag - p_t^T p_t)`` is exactly the Jacobian of the
   softmax, which is what the paper's chain-rule derivation produces.

2. **Hoyer sparsity in the loss.** The paper enforces Hoyer-sparse
   attention. We add Hoyer's negative as a loss term:

       Hoyer(p) = (sqrt(T) - ||p||_1 / ||p||_2) / (sqrt(T) - 1)

   with ``loss = MSE - lambda * mean_t Hoyer(p_t)``.

3. **No Eq. 34 Taylor formula for z_t-from-S_t.** That formula requires
   an additional pseudo-inverse at every ODE step and is numerically
   delicate. Since we integrate z_t directly (Eq. 11), we don't need
   the backward derivation — we have z_t available throughout.

4. **No Eq. 32 Hoyer-projected p_t.** The paper projects p_t onto the
   sparse cone at every ODE step via a closed-form Lagrange multiplier
   (Eq. 32). We approximate this by leaving the softmax p_t in the
   dynamics and pushing sparsity via the loss term instead. Cleaner
   numerics; the limit ``lambda -> infty`` recovers the projection.

Net effect: same dz/dt and dS/dt as the paper, with attention sparsity
encouraged but not strictly enforced at each ODE step.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from .encoder import IrregularEncoder


def hoyer_sparsity(p: Tensor, eps: float = 1e-8) -> Tensor:
    """Hoyer's sparsity metric, in [0, 1].

    Args:
        p: (..., T) — probability vectors (last dim).

    Returns:
        (..., ) — Hoyer score per row. 0 = uniform, 1 = one-hot.
    """
    T = p.shape[-1]
    sqrt_T = math.sqrt(T)
    l1 = p.abs().sum(-1)
    l2 = p.pow(2).sum(-1).clamp_min(eps).sqrt()
    return (sqrt_T - l1 / l2) / (sqrt_T - 1.0)


class _PhiField(nn.Module):
    """The MLP phi(z, t) — the only learnable dynamics piece (Eq. 11)."""

    def __init__(self, d_latent: int, mlp_dim: int) -> None:
        super().__init__()
        # Include t as an input (per ``phi(z_t, t)``).
        self.net = nn.Sequential(
            nn.Linear(d_latent + 1, mlp_dim),
            nn.SiLU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.SiLU(),
            nn.Linear(mlp_dim, d_latent),
        )

    def forward(self, z: Tensor, t: Tensor) -> Tensor:
        # z: (B, d), t: scalar
        B = z.shape[0]
        t_b = t.expand(B, 1) if t.dim() == 0 else t.view(1, 1).expand(B, 1)
        return self.net(torch.cat([z, t_b], dim=-1))


class _JointDIFFODEField(nn.Module):
    """Vector field for the joint state (z_t, S_t) of dimension 2*d.

    Receives Z (fixed during integration) via ``set_keys``.
    """

    def __init__(self, d_latent: int, mlp_dim: int) -> None:
        super().__init__()
        self.d = d_latent
        self.phi = _PhiField(d_latent, mlp_dim)
        self.Z: Tensor | None = None       # (B, T, d)
        self.ZT: Tensor | None = None      # (B, d, T)
        # Trace of p_t for Hoyer loss — appended at every solver eval.
        self.p_trace: list[Tensor] = []
        self._record_p = False

    def set_keys(self, Z: Tensor) -> None:
        self.Z = Z
        self.ZT = Z.transpose(-1, -2)
        self.p_trace = []

    def start_recording(self) -> None:
        self._record_p = True
        self.p_trace = []

    def stop_recording(self) -> None:
        self._record_p = False

    def forward(self, t: Tensor, state: Tensor) -> Tensor:
        # state: (B, 2*d) — [z_t; S_t]
        z = state[..., : self.d]
        # S not used inside the vector field — only its dynamics are computed.
        # (z is the query; S is updated via the closed form from z.)

        # p_t = softmax(z Z^T / sqrt(d))
        logits = torch.einsum("bd,btd->bt", z, self.Z) / math.sqrt(self.d)
        p = torch.softmax(logits, dim=-1)                                # (B, T)

        if self._record_p:
            # Detach so the trace doesn't keep huge grads — Hoyer is a side
            # objective; we compute it on the trace post hoc.
            self.p_trace.append(p)

        # phi(z, t)
        phi_v = self.phi(z, t)                                           # (B, d)

        # dz/dt = phi (Eq. 11)
        dz = phi_v

        # dS/dt = phi @ [Z^T (P_diag - p^T p) Z / sqrt(d)] (Eq. 12)
        # Compute bracket M = Z^T (P_diag - p_outer) Z / sqrt(d), shape (B,d,d).
        # Use the identity: Z^T diag(p) Z = sum_i p_i z_i z_i^T (works batched).
        # And:                Z^T p^T p Z = (Z^T p^T)(p Z) = m m^T where m = Z^T p^T.
        # So Z^T (P_diag - p^T p) Z = sum_i p_i z_i z_i^T  - m m^T.
        # m = sum_i p_i z_i = p Z (this is exactly S_t!).
        # So bracket = sum_i p_i z_i z_i^T - S_t S_t^T.
        S_pred = torch.einsum("bt,btd->bd", p, self.Z)                   # (B, d)
        ZpZ = torch.einsum("bt,btd,bte->bde", p, self.Z, self.Z)         # (B, d, d)
        outer = S_pred.unsqueeze(-1) * S_pred.unsqueeze(-2)               # (B, d, d)
        bracket = (ZpZ - outer) / math.sqrt(self.d)                       # (B, d, d)

        # dS = phi @ bracket — phi as a row vector
        dS = torch.einsum("bd,bde->be", phi_v, bracket)                   # (B, d)

        return torch.cat([dz, dS], dim=-1)


class FaithfulDIFFODE(nn.Module):
    """DIFFODE with joint (z, S) integration and Hoyer sparsity in loss.

    Forward returns predictions and *also* exposes the average Hoyer
    score of the trace, which the training loop can subtract from MSE.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,           # d
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
        self.d = hidden_dim
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
        self.field = _JointDIFFODEField(hidden_dim, mlp_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.SiLU(),
            nn.Linear(mlp_dim, out_dim),
        )

        # Hoyer term collected on each forward pass — training loop reads it.
        self.last_hoyer: Tensor | None = None

    def forward(self, x_obs: Tensor, t_obs: Tensor, t_query: Tensor) -> Tensor:
        from torchdiffeq import odeint, odeint_adjoint

        Z = self.encoder(x_obs, t_obs)                                   # (B, T, d)
        B, T, d = Z.shape

        # Initial state at t_obs[-1]: z_0 = Z[:, -1, :]; S_0 = attention at z_0
        z0 = Z[:, -1, :]                                                  # (B, d)
        logits0 = torch.einsum("bd,btd->bt", z0, Z) / math.sqrt(d)
        p0 = torch.softmax(logits0, dim=-1)
        S0 = torch.einsum("bt,btd->bd", p0, Z)                            # (B, d)
        state0 = torch.cat([z0, S0], dim=-1)                              # (B, 2d)

        # Inject keys + reset p-trace.
        self.field.set_keys(Z)
        self.field.start_recording()

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
        state_traj = solver_fn(
            self.field, state0, t_full,
            method=self.solver, rtol=self.rtol, atol=self.atol,
        )                                                                # (>=Q, B, 2d)
        state_traj = state_traj[drop:]                                    # (Q, B, 2d)

        # Extract S_t and decode.
        S_traj = state_traj[..., d:]                                      # (Q, B, d)
        y = self.decoder(S_traj)                                          # (Q, B, out_dim)

        # Aggregate Hoyer for the loss. Keep gradients so the training loop
        # can subtract lambda * hoyer from MSE and actually drive sparsity.
        if self.field.p_trace:
            p_all = torch.stack(self.field.p_trace, dim=0)                # (steps, B, T)
            self.last_hoyer = hoyer_sparsity(p_all).mean()
        else:
            self.last_hoyer = None
        self.field.stop_recording()
        return y
