"""ODE-RNN-Gated — learnable gate on the spectral branch.

E007 (Hybrid) showed that even with zero-init the spectral branch
drifts away from 0 during training and acts as noise on aperiodic
cells. This variant adds a learnable scalar gate ``α`` (sigmoid'd) on
the frequency branch:

    f(h, t) = MLP_time([h; φ(t)])  +  σ(g) · IFFT(M(g_θ([P(FFT(h)); φ(t)])))

We initialise the raw gate ``g`` so ``σ(g) ≈ 0`` (e.g. g = -6 gives
σ ≈ 0.0025). The optimiser can either push g up (engaging the spectral
path) or keep it down (suppressing it). On cells where the spectral
basis is mismatched, gradients on the spectral path will be small and
g will stay negative — the freq branch effectively turns off.

If this variant matches ODE-RNN on aperiodic cells while keeping the
hybrid's periodic_3d_a wins, we have a model that is *strictly ≥ both
parents* per cell.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .filter_k import FilterK
from .ode_rnn_hybrid import _FreqBranch, _TimeBranch


class _GatedHybridField(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        time_mlp_dim: int,
        fode_mlp_dim: int,
        fode_depth: int,
        time_emb_dim: int,
        gate_init: float = -6.0,
    ) -> None:
        super().__init__()
        self.time_branch = _TimeBranch(hidden_dim, time_mlp_dim, time_emb_dim)
        self.freq_branch = _FreqBranch(
            n=hidden_dim, mlp_dim=fode_mlp_dim, depth=fode_depth, time_emb_dim=time_emb_dim
        )
        # raw gate, σ(gate_raw) gives the actual coefficient ∈ (0, 1)
        self.gate_raw = nn.Parameter(torch.tensor(gate_init, dtype=torch.float32))

    @property
    def gate(self) -> Tensor:
        return torch.sigmoid(self.gate_raw)

    def forward(self, t: Tensor, h: Tensor) -> Tensor:
        return self.time_branch(t, h) + self.gate * self.freq_branch(t, h)


class ODE_RNN_Gated(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        time_mlp_dim: int = 64,
        fode_mlp_dim: int = 16,
        fode_depth: int = 3,
        time_emb_dim: int = 16,
        gate_init: float = -6.0,
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
        self.field = _GatedHybridField(
            hidden_dim=hidden_dim,
            time_mlp_dim=time_mlp_dim,
            fode_mlp_dim=fode_mlp_dim,
            fode_depth=fode_depth,
            time_emb_dim=time_emb_dim,
            gate_init=gate_init,
        )
        self.filter_k = FilterK(hidden_dim=hidden_dim, init="uniform")
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
        z_filt = self.filter_k(z)
        return self.decoder(z_filt)
