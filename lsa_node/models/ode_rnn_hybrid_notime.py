"""ODE-RNN-Hybrid-NoTime — parallel paths *without* time conditioning.

Isolates whether the time embedding (φ(t) injected into both MLPs) is
the cause of the aperiodic regression we see in ``ODE_RNN_Hybrid``.

The freq branch's MLP sees only the spectrum; the time branch's MLP
sees only the hidden state. Both still have zero-init last layers.

    f(h, t) = MLP_time(h) + IFFT( M( g_θ( P(FFT(h)) ) ) )

Expected behaviour:
- If the time embedding was hurting autonomous systems, this should
  *improve* on the standard Hybrid for LV / glycolytic / Lorenz.
- On periodic_3d_a (explicitly time-varying signal: sin(t), cos(t)),
  this should slightly *underperform* the standard Hybrid.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .filter_k import FilterK


class _TimeBranchNoT(nn.Module):
    """``MLP(h)`` with zero-init last layer — no time conditioning."""

    def __init__(self, hidden_dim: int, mlp_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.SiLU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.SiLU(),
            nn.Linear(mlp_dim, hidden_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, t: Tensor, h: Tensor) -> Tensor:
        del t
        return self.net(h)


class _FreqBranchNoT(nn.Module):
    """``IFFT(M(g_θ(P(FFT(h)))))`` with zero-init last layer — no time."""

    def __init__(self, n: int, mlp_dim: int, depth: int) -> None:
        super().__init__()
        self.n = n
        self.n_freqs = n // 2 + 1
        c = 2 * self.n_freqs
        layers: list[nn.Module] = [nn.Linear(c, mlp_dim), nn.ReLU()]
        for _ in range(depth - 2):
            layers += [nn.Linear(mlp_dim, mlp_dim), nn.ReLU()]
        layers += [nn.Linear(mlp_dim, c)]
        self.mlp = nn.Sequential(*layers)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def _enforce_conj_sym(self, z: Tensor) -> Tensor:
        re, im = z.real.clone(), z.imag.clone()
        im[..., 0] = 0.0
        if self.n % 2 == 0:
            im[..., -1] = 0.0
        return torch.complex(re, im)

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        del t
        X = torch.fft.rfft(x, n=self.n)
        X_info = torch.cat([X.real, X.imag], dim=-1)
        Z_info = self.mlp(X_info)
        Z_re, Z_im = Z_info.split(self.n_freqs, dim=-1)
        Z = self._enforce_conj_sym(torch.complex(Z_re, Z_im))
        return torch.fft.irfft(Z, n=self.n)


class _HybridFieldNoT(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        time_mlp_dim: int,
        fode_mlp_dim: int,
        fode_depth: int,
    ) -> None:
        super().__init__()
        self.time_branch = _TimeBranchNoT(hidden_dim, time_mlp_dim)
        self.freq_branch = _FreqBranchNoT(
            n=hidden_dim, mlp_dim=fode_mlp_dim, depth=fode_depth
        )

    def forward(self, t: Tensor, h: Tensor) -> Tensor:
        return self.time_branch(t, h) + self.freq_branch(t, h)


class ODE_RNN_Hybrid_NoTime(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        time_mlp_dim: int = 64,
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
        self.field = _HybridFieldNoT(
            hidden_dim=hidden_dim,
            time_mlp_dim=time_mlp_dim,
            fode_mlp_dim=fode_mlp_dim,
            fode_depth=fode_depth,
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
