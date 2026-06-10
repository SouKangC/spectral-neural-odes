"""ODE-RNN-Hybrid — time-domain + spectral parallel paths with time conditioning.

E006 found that ``ODE_RNN_FODE`` wins on cells with explicit periodic
frequency structure (`periodic_3d_a`) but loses on aperiodic / chaotic
cells where FODE's spectral basis is the wrong inductive bias.

The fix: don't *replace* the time-domain MLP with the spectral field —
*add* it as a residual branch. Both branches' final layers are
zero-initialised so the vector field starts at 0 and the model learns
to use whichever path helps per cell.

In addition we restore time conditioning that FODE paper Eq. 10
explicitly uses but our previous ``_FODEField`` ignored: the MLP now
sees a sinusoidal embedding of the integration time ``t`` alongside the
spectrum.

    f(h, t) = MLP_time( h, φ(t) )  +  IFFT( M( g_θ( P(FFT(h)), φ(t) ) ) )

Both MLP_time and g_θ have zero-init output layers so the model behaves
as pure ODE-RNN at the start of training and can selectively engage the
spectral path during optimisation.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from .filter_k import FilterK


# ---------------------------------------------------------------------------
# Time embedding (sinusoidal, shared by both branches)
# ---------------------------------------------------------------------------
def _time_emb(t: Tensor, dim: int = 16, max_period: float = 10000.0) -> Tensor:
    """Sinusoidal embedding of a scalar ``t`` returning shape ``(dim,)``."""
    if dim % 2 != 0:
        raise ValueError(f"time_emb_dim must be even, got {dim}")
    half = dim // 2
    device = t.device
    dtype = t.dtype if t.dtype.is_floating_point else torch.float32
    inv_freq = torch.exp(
        -math.log(max_period) * torch.arange(half, device=device, dtype=dtype) / half
    )
    angles = t.to(dtype) * inv_freq                     # (half,)
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


# ---------------------------------------------------------------------------
# Two parallel branches
# ---------------------------------------------------------------------------
class _TimeBranch(nn.Module):
    """``f_time(h, φ(t)) = MLP([h; φ(t)])`` — zero-init last layer."""

    def __init__(self, hidden_dim: int, mlp_dim: int, time_emb_dim: int) -> None:
        super().__init__()
        self.time_emb_dim = time_emb_dim
        in_dim = hidden_dim + time_emb_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, mlp_dim),
            nn.SiLU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.SiLU(),
            nn.Linear(mlp_dim, hidden_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, t: Tensor, h: Tensor) -> Tensor:
        emb = _time_emb(t, dim=self.time_emb_dim).to(h.dtype)
        emb = emb.expand(*h.shape[:-1], self.time_emb_dim)
        return self.net(torch.cat([h, emb], dim=-1))


class _FreqBranch(nn.Module):
    """``f_freq(h, t) = IFFT( M( g_θ([P(FFT(h)); φ(t)]) ) )`` — zero-init last layer.

    Faithful to FODE Eq. 10 including the time conditioning that our
    previous ``_FODEField`` skipped.
    """

    def __init__(
        self,
        n: int,
        mlp_dim: int,
        depth: int,
        time_emb_dim: int,
    ) -> None:
        super().__init__()
        self.n = n
        self.n_freqs = n // 2 + 1
        self.time_emb_dim = time_emb_dim

        c = 2 * self.n_freqs                            # real + imag concat
        in_dim = c + time_emb_dim
        layers: list[nn.Module] = [nn.Linear(in_dim, mlp_dim), nn.ReLU()]
        for _ in range(depth - 2):
            layers += [nn.Linear(mlp_dim, mlp_dim), nn.ReLU()]
        layers += [nn.Linear(mlp_dim, c)]
        self.mlp = nn.Sequential(*layers)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def _enforce_conj_sym(self, z: Tensor) -> Tensor:
        re, im = z.real.clone(), z.imag.clone()
        im[..., 0] = 0.0                                # DC
        if self.n % 2 == 0:
            im[..., -1] = 0.0                           # Nyquist
        return torch.complex(re, im)

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        X = torch.fft.rfft(x, n=self.n)                 # complex (..., n_freqs)
        X_info = torch.cat([X.real, X.imag], dim=-1)    # real (..., 2*n_freqs)
        emb = _time_emb(t, dim=self.time_emb_dim).to(X_info.dtype)
        emb = emb.expand(*X_info.shape[:-1], self.time_emb_dim)
        Z_info = self.mlp(torch.cat([X_info, emb], dim=-1))
        Z_re, Z_im = Z_info.split(self.n_freqs, dim=-1)
        Z = self._enforce_conj_sym(torch.complex(Z_re, Z_im))
        return torch.fft.irfft(Z, n=self.n)             # (..., n)


# ---------------------------------------------------------------------------
# Hybrid vector field
# ---------------------------------------------------------------------------
class _HybridField(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        time_mlp_dim: int,
        fode_mlp_dim: int,
        fode_depth: int,
        time_emb_dim: int,
    ) -> None:
        super().__init__()
        self.time_branch = _TimeBranch(hidden_dim, time_mlp_dim, time_emb_dim)
        self.freq_branch = _FreqBranch(
            n=hidden_dim, mlp_dim=fode_mlp_dim, depth=fode_depth, time_emb_dim=time_emb_dim
        )

    def forward(self, t: Tensor, h: Tensor) -> Tensor:
        return self.time_branch(t, h) + self.freq_branch(t, h)


# ---------------------------------------------------------------------------
# Top-level model: ODE-RNN encoder + hybrid vector field
# ---------------------------------------------------------------------------
class ODE_RNN_Hybrid(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        time_mlp_dim: int = 64,
        fode_mlp_dim: int = 16,
        fode_depth: int = 3,
        time_emb_dim: int = 16,
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
        self.field = _HybridField(
            hidden_dim=hidden_dim,
            time_mlp_dim=time_mlp_dim,
            fode_mlp_dim=fode_mlp_dim,
            fode_depth=fode_depth,
            time_emb_dim=time_emb_dim,
        )
        self.filter_k = FilterK(hidden_dim=hidden_dim, init="uniform")
        self.decoder = nn.Linear(hidden_dim, out_dim)

    def forward(self, x_obs: Tensor, t_obs: Tensor, t_query: Tensor) -> Tensor:
        from torchdiffeq import odeint, odeint_adjoint
        solver_fn = odeint_adjoint if self.use_adjoint else odeint
        B, T, _ = x_obs.shape
        t_grid = t_obs[0].to(x_obs.dtype)

        # Encode the irregular observation window.
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

        # Forecast.
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
