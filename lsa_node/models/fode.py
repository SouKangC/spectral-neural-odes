"""Faithful FODE (Guo & Weng 2025) — vanilla, no STFT, no attention.

Vector field per FODE paper Eq. 10:

    f_FODE(x, t; θ_g) = IFFT( M( g_θ( P( FFT(x) ), t ) ) )

where ``P`` concatenates real and imaginary parts, ``g_θ`` is a 3-layer
MLP, ``M`` enforces conjugate symmetry. The state ``x`` is interpreted
as a length-``N`` 1-D signal; FFT is *global* (not STFT).

We use this as a strong baseline rather than as our own method —
contrast against LSA-NODE / STFT-FODE.

Reported FODE numbers we should be near (Table II MAPE %):
* Unstable Oscillator: 8.98 ± 1.21
* Forced Vibration: 1.34 ± 0.61
* Lotka–Volterra: 1.87 ± 0.09
* Glycolytic Oscillator: 0.51 ± 0.04
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .filter_k import FilterK


def _conjugate_symmetry_complex(z: Tensor, n: int) -> Tensor:
    """Enforce DC.imag = 0 (and Nyquist.imag = 0 if ``n`` is even).

    ``z`` is the half-spectrum (output of ``torch.fft.rfft``), shape
    ``(..., n//2 + 1)``. Returns a tensor of the same shape with the
    invalid imaginary entries zeroed.
    """
    re, im = z.real.clone(), z.imag.clone()
    im[..., 0] = 0.0                                # DC
    if n % 2 == 0:
        im[..., -1] = 0.0                           # Nyquist
    return torch.complex(re, im)


class _FODEField(nn.Module):
    """``g_θ`` per FODE Eq. 10 — MLP on the concatenated (Re, Im) spectrum."""

    def __init__(self, n: int, mlp_dim: int = 16, depth: int = 3) -> None:
        super().__init__()
        n_freqs = n // 2 + 1
        c = 2 * n_freqs                              # real + imag concat
        self.n = n
        self.n_freqs = n_freqs

        layers: list[nn.Module] = [nn.Linear(c, mlp_dim), nn.ReLU()]
        for _ in range(depth - 2):
            layers += [nn.Linear(mlp_dim, mlp_dim), nn.ReLU()]
        layers += [nn.Linear(mlp_dim, c)]
        self.mlp = nn.Sequential(*layers)
        # Zero-init last layer so the block starts as f ≈ 0.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        del t
        # x: (..., n) — interpret as a 1-D signal of length n
        X = torch.fft.rfft(x, n=self.n)              # complex (..., n_freqs)
        X_info = torch.cat([X.real, X.imag], dim=-1) # real (..., 2*n_freqs)
        Z_info = self.mlp(X_info)
        Z_re, Z_im = Z_info.split(self.n_freqs, dim=-1)
        Z = torch.complex(Z_re, Z_im)
        Z = _conjugate_symmetry_complex(Z, n=self.n)
        return torch.fft.irfft(Z, n=self.n)          # real (..., n)


class FODE(nn.Module):
    """Faithful FODE for forecasting.

    Encoder lifts the last observation to a length-``hidden_dim`` signal
    space; FODE vector field evolves it via global FFT + MLP; filter K
    and decoder produce the per-timestep prediction.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        mlp_dim: int = 16,
        depth: int = 3,
        solver: str = "dopri5",
        rtol: float = 1.0e-3,
        atol: float = 1.0e-4,
        use_adjoint: bool = False,
        # ignored kwargs to share YAML configs across models
        **_ignored,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.use_adjoint = use_adjoint

        self.encoder = nn.Linear(in_dim, hidden_dim)
        self.field = _FODEField(n=hidden_dim, mlp_dim=mlp_dim, depth=depth)
        self.filter_k = FilterK(hidden_dim=hidden_dim, init="uniform")
        self.decoder = nn.Linear(hidden_dim, out_dim)

    def forward(self, x_obs: Tensor, t_obs: Tensor, t_query: Tensor) -> Tensor:
        del t_obs
        h0 = self.encoder(x_obs[:, -1, :])                           # (B, N)
        t_query = t_query.to(device=h0.device, dtype=h0.dtype)
        from torchdiffeq import odeint, odeint_adjoint
        solver_fn = odeint_adjoint if self.use_adjoint else odeint
        h = solver_fn(self.field, h0, t_query,
                      method=self.solver, rtol=self.rtol, atol=self.atol)
        h_filt = self.filter_k(h)
        return self.decoder(h_filt)
