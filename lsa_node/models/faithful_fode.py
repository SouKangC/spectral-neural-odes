"""Faithful FODE re-implementation (Guo & Weng 2025).

Differences from ``fode.py``:

1. **Vector field receives time t.** Paper Eq. 10: ``g_θ(P(FFT(x)), t)``.
   Our v1 dropped ``t`` (``del t`` in ``_FODEField.forward``). This
   restores non-autonomous behaviour.

2. **State dim defaults to N=16.** The paper uses N=16; we used 64 in
   v1 to match the rest of our family. Re-running at N=16 with the
   paper's mlp_dim=16, depth=3, ReLU should be closer to the published
   numbers.

3. **MLP keeps paper's spec exactly.** 3-layer ReLU, hidden=16, last
   layer NOT zero-initialised (paper doesn't say to — that was our
   stability hack).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .filter_k import FilterK


def _conjugate_symmetry_complex(z: Tensor, n: int) -> Tensor:
    re, im = z.real.clone(), z.imag.clone()
    im[..., 0] = 0.0
    if n % 2 == 0:
        im[..., -1] = 0.0
    return torch.complex(re, im)


class _FaithfulFODEField(nn.Module):
    """g_θ(P(FFT(x)), t) per paper Eq. 10 — MLP on spectrum WITH time."""

    def __init__(self, n: int, mlp_dim: int = 16, depth: int = 3) -> None:
        super().__init__()
        n_freqs = n // 2 + 1
        c = 2 * n_freqs
        self.n = n
        self.n_freqs = n_freqs

        # Input dim is c + 1 — we concat t as a scalar.
        layers: list[nn.Module] = [nn.Linear(c + 1, mlp_dim), nn.ReLU()]
        for _ in range(depth - 2):
            layers += [nn.Linear(mlp_dim, mlp_dim), nn.ReLU()]
        layers += [nn.Linear(mlp_dim, c)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        # x: (..., n)
        X = torch.fft.rfft(x, n=self.n)
        X_info = torch.cat([X.real, X.imag], dim=-1)             # (..., 2*n_freqs)
        # Concat t as an extra channel.
        t_b = t.expand(*X_info.shape[:-1], 1) if t.dim() == 0 \
              else t.view(*[1] * (X_info.dim() - 1), 1).expand(*X_info.shape[:-1], 1)
        X_with_t = torch.cat([X_info, t_b], dim=-1)              # (..., 2*n_freqs+1)
        Z_info = self.mlp(X_with_t)
        Z_re, Z_im = Z_info.split(self.n_freqs, dim=-1)
        Z = torch.complex(Z_re, Z_im)
        Z = _conjugate_symmetry_complex(Z, n=self.n)
        return torch.fft.irfft(Z, n=self.n)


class FaithfulFODE(nn.Module):
    """Faithful FODE for forecasting, paper-spec defaults."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 16,         # paper N
        mlp_dim: int = 16,            # paper hidden
        depth: int = 3,
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

        self.encoder = nn.Linear(in_dim, hidden_dim)
        self.field = _FaithfulFODEField(n=hidden_dim, mlp_dim=mlp_dim, depth=depth)
        self.filter_k = FilterK(hidden_dim=hidden_dim, init="uniform")
        self.decoder = nn.Linear(hidden_dim, out_dim)

    def forward(self, x_obs: Tensor, t_obs: Tensor, t_query: Tensor) -> Tensor:
        del t_obs
        h0 = self.encoder(x_obs[:, -1, :])
        t_query = t_query.to(device=h0.device, dtype=h0.dtype)
        from torchdiffeq import odeint, odeint_adjoint
        solver_fn = odeint_adjoint if self.use_adjoint else odeint
        h = solver_fn(self.field, h0, t_query,
                      method=self.solver, rtol=self.rtol, atol=self.atol)
        h_filt = self.filter_k(h)
        return self.decoder(h_filt)
