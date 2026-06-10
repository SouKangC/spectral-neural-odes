"""STFT-FODE ablation.

FODE (Guo & Weng 2025) parameterises the Neural-ODE vector field via a
dense MLP on the full spectrum:

    f_FODE(x, t) = IFFT( g_θ(FFT(x), t) )

This ablation keeps FODE's mechanism but swaps the global FFT for the
short-time variant LSA-NODE uses, *and removes the attention block*:

    f_STFT-FODE(h, t) = ISTFT( MLP_per_frame(STFT(h)) )

If this beats the current LSA-NODE on the fast grid, the
attention+dictionary part of LSA-NODE is the load-bearing failure mode.
See docs/experiments/E003 §Decision and the investigation thread.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from .filter_k import FilterK
from .stft_attention import conjugate_symmetry_projection, istft, stft


class STFTFODEFunc(nn.Module):
    """Per-frame MLP on the half-spectrum representation."""

    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        mlp_dim: int = 128,
        depth: int = 3,
    ) -> None:
        super().__init__()
        c = 2 * (n_fft // 2 + 1)
        self.n_fft = n_fft
        self.hop_length = hop_length
        layers: list[nn.Module] = [nn.Linear(c, mlp_dim), nn.SiLU()]
        for _ in range(depth - 2):
            layers += [nn.Linear(mlp_dim, mlp_dim), nn.SiLU()]
        layers += [nn.Linear(mlp_dim, c)]
        self.mlp = nn.Sequential(*layers)
        # Zero-init last layer ⇒ block starts as STFT→identity→ISTFT, i.e.
        # f(h, t) ≈ 0 at init; the model has to climb out by learning.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, t: Tensor, h: Tensor) -> Tensor:
        del t
        N = h.shape[-1]
        H = stft(h, n_fft=self.n_fft, hop_length=self.hop_length)   # (B, L, c)
        Z = self.mlp(H)                                              # same shape
        Z = conjugate_symmetry_projection(Z, n_fft=self.n_fft)
        return istft(Z, n_fft=self.n_fft, hop_length=self.hop_length, length=N)


class STFTFODE(nn.Module):
    """Same skeleton as :class:`~lsa_node.models.lsa_node.LSANODE` but with
    :class:`STFTFODEFunc` as the vector field — no attention, no dictionary.

    Encoder produces a length-``hidden_dim`` patch from the last
    observation (single linear layer; we don't need the per-observation
    dictionary anymore). Decoder + filter K mirror LSANODE for parity.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        n_fft: int,
        hop_length: int,
        mlp_dim: int = 128,
        depth: int = 3,
        solver: str = "dopri5",
        rtol: float = 1e-3,
        atol: float = 1e-4,
        use_adjoint: bool = False,
        # The remaining kwargs are accepted-and-ignored so the same YAML
        # config used for LSANODE works here (d_att, n_heads, etc.).
        **_ignored,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.use_adjoint = use_adjoint

        # Trivial encoder: lift the last observation into the patch space.
        # We can't share LSANODE's per-window patch encoder cleanly without
        # the dictionary — so just project the last obs to R^N.
        self.encoder = nn.Linear(in_dim, hidden_dim)
        self.odefunc = STFTFODEFunc(
            n_fft=n_fft, hop_length=hop_length, mlp_dim=mlp_dim, depth=depth
        )
        self.filter_k = FilterK(hidden_dim=hidden_dim, init="uniform")
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x_obs: Tensor, t_obs: Tensor, t_query: Tensor) -> Tensor:
        del t_obs
        h0 = self.encoder(x_obs[:, -1, :])                          # (B, N)
        t_query = t_query.to(device=h0.device, dtype=h0.dtype)
        from torchdiffeq import odeint, odeint_adjoint
        solver_fn = odeint_adjoint if self.use_adjoint else odeint
        h = solver_fn(self.odefunc, h0, t_query,
                      method=self.solver, rtol=self.rtol, atol=self.atol)  # (Q, B, N)
        h_filt = self.filter_k(h)
        return self.head(h_filt)
