"""Top-level LSA-NODE model.

Wires together:

    irregular encoder  →  spectral dictionary  →  Neural ODE
        →  filter K  →  decoder

Forward pass (regression):

    z = encoder(x_obs, t_obs)                       # (B, T, N)
    K, V = build_dictionary(z)                       # (B, T*L, 2*n_freqs)
    odefunc.set_dictionary(K, V, signal_length=N)
    h0 = z[:, -1]                                    # (B, N) — seed the ODE
    h = odeint(odefunc, h0, t_query)                 # (Q, B, N)
    h_filt = filter_k(h)
    y = decoder(h_filt)                              # (Q, B, out_dim)

Per ``docs/qa_design_choices.md`` Q2, encoder output dim and ODE-state dim
are tied — both equal ``hidden_dim = N``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .decoder import RegressionDecoder
from .encoder import IrregularEncoder
from .filter_k import FilterK
from .ode_func import ODEFunc
from .stft_attention import stft


class LSANODE(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,        # N — patch / ODE-state length
        n_fft: int,
        hop_length: int,
        d_att: int,
        n_heads: int = 4,
        time_emb_dim: int = 16,
        encoder_hidden: int = 64,
        decoder_hidden: int = 64,
        solver: str = "dopri5",
        rtol: float = 1e-5,
        atol: float = 1e-7,
        use_adjoint: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dim < n_fft:
            raise ValueError(
                f"hidden_dim (={hidden_dim}) must be >= n_fft (={n_fft}) "
                "so the STFT has at least one frame"
            )
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.n_fft = n_fft
        self.hop_length = hop_length
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
        self.odefunc = ODEFunc(
            n_fft=n_fft,
            hop_length=hop_length,
            d_model=d_att,
            n_heads=n_heads,
        )
        self.filter_k = FilterK(hidden_dim=hidden_dim, init="uniform")
        self.head = RegressionDecoder(
            hidden_dim=hidden_dim, out_dim=out_dim, mlp_dim=decoder_hidden
        )

    # ---- dictionary builder ---------------------------------------------
    def build_dictionary(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """Spectral dictionary from per-observation patches.

        Args:
            z: ``(B, T, N)`` — per-observation length-``N`` time-domain
                patches from the encoder.

        Returns:
            ``(K_spec, V_spec)`` both of shape ``(B, T*L, 2 * n_freqs)``.
        """
        B, T, N = z.shape
        assert N == self.hidden_dim, f"z last dim {N} != hidden_dim {self.hidden_dim}"
        # (B*T, N) → (B*T, L, 2*n_freqs)
        H = stft(
            z.reshape(B * T, N),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        L = H.shape[1]
        cdim = H.shape[-1]
        H = H.reshape(B, T * L, cdim)
        # K and V share the same source; the W_K / W_V projections inside
        # the attention block differentiate them.
        return H, H

    # ---- forward --------------------------------------------------------
    def forward(
        self,
        x_obs: Tensor,
        t_obs: Tensor,
        t_query: Tensor,
    ) -> Tensor:
        """
        Args:
            x_obs:   ``(B, T, in_dim)`` observed values.
            t_obs:   ``(B, T)`` observation timestamps.
            t_query: ``(Q,)`` timestamps at which we want predictions.
                Must be monotonically increasing for ``torchdiffeq``.

        Returns:
            ``(Q, B, out_dim)`` predictions at the query times.
        """
        if x_obs.dim() != 3 or t_obs.dim() != 2:
            raise ValueError(f"x_obs (B,T,in_dim) and t_obs (B,T); got {x_obs.shape}, {t_obs.shape}")
        if t_query.dim() != 1:
            raise ValueError(f"t_query must be 1-D, got shape {t_query.shape}")

        z = self.encoder(x_obs, t_obs)                          # (B, T, N)
        K_spec, V_spec = self.build_dictionary(z)
        # The dictionary must remain attached through the adjoint backward
        # solve (which re-evaluates the vector field). We rely on the
        # next forward to overwrite it; no `clear_dictionary()` here.
        self.odefunc.set_dictionary(K_spec, V_spec, signal_length=self.hidden_dim)
        h0 = z[:, -1, :]                                        # (B, N) — last patch
        t_query = t_query.to(device=h0.device, dtype=h0.dtype)

        # Lazy import so unit tests without torchdiffeq still work.
        from torchdiffeq import odeint, odeint_adjoint

        solver_fn = odeint_adjoint if self.use_adjoint else odeint
        h = solver_fn(
            self.odefunc,
            h0,
            t_query,
            method=self.solver,
            rtol=self.rtol,
            atol=self.atol,
        )                                                       # (Q, B, N)

        h_filt = self.filter_k(h)                                # (Q, B, N)
        y = self.head(h_filt)                                    # (Q, B, out_dim)
        return y
