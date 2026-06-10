"""Irregular-timestamp encoder ``ψ : (x_t, t, E(x_t)) → z_t``.

Per ``docs/qa_design_choices.md`` Q2, the output ``z_t ∈ R^N`` is
interpreted as a **length-N local time-domain patch** centered at ``t``
— the encoder learns "what the signal looks like near ``t``." That
interpretation is what makes STFT(z_t) well-defined and what forces the
encoder latent dimension to equal the ODE-state dimension ``N``.

The current implementation is a plain MLP over the concatenation of:

* the observed value ``x_t``,
* a sinusoidal embedding of the (continuous) timestamp ``t``,
* optional external features ``E(x_t)``.

A more elaborate encoder (e.g. a small Transformer that conditions on
neighbouring observations) could replace this without changing any
downstream code — the contract is just "input is per-observation, output
is shape ``(B, T, N)`` in ``R^N``."
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


def sinusoidal_time_embedding(t: Tensor, dim: int, max_period: float = 10000.0) -> Tensor:
    """Standard sinusoidal positional encoding adapted to continuous t.

    Args:
        t: shape ``(...,)`` — real-valued timestamps.
        dim: embedding dimension (must be even).
        max_period: largest period in the sin/cos basis.

    Returns:
        Tensor of shape ``(..., dim)``.
    """
    if dim % 2 != 0:
        raise ValueError(f"time_emb_dim must be even, got {dim}")
    half = dim // 2
    # log-spaced angular frequencies
    inv_freq = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=t.dtype) / half
    )
    angles = t.unsqueeze(-1) * inv_freq  # (..., half)
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


class IrregularEncoder(nn.Module):
    """``ψ : (x_t, t, [E(x_t)]) → z_t ∈ R^N``.

    Args:
        in_dim: feature dimension of ``x_t``.
        time_emb_dim: dimension of the sinusoidal time embedding.
        d_latent: output dimension ``N`` (the patch / ODE-state length).
        hidden_dim: width of the internal MLP hidden layer.
        aux_dim: dimension of the optional external feature ``E(x_t)``;
            set to ``0`` to disable.
    """

    def __init__(
        self,
        in_dim: int,
        time_emb_dim: int,
        d_latent: int,
        hidden_dim: int = 64,
        aux_dim: int = 0,
    ) -> None:
        super().__init__()
        if time_emb_dim % 2 != 0:
            raise ValueError(f"time_emb_dim must be even, got {time_emb_dim}")
        self.in_dim = in_dim
        self.time_emb_dim = time_emb_dim
        self.aux_dim = aux_dim
        self.d_latent = d_latent

        in_total = in_dim + time_emb_dim + aux_dim
        self.net = nn.Sequential(
            nn.Linear(in_total, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, d_latent),
        )

    def forward(self, x: Tensor, t: Tensor, aux: Tensor | None = None) -> Tensor:
        """
        Args:
            x:   ``(B, T, in_dim)`` observed values.
            t:   ``(B, T)`` timestamps.
            aux: optional ``(B, T, aux_dim)`` external features.

        Returns:
            ``(B, T, d_latent)`` — per-observation length-``d_latent``
            time-domain patches.
        """
        if x.dim() != 3 or t.dim() != 2:
            raise ValueError(f"expected x:(B,T,in_dim) and t:(B,T), got {x.shape}, {t.shape}")
        if x.shape[:2] != t.shape:
            raise ValueError(f"x and t batch/time shapes mismatch: {x.shape} vs {t.shape}")
        if x.shape[-1] != self.in_dim:
            raise ValueError(f"x.in_dim={x.shape[-1]} != self.in_dim={self.in_dim}")

        phi_t = sinusoidal_time_embedding(t, dim=self.time_emb_dim)  # (B, T, time_emb_dim)
        parts = [x, phi_t]
        if self.aux_dim > 0:
            if aux is None:
                raise ValueError(f"aux required (aux_dim={self.aux_dim})")
            if aux.shape[-1] != self.aux_dim:
                raise ValueError(f"aux.last_dim={aux.shape[-1]} != aux_dim={self.aux_dim}")
            parts.append(aux)
        elif aux is not None:
            raise ValueError("aux provided but aux_dim=0 — disable aux or set aux_dim")

        feat = torch.cat(parts, dim=-1)
        return self.net(feat)
