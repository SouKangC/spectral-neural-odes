"""Neural ODE vector field ``f_θ(h, t)`` parameterized by STFT-attention.

See ``docs/ideas.md`` §5.3 and ``docs/qa_design_choices.md`` Q1/Q2.

Each call to ``forward`` is one solver evaluation. The pipeline is:

    h(t) ∈ R^{B, N}
      ├── STFT  ──► H^{(t)} ∈ R^{B, L, 2·n_freqs}       (queries)
      │
      │  attention against precomputed dictionary (K, V)
      │
      ├── (W_O, residual MLP_g)
      │
      ├── conjugate-symmetry projection M
      │
      └── ISTFT  ──► dh/dt ∈ R^{B, N}

The dictionary ``(K, V)`` is computed once per sequence by ``LSANODE`` and
attached to this module via :meth:`set_dictionary` before ``odeint`` is
called. The dictionary contract is "channels are STFT frames stacked over
all observations; RoPE position within each observation's local window
group."
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from .stft_attention import (
    STFTAttentionBlock,
    conjugate_symmetry_projection,
    istft,
    stft,
)


class ODEFunc(nn.Module):
    """``f_θ(h, t)`` for ``torchdiffeq.odeint``."""

    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        d_model: int,
        n_heads: int = 4,
        mlp_dim: int | None = None,
        dropout: float = 0.0,
        rope_max_pos: int = 512,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.d_model = d_model
        self.block = STFTAttentionBlock(
            n_fft=n_fft,
            d_model=d_model,
            n_heads=n_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            max_pos=rope_max_pos,
        )

        self._K: Tensor | None = None
        self._V: Tensor | None = None
        self._N: int | None = None

    # ---- dictionary lifecycle --------------------------------------------
    def set_dictionary(self, K_spec: Tensor, V_spec: Tensor, signal_length: int) -> None:
        """Attach the per-sequence spectral dictionary.

        Args:
            K_spec: ``(B, M, 2 * n_freqs)`` — half-spectrum frames (real/imag stacked),
                typically ``M = T * L`` if frames from all observations are flattened.
            V_spec: same shape as ``K_spec``.
            signal_length: ``N`` — length of the ODE state ``h(t)``; ISTFT
                needs this to disambiguate the inverse.
        """
        expected_c = 2 * (self.n_fft // 2 + 1)
        if K_spec.shape[-1] != expected_c or V_spec.shape[-1] != expected_c:
            raise ValueError(
                f"K/V last dim must be {expected_c} (= 2 * (n_fft // 2 + 1))"
            )
        self._K = K_spec
        self._V = V_spec
        self._N = int(signal_length)

    def clear_dictionary(self) -> None:
        self._K = None
        self._V = None
        self._N = None

    # ---- the vector field -----------------------------------------------
    def forward(self, t: Tensor, h: Tensor) -> Tensor:
        """Compute ``dh/dt`` at the current solver point.

        Args:
            t: scalar tensor (current time) — accepted for ``torchdiffeq``
                compatibility but unused by the block. Time conditioning
                could be added by feeding ``t`` into MLP_g.
            h: ``(B, N)`` hidden state.

        Returns:
            ``(B, N)`` — the time derivative.
        """
        del t
        if self._K is None or self._V is None or self._N is None:
            raise RuntimeError(
                "set_dictionary() must be called before invoking the ODE solver"
            )
        if h.dim() != 2:
            raise ValueError(f"h must be (B, N), got shape {h.shape}")

        N = h.shape[-1]
        # 1. local STFT of the current state.
        H_q = stft(h, n_fft=self.n_fft, hop_length=self.hop_length)  # (B, L, 2*n_freqs)

        # 2. attend against the dictionary.
        Z = self.block(H_q, self._K, self._V)  # (B, L, 2*n_freqs)

        # 3. enforce conjugate symmetry for a real ISTFT output.
        Z = conjugate_symmetry_projection(Z, n_fft=self.n_fft)

        # 4. inverse STFT back to time domain.
        dh = istft(Z, n_fft=self.n_fft, hop_length=self.hop_length, length=N)
        return dh
