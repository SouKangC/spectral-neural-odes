"""STFT, ISTFT, and conjugate-symmetry projection for LSA-NODE.

See ``docs/02_fourier_stft.md`` and ``docs/ideas.md`` §5.2–§5.3.

The state ``h(t)`` is real-valued; we apply STFT inside the ODE vector
field, attend over the resulting (time, freq) grid, then ISTFT back.

### Frame representation

For a real signal we only need the half spectrum: ``n_freqs = n_fft // 2 + 1``
unique bins. We stack ``[Re(bins); Im(bins)]`` into a single real tensor of
width ``2 * n_freqs = n_fft + 2`` per frame. This:

* matches ``torch.stft`` / ``torch.istft`` conventions exactly,
* is autograd-friendly (no complex tensors crossing the attention block),
* halves the channel dim vs. carrying the redundant negative-frequency half.

### Conjugate-symmetry projection :math:`\\mathcal{M}`

For a real-valued ISTFT output we additionally need:

* DC bin (k = 0) has zero imaginary part;
* Nyquist bin (k = n_fft / 2, only present when ``n_fft`` is even) has
  zero imaginary part.

``conjugate_symmetry_projection`` enforces both. The rest of the half-
spectrum bins can be arbitrary complex values — their negative-frequency
mirrors are *implicit* in the half representation, so symmetry holds for
free.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .rope import apply_rope, rope_cache


# ---------------------------------------------------------------------------
# Window cache (avoids re-allocating Hann window on every STFT call).
# ---------------------------------------------------------------------------
_WINDOW_CACHE: dict[tuple[int, torch.device, torch.dtype], Tensor] = {}


def _get_window(
    n_fft: int,
    window: Tensor | None,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if window is not None:
        return window.to(device=device, dtype=dtype)
    key = (n_fft, device, dtype)
    w = _WINDOW_CACHE.get(key)
    if w is None:
        w = torch.hann_window(n_fft, device=device, dtype=dtype)
        _WINDOW_CACHE[key] = w
    return w


# ---------------------------------------------------------------------------
# STFT / ISTFT
# ---------------------------------------------------------------------------
def stft(
    x: Tensor,
    n_fft: int,
    hop_length: int,
    window: Tensor | None = None,
    center: bool = True,
) -> Tensor:
    """Real-valued STFT.

    Args:
        x: shape ``(..., N)`` — a 1-D real signal (leading dims are batch).
        n_fft: window length (number of frequency bins is ``n_fft // 2 + 1``).
        hop_length: stride between consecutive frames.
        window: optional 1-D window of length ``n_fft``; default Hann.
        center: pad ``x`` so the first frame is centered at ``t = 0``
            (matches ``torch.stft``'s default).

    Returns:
        Tensor of shape ``(..., L, 2 * (n_fft // 2 + 1))``. The last dim
        concatenates real and imaginary parts of the half spectrum.
        ``L`` is determined by ``N``, ``n_fft``, ``hop_length``, and ``center``.
    """
    if x.dim() == 0:
        raise ValueError("stft input must have at least one dim (the signal)")
    win = _get_window(n_fft, window, x.device, x.dtype)

    # Collapse leading dims so torch.stft sees a (B, N) tensor.
    lead = x.shape[:-1]
    flat = x.reshape(-1, x.shape[-1]) if lead else x[None]
    spec = torch.stft(
        flat,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=win,
        center=center,
        return_complex=True,
    )  # (B, n_freqs, L)
    # Move L before n_freqs and split real/imag.
    spec = spec.transpose(-1, -2)                       # (B, L, n_freqs)
    out = torch.cat([spec.real, spec.imag], dim=-1)     # (B, L, 2*n_freqs)
    out = out.reshape(*lead, *out.shape[-2:])
    return out


def istft(
    h_info: Tensor,
    n_fft: int,
    hop_length: int,
    window: Tensor | None = None,
    length: int | None = None,
    center: bool = True,
) -> Tensor:
    """Inverse of :func:`stft`.

    Args:
        h_info: shape ``(..., L, 2 * (n_fft // 2 + 1))`` — output of :func:`stft`.
        length: original signal length; required to disambiguate the inverse
            (``torch.istft`` won't otherwise know the un-padded tail).

    Returns:
        Tensor of shape ``(..., length)``.
    """
    n_freqs = n_fft // 2 + 1
    if h_info.shape[-1] != 2 * n_freqs:
        raise ValueError(
            f"h_info last dim must be 2*(n_fft//2+1)={2 * n_freqs}, got {h_info.shape[-1]}"
        )
    win = _get_window(n_fft, window, h_info.device, h_info.dtype)

    re, im = h_info[..., :n_freqs], h_info[..., n_freqs:]
    spec_complex = torch.complex(re, im).transpose(-1, -2)  # (..., n_freqs, L)

    lead = spec_complex.shape[:-2]
    flat = spec_complex.reshape(-1, *spec_complex.shape[-2:])

    out = torch.istft(
        flat,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=win,
        center=center,
        length=length,
        return_complex=False,
    )  # (B, length)
    return out.reshape(*lead, out.shape[-1])


def conjugate_symmetry_projection(h_info: Tensor, n_fft: int) -> Tensor:
    """Project ``h_info`` onto the subspace whose ISTFT is purely real.

    In the half-spectrum + real/imag-split representation this just means:

    * the DC bin's imaginary part is zero,
    * the Nyquist bin's imaginary part is zero (only if ``n_fft`` is even).

    All other (half-spectrum) bins are unconstrained.

    Args:
        h_info: ``(..., L, 2 * (n_fft // 2 + 1))``.
        n_fft: window length used to produce ``h_info``.

    Returns:
        Tensor of the same shape as ``h_info`` with the projection applied.
    """
    n_freqs = n_fft // 2 + 1
    if h_info.shape[-1] != 2 * n_freqs:
        raise ValueError(
            f"h_info last dim must be 2*(n_fft//2+1)={2 * n_freqs}, got {h_info.shape[-1]}"
        )

    # Build a multiplicative mask: 1 everywhere except the two imag entries
    # we need to zero out (DC.imag at index n_freqs, and Nyquist.imag at the
    # last index iff n_fft is even).
    mask = torch.ones(2 * n_freqs, device=h_info.device, dtype=h_info.dtype)
    mask[n_freqs] = 0.0  # DC imag
    if n_fft % 2 == 0:
        mask[-1] = 0.0  # Nyquist imag
    return h_info * mask


# ---------------------------------------------------------------------------
# Attention block — scaffold for Phase 2.7.
# ---------------------------------------------------------------------------
class STFTAttentionBlock(nn.Module):
    """Frequency-domain attention block used inside the ODE vector field.

    Pipeline (per forward call, see ``docs/ideas.md`` §5.3):

        q_spec (B, L_q, C) ─ W_Q ─► Q (B, L_q, d_att)
        k_spec (B, L_k, C) ─ W_K ─► K (B, L_k, d_att)
        v_spec (B, L_k, C) ─ W_V ─► V (B, L_k, d_att)

        Q, K  ──(RoPE on the position dim)──►  Q̃, K̃

        softmax(Q̃ K̃ᵀ / √d_head) V  ─ W_O ─► attn (B, L_q, C)

        out = attn + MLP_g(attn, t)            (residual; see qa_design_choices.md Q1)

    The ``q_spec``/``k_spec``/``v_spec`` inputs are the half-spectrum
    real/imag-stacked STFT frames, so the *channel* dimension carries
    frequency. Attention runs across STFT frames (tokens), not bins.

    The output dimension is matched to ``C = 2 * (n_fft // 2 + 1)`` so the
    result can be split back into real/imag halves for ISTFT.
    """

    def __init__(
        self,
        n_fft: int,
        d_model: int,
        n_heads: int = 4,
        mlp_dim: int | None = None,
        dropout: float = 0.0,
        max_pos: int = 512,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.n_fft = n_fft
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        if self.d_head % 2 != 0:
            raise ValueError(f"d_head={self.d_head} must be even for RoPE")

        c = 2 * (n_fft // 2 + 1)
        self.channel_dim = c

        self.W_q = nn.Linear(c, d_model, bias=False)
        self.W_k = nn.Linear(c, d_model, bias=False)
        self.W_v = nn.Linear(c, d_model, bias=False)
        self.W_o = nn.Linear(d_model, c, bias=False)

        # Residual MLP (zero-initialized output so the block starts as
        # identity-on-attention — stabilizes the ODE early in training;
        # see docs/qa_design_choices.md Q1).
        mlp_dim = mlp_dim or 2 * c
        self.mlp = nn.Sequential(
            nn.Linear(c, mlp_dim),
            nn.SiLU(),
            nn.Linear(mlp_dim, c),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

        self.dropout = dropout

        # Precompute RoPE tables up to ``max_pos``; resize automatically
        # if a longer sequence comes in.
        cos, sin = rope_cache(max_pos, self.d_head, base=rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self._rope_base = rope_base

    def _rope(self, x: Tensor, pos_start: int = 0) -> Tensor:
        """Apply RoPE rotation. ``x`` has shape ``(B, H, L, d_head)``;
        positions are ``[pos_start, pos_start + L)``."""
        L = x.shape[-2]
        if pos_start + L > self.rope_cos.shape[0]:
            # Grow the cache.
            cos, sin = rope_cache(pos_start + L, self.d_head, base=self._rope_base,
                                  device=x.device, dtype=x.dtype)
            self.rope_cos = cos
            self.rope_sin = sin
        cos = self.rope_cos[pos_start: pos_start + L].to(x.dtype)
        sin = self.rope_sin[pos_start: pos_start + L].to(x.dtype)
        return apply_rope(x, cos, sin)

    def forward(
        self,
        q_spec: Tensor,
        k_spec: Tensor,
        v_spec: Tensor,
        q_pos_start: int = 0,
        k_pos_start: int = 0,
    ) -> Tensor:
        """
        Args:
            q_spec: ``(B, L_q, C)`` — current state's STFT frames.
            k_spec: ``(B, L_k, C)`` — dictionary keys (precomputed).
            v_spec: ``(B, L_k, C)`` — dictionary values (precomputed).
            q_pos_start: position offset to start the RoPE rotation of Q.
                Default 0 (queries are indexed 0..L_q-1).
            k_pos_start: position offset for K. Default 0 (keys are indexed
                0..L_k-1). Useful if the dictionary is a concatenation of
                multiple per-observation patches and you want to RoPE only
                within each.

        Returns:
            ``(B, L_q, C)`` — attended spectrum, ready for ISTFT.
        """
        B, Lq, C = q_spec.shape
        if C != self.channel_dim:
            raise ValueError(f"q_spec channel={C} != expected {self.channel_dim}")
        if k_spec.shape[-1] != C or v_spec.shape[-1] != C:
            raise ValueError("k_spec / v_spec channels must match q_spec")
        if k_spec.shape[0] != B or v_spec.shape[0] != B:
            raise ValueError("batch sizes must match across q_spec, k_spec, v_spec")

        # (B, L, d_model) → (B, H, L, d_head)
        def split_heads(x: Tensor) -> Tensor:
            return x.reshape(x.shape[0], x.shape[1], self.n_heads, self.d_head).transpose(1, 2)

        Q = split_heads(self.W_q(q_spec))
        K = split_heads(self.W_k(k_spec))
        V = split_heads(self.W_v(v_spec))

        Q = self._rope(Q, pos_start=q_pos_start)
        K = self._rope(K, pos_start=k_pos_start)

        # Standard scaled dot-product attention. (PyTorch ≥ 2 will dispatch
        # to a fused kernel automatically.)
        attn = F.scaled_dot_product_attention(
            Q, K, V, dropout_p=self.dropout if self.training else 0.0
        )  # (B, H, Lq, d_head)

        # Merge heads back: (B, H, Lq, d_head) → (B, Lq, d_model) → (B, Lq, C)
        attn = attn.transpose(1, 2).reshape(B, Lq, self.d_model)
        attn = self.W_o(attn)

        # Residual MLP (zero-init means out ≈ attn at the start of training).
        out = attn + self.mlp(attn)
        return out
