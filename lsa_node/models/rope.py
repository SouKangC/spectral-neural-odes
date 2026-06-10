"""Rotary positional embedding (RoPE).

See ``docs/04_rope.md`` for the derivation. We rotate even/odd-paired
feature dimensions of Q and K (never V) by an angle proportional to the
token's position. The dot product then depends only on the *relative*
position of two tokens.
"""

from __future__ import annotations

import torch
from torch import Tensor


def rope_cache(
    seq_len: int,
    head_dim: int,
    base: float = 10000.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:
    """Precompute (cos, sin) tables of shape (seq_len, head_dim/2).

    Args:
        seq_len: maximum position index needed.
        head_dim: per-head embedding dimension; must be even.
        base: the RoPE base, typically 10_000.
    """
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")
    half = head_dim // 2
    inv_freq = base ** (-torch.arange(0, half, device=device, dtype=dtype) * 2.0 / head_dim)
    pos = torch.arange(seq_len, device=device, dtype=dtype)
    angles = pos[:, None] * inv_freq[None, :]
    return angles.cos(), angles.sin()


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply RoPE rotation to ``x``.

    Pairs the **even/odd** feature dimensions and rotates each pair by the
    angle implied by ``cos`` / ``sin`` for that token's position. See
    ``docs/04_rope.md`` §5.2 for the derivation and ``docs/04_rope.md``
    §4 / Q&A for why we only rotate Q and K (never V).

    Args:
        x:   shape (..., seq, head_dim). ``head_dim`` must be even.
        cos: shape (seq, head_dim/2) — broadcasts over the leading dims of x.
        sin: same shape as ``cos``.

    Returns:
        Tensor of the same shape and dtype as ``x``.
    """
    if x.shape[-1] != 2 * cos.shape[-1]:
        raise ValueError(
            f"x.head_dim={x.shape[-1]} must equal 2*cos.shape[-1]={2 * cos.shape[-1]}"
        )
    if x.shape[-2] != cos.shape[-2]:
        raise ValueError(
            f"x.seq={x.shape[-2]} must equal cos.seq={cos.shape[-2]}"
        )

    # Split into even/odd halves: x = [x_even, x_odd] interleaved along last dim.
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    # Cast cos/sin to x's dtype so the result stays in x's dtype.
    cos_b = cos.to(x.dtype)
    sin_b = sin.to(x.dtype)

    # 2-D rotation per pair: (e, o) ↦ (e cosθ − o sinθ, e sinθ + o cosθ).
    rot_even = x_even * cos_b - x_odd * sin_b
    rot_odd = x_even * sin_b + x_odd * cos_b

    # Re-interleave back into the original layout.
    out = torch.empty_like(x)
    out[..., 0::2] = rot_even
    out[..., 1::2] = rot_odd
    return out
