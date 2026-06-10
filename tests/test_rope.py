"""Property tests for RoPE — see ``docs/04_rope.md``."""

from __future__ import annotations

import torch

from lsa_node.models.rope import apply_rope, rope_cache

torch.manual_seed(0)


def test_rope_relative_dot_product() -> None:
    """The dot product of RoPE-rotated tokens of the **same** underlying
    vector depends only on the relative position offset."""
    seq_len, d = 128, 32
    cos, sin = rope_cache(seq_len, d)

    # Use one vector replicated across all positions: G[i, j] = v^T R_{j-i} v
    # then only depends on (j - i), so G is Toeplitz (constant on each diagonal).
    v = torch.randn(d)
    rep = v.expand(seq_len, d).contiguous()
    rep_rot = apply_rope(rep, cos, sin)
    G = rep_rot @ rep_rot.T

    diff = G[1:, 1:] - G[:-1, :-1]
    # float32 accumulation noise across a 128×128 gram can reach ~1e-4.
    # The property (Toeplitz structure) holds exactly in real arithmetic.
    assert diff.abs().max() < 5e-4, diff.abs().max()


def test_rope_relative_inner_product_explicit() -> None:
    """Stronger version: <RoPE(q, m), RoPE(k, n)> equals <RoPE(q, 0), RoPE(k, n-m)>
    for any q, k and any pair of positions m, n."""
    d = 32
    M, N = 50, 100  # m and n positions to test
    cos, sin = rope_cache(N + M + 1, d)

    q = torch.randn(d)
    k = torch.randn(d)

    for m, n in [(5, 13), (0, 20), (42, 77), (10, 10), (3, 4)]:
        rel = n - m
        # rotate q at m, k at n
        qm = apply_rope(q[None].expand(m + 1, d).contiguous(), cos[: m + 1], sin[: m + 1])[m]
        kn = apply_rope(k[None].expand(n + 1, d).contiguous(), cos[: n + 1], sin[: n + 1])[n]
        lhs = (qm * kn).sum()

        # rotate q at 0 (identity), k at rel
        q0 = q
        krel_abs = rel if rel >= 0 else rel + cos.shape[0]  # rotations wrap unitarily
        kr = apply_rope(
            k[None].expand(abs(krel_abs) + 1, d).contiguous(),
            cos[: abs(krel_abs) + 1],
            sin[: abs(krel_abs) + 1],
        )[abs(krel_abs)]
        if rel < 0:
            # for negative rel, rotate q by +|rel| instead (symmetric)
            qr = apply_rope(
                q[None].expand(abs(rel) + 1, d).contiguous(),
                cos[: abs(rel) + 1],
                sin[: abs(rel) + 1],
            )[abs(rel)]
            rhs = (qr * k).sum()
        else:
            rhs = (q0 * kr).sum()

        assert torch.allclose(lhs, rhs, atol=1e-5), (m, n, lhs.item(), rhs.item())


def test_rope_norm_preserved() -> None:
    seq_len, d = 64, 16
    cos, sin = rope_cache(seq_len, d)
    q = torch.randn(seq_len, d)
    qt = apply_rope(q, cos, sin)
    assert torch.allclose(q.norm(dim=-1), qt.norm(dim=-1), atol=1e-5)


def test_rope_zero_position_is_identity() -> None:
    """Position 0 has angle 0 ⇒ rotation is the identity."""
    seq_len, d = 8, 16
    cos, sin = rope_cache(seq_len, d)
    q = torch.randn(seq_len, d)
    qt = apply_rope(q, cos, sin)
    assert torch.allclose(qt[0], q[0], atol=1e-6)


def test_rope_cache_odd_head_dim_raises() -> None:
    import pytest

    with pytest.raises(ValueError, match="even"):
        rope_cache(8, 7)


def test_rope_supports_batched_inputs() -> None:
    """apply_rope must broadcast over arbitrary leading dims."""
    B, H, L, d = 4, 3, 16, 32
    cos, sin = rope_cache(L, d)
    x = torch.randn(B, H, L, d)
    y = apply_rope(x, cos, sin)
    assert y.shape == x.shape
    # Check one element matches the unbatched path.
    y0 = apply_rope(x[0, 0], cos, sin)
    assert torch.allclose(y[0, 0], y0, atol=1e-6)
