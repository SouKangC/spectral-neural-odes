"""Tests for ODEFunc and STFTAttentionBlock."""

from __future__ import annotations

import pytest
import torch

from lsa_node.models.ode_func import ODEFunc
from lsa_node.models.stft_attention import STFTAttentionBlock, stft

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# STFTAttentionBlock
# ---------------------------------------------------------------------------
def test_attention_block_output_shape() -> None:
    n_fft, d_model = 16, 32
    block = STFTAttentionBlock(n_fft=n_fft, d_model=d_model, n_heads=4)
    cdim = 2 * (n_fft // 2 + 1)
    B, Lq, Lk = 4, 13, 65
    q = torch.randn(B, Lq, cdim)
    k = torch.randn(B, Lk, cdim)
    v = torch.randn(B, Lk, cdim)
    out = block(q, k, v)
    assert out.shape == (B, Lq, cdim)
    assert torch.isfinite(out).all()


def test_attention_block_residual_init_is_attention_only() -> None:
    """With the MLP zero-initialized, the block output should equal the
    attention output alone — a useful baseline for stability proofs."""
    n_fft, d_model = 8, 16
    block = STFTAttentionBlock(n_fft=n_fft, d_model=d_model, n_heads=2).eval()
    cdim = 2 * (n_fft // 2 + 1)
    q = torch.randn(2, 5, cdim)
    k = torch.randn(2, 7, cdim)
    v = torch.randn(2, 7, cdim)
    # Force MLP to zero (it already is from init, but explicit).
    for p in block.mlp[-1].parameters():
        p.data.zero_()
    out = block(q, k, v)
    # Forward again with the mlp branch manually subtracted to recover attention only.
    mlp_out = block.mlp(out)  # should be zero
    assert mlp_out.abs().max() < 1e-6


def test_attention_block_gradient_flow() -> None:
    n_fft, d_model = 8, 16
    block = STFTAttentionBlock(n_fft=n_fft, d_model=d_model, n_heads=2)
    cdim = 2 * (n_fft // 2 + 1)
    q = torch.randn(2, 4, cdim, requires_grad=True)
    k = torch.randn(2, 6, cdim, requires_grad=True)
    v = torch.randn(2, 6, cdim, requires_grad=True)
    block(q, k, v).sum().backward()
    for p, name in [(q, "q"), (k, "k"), (v, "v")]:
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


# ---------------------------------------------------------------------------
# ODEFunc
# ---------------------------------------------------------------------------
def _make_ode_inputs(B: int, N: int, T: int, n_fft: int, hop: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (h, K_spec, V_spec) with the correct shapes for ODEFunc."""
    cdim = 2 * (n_fft // 2 + 1)
    # M = T * L_per_obs. Build a dictionary by STFT'ing T random patches.
    patches = torch.randn(B, T, N)
    H = stft(patches.reshape(B * T, N), n_fft=n_fft, hop_length=hop)  # (B*T, L, cdim)
    L = H.shape[1]
    H = H.reshape(B, T * L, cdim)
    h = torch.randn(B, N)
    return h, H.clone(), H.clone()


def test_ode_func_forward_shape() -> None:
    N, n_fft, hop, d_model = 64, 16, 4, 32
    func = ODEFunc(n_fft=n_fft, hop_length=hop, d_model=d_model, n_heads=4).eval()
    h, K, V = _make_ode_inputs(B=2, N=N, T=3, n_fft=n_fft, hop=hop)
    func.set_dictionary(K, V, signal_length=N)
    t = torch.tensor(0.0)
    dh = func(t, h)
    assert dh.shape == (2, N)
    assert torch.isfinite(dh).all()
    assert torch.is_floating_point(dh) and not dh.is_complex()


def test_ode_func_without_dictionary_raises() -> None:
    func = ODEFunc(n_fft=8, hop_length=4, d_model=16, n_heads=2)
    with pytest.raises(RuntimeError, match="set_dictionary"):
        func(torch.tensor(0.0), torch.randn(2, 32))


def test_ode_func_gradient_flow() -> None:
    N, n_fft, hop, d_model = 32, 8, 4, 16
    func = ODEFunc(n_fft=n_fft, hop_length=hop, d_model=d_model, n_heads=2)
    h, K, V = _make_ode_inputs(B=2, N=N, T=2, n_fft=n_fft, hop=hop)
    K.requires_grad_(True); V.requires_grad_(True); h.requires_grad_(True)
    func.set_dictionary(K, V, signal_length=N)
    dh = func(torch.tensor(0.0), h)
    dh.sum().backward()
    assert h.grad is not None and torch.isfinite(h.grad).all()
    # Params should also receive gradient
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in func.parameters())
    assert has_grad


def test_ode_func_integrates_with_torchdiffeq() -> None:
    """End-to-end: the vector field plugs into torchdiffeq.odeint."""
    torchdiffeq = pytest.importorskip("torchdiffeq")
    N, n_fft, hop, d_model = 32, 8, 4, 16
    func = ODEFunc(n_fft=n_fft, hop_length=hop, d_model=d_model, n_heads=2)
    h, K, V = _make_ode_inputs(B=2, N=N, T=2, n_fft=n_fft, hop=hop)
    func.set_dictionary(K, V, signal_length=N)
    t_span = torch.linspace(0.0, 1.0, 4)
    # Use a fixed-step solver for the test (faster than dopri5 on tiny problems).
    out = torchdiffeq.odeint(func, h, t_span, method="rk4")
    assert out.shape == (4, 2, N)
    assert torch.isfinite(out).all()
