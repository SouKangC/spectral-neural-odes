"""Tests for IrregularEncoder."""

from __future__ import annotations

import pytest
import torch

from lsa_node.models.encoder import IrregularEncoder, sinusoidal_time_embedding

torch.manual_seed(0)


def test_sinusoidal_time_embedding_shape_and_finite() -> None:
    t = torch.linspace(0, 100, 50)
    emb = sinusoidal_time_embedding(t, dim=32)
    assert emb.shape == (50, 32)
    assert torch.isfinite(emb).all()
    # Range stays in [-1, 1]: it's sin/cos.
    assert emb.min().item() >= -1.0 - 1e-6 and emb.max().item() <= 1.0 + 1e-6


def test_sinusoidal_time_embedding_odd_dim_raises() -> None:
    with pytest.raises(ValueError):
        sinusoidal_time_embedding(torch.tensor([1.0]), dim=7)


def test_encoder_output_shape() -> None:
    B, T, in_dim, N = 4, 10, 3, 64
    enc = IrregularEncoder(in_dim=in_dim, time_emb_dim=16, d_latent=N)
    x = torch.randn(B, T, in_dim)
    t = torch.rand(B, T) * 10
    z = enc(x, t)
    assert z.shape == (B, T, N)
    assert torch.isfinite(z).all()


def test_encoder_with_aux() -> None:
    B, T, in_dim, aux_dim, N = 2, 5, 1, 4, 32
    enc = IrregularEncoder(in_dim=in_dim, time_emb_dim=8, d_latent=N, aux_dim=aux_dim)
    x = torch.randn(B, T, in_dim)
    t = torch.rand(B, T)
    aux = torch.randn(B, T, aux_dim)
    z = enc(x, t, aux)
    assert z.shape == (B, T, N)


def test_encoder_aux_mismatch_raises() -> None:
    enc = IrregularEncoder(in_dim=1, time_emb_dim=8, d_latent=16, aux_dim=4)
    x = torch.randn(2, 3, 1)
    t = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="aux required"):
        enc(x, t, None)


def test_encoder_gradient_flows() -> None:
    enc = IrregularEncoder(in_dim=1, time_emb_dim=8, d_latent=16)
    x = torch.randn(2, 4, 1, requires_grad=True)
    t = torch.rand(2, 4)
    z = enc(x, t)
    z.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
