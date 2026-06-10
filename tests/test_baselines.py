"""Shape / signature parity tests for NODE and NCDE baselines."""

from __future__ import annotations

import torch

from lsa_node.models.baselines import NCDEBaseline, NODEBaseline


def _toy_batch(B=2, T=8, Q=5, in_dim=3):
    x_obs = torch.randn(B, T, in_dim)
    t_obs = torch.linspace(0.0, 1.0, T).unsqueeze(0).expand(B, -1).contiguous()
    t_query = torch.linspace(1.0, 2.0, Q)
    return x_obs, t_obs, t_query


def test_node_baseline_shape_and_grad() -> None:
    model = NODEBaseline(in_dim=3, out_dim=3, hidden_dim=16, mlp_dim=32,
                         solver="rk4", use_adjoint=False)
    x_obs, t_obs, t_query = _toy_batch()
    y = model(x_obs, t_obs, t_query)
    assert y.shape == (5, 2, 3)
    y.sum().backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grad


def test_ncde_baseline_shape_and_grad() -> None:
    model = NCDEBaseline(in_dim=3, out_dim=3, hidden_dim=16, mlp_dim=32, solver="rk4")
    x_obs, t_obs, t_query = _toy_batch()
    y = model(x_obs, t_obs, t_query)
    assert y.shape == (5, 2, 3)
    y.sum().backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grad


def test_ncde_baseline_left_padded_irregular_grid() -> None:
    """Regression test for the irregular-sampling failure in E002 NCDE
    (job 18296828). When ``collate_ts_items`` left-pads with a constant
    time prefix, ``torchcde.natural_cubic_coeffs`` complains the time
    grid isn't monotonic. NCDEBaseline must strip the pad."""
    B, T, in_dim, Q = 3, 8, 3, 4
    # Build a left-padded t_obs: first 3 entries equal, rest strictly increasing.
    t_real = torch.linspace(0.3, 1.0, T - 3)
    t_obs = torch.zeros(B, T)
    for i in range(B):
        n_pad = 3 + i  # different pad widths per item
        if n_pad >= T:
            n_pad = T - 2
        first_real_t = t_real[0]
        t_obs[i, :n_pad] = first_real_t
        t_obs[i, n_pad:] = torch.linspace(first_real_t.item(), 1.0, T - n_pad)
    x_obs = torch.randn(B, T, in_dim)
    t_query = torch.linspace(1.0, 2.0, Q)

    model = NCDEBaseline(in_dim=in_dim, out_dim=in_dim, hidden_dim=8, mlp_dim=16, solver="rk4")
    y = model(x_obs, t_obs, t_query)
    assert y.shape == (Q, B, in_dim)
    assert torch.isfinite(y).all()
    y.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
