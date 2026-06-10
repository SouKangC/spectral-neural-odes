"""Shape + gradient sanity tests for the strong baselines."""

from __future__ import annotations

import torch

from lsa_node.models.diffode import DIFFODE
from lsa_node.models.fode import FODE
from lsa_node.models.ode_rnn import ODE_RNN


def _toy(B=2, T=8, Q=5, in_dim=3):
    x = torch.randn(B, T, in_dim)
    t_obs = torch.linspace(0.0, 1.0, T).unsqueeze(0).expand(B, -1).contiguous()
    t_q = torch.linspace(1.0, 2.0, Q)
    return x, t_obs, t_q


def _grad_ok(model):
    return any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_ode_rnn_shape_and_grad() -> None:
    model = ODE_RNN(in_dim=3, out_dim=3, hidden_dim=16, mlp_dim=32,
                    solver="rk4", use_adjoint=False)
    x, t_obs, t_q = _toy()
    y = model(x, t_obs, t_q)
    assert y.shape == (5, 2, 3)
    y.sum().backward()
    assert _grad_ok(model)


def test_fode_shape_and_grad() -> None:
    model = FODE(in_dim=3, out_dim=3, hidden_dim=16, mlp_dim=16, depth=3,
                 solver="rk4", use_adjoint=False)
    x, t_obs, t_q = _toy()
    y = model(x, t_obs, t_q)
    assert y.shape == (5, 2, 3)
    y.sum().backward()
    assert _grad_ok(model)


def test_diffode_shape_and_grad() -> None:
    model = DIFFODE(in_dim=3, out_dim=3, hidden_dim=16, mlp_dim=32,
                    time_emb_dim=8, encoder_hidden=16,
                    solver="rk4", use_adjoint=False)
    x, t_obs, t_q = _toy()
    y = model(x, t_obs, t_q)
    assert y.shape == (5, 2, 3)
    y.sum().backward()
    assert _grad_ok(model)
