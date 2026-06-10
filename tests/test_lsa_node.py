"""End-to-end tests for LSANODE.

Phase 2.11 checks the forward signature and shapes; Phase 2.12 verifies
the model can actually fit something (overfit on a tiny synthetic batch).
"""

from __future__ import annotations

import math

import pytest
import torch

from lsa_node.models.lsa_node import LSANODE

torch.manual_seed(0)


@pytest.fixture
def tiny_model() -> LSANODE:
    return LSANODE(
        in_dim=1,
        out_dim=1,
        hidden_dim=32,         # N
        n_fft=8,
        hop_length=4,
        d_att=16,
        n_heads=2,
        time_emb_dim=8,
        encoder_hidden=32,
        decoder_hidden=32,
        solver="rk4",          # fixed-step ⇒ faster + more reproducible than dopri5
        use_adjoint=False,
    )


def _toy_batch(B: int = 4, T: int = 8, Q: int = 5, seed: int = 0) -> tuple[torch.Tensor, ...]:
    g = torch.Generator().manual_seed(seed)
    t_obs = torch.linspace(0.0, 1.0, T).unsqueeze(0).expand(B, -1).contiguous()
    x_obs = torch.randn(B, T, 1, generator=g)
    t_query = torch.linspace(0.0, 1.0, Q)
    y_true = torch.randn(Q, B, 1, generator=g)
    return x_obs, t_obs, t_query, y_true


def test_lsa_node_forward_shape(tiny_model: LSANODE) -> None:
    x_obs, t_obs, t_query, _ = _toy_batch()
    out = tiny_model(x_obs, t_obs, t_query)
    assert out.shape == (t_query.shape[0], x_obs.shape[0], 1)
    assert torch.isfinite(out).all()


def test_lsa_node_dictionary_persists_for_adjoint(tiny_model: LSANODE) -> None:
    """After forward, the dictionary stays attached — adjoint backward
    re-evaluates the vector field and needs it. The next forward
    overwrites it; that's the lifecycle contract."""
    x_obs, t_obs, t_query, _ = _toy_batch()
    _ = tiny_model(x_obs, t_obs, t_query)
    # Manual call to the vector field should still work (dictionary
    # remains from the last forward).
    dh = tiny_model.odefunc(
        torch.tensor(0.0),
        torch.zeros(x_obs.shape[0], tiny_model.hidden_dim),
    )
    assert dh.shape == (x_obs.shape[0], tiny_model.hidden_dim)


def test_lsa_node_adjoint_backward_works() -> None:
    """The adjoint backward solve must succeed: re-evaluating ``f_θ``
    requires the dictionary to remain attached. Regression test for the
    bug seen in E001-LSA (job 18296722)."""
    model = LSANODE(
        in_dim=1, out_dim=1, hidden_dim=16, n_fft=8, hop_length=4,
        d_att=16, n_heads=2, solver="dopri5", use_adjoint=True,
        rtol=1e-3, atol=1e-4,
    )
    B, T, Q = 2, 4, 3
    x_obs = torch.randn(B, T, 1)
    t_obs = torch.linspace(0.0, 1.0, T).unsqueeze(0).expand(B, -1).contiguous()
    t_query = torch.linspace(1.0, 2.0, Q)
    y = model(x_obs, t_obs, t_query)
    y.sum().backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grad


def test_lsa_node_gradient_flows(tiny_model: LSANODE) -> None:
    x_obs, t_obs, t_query, _ = _toy_batch(B=2, T=4, Q=3)
    out = tiny_model(x_obs, t_obs, t_query)
    out.sum().backward()
    n_with_grad = sum(1 for p in tiny_model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    assert n_with_grad > 0, "no parameters received gradient"


@pytest.mark.slow
def test_lsa_node_overfits_tiny_batch() -> None:
    """The model must be expressive enough to drive loss on a tiny batch
    significantly down within a small number of steps."""
    torch.manual_seed(7)
    model = LSANODE(
        in_dim=1,
        out_dim=1,
        hidden_dim=32,
        n_fft=8,
        hop_length=4,
        d_att=16,
        n_heads=2,
        time_emb_dim=8,
        encoder_hidden=32,
        decoder_hidden=32,
        solver="rk4",
        use_adjoint=False,
    )

    B, T, Q = 2, 6, 4
    t_obs = torch.linspace(0.0, 1.0, T).unsqueeze(0).expand(B, -1).contiguous()
    x_obs = torch.randn(B, T, 1)
    t_query = torch.linspace(0.0, 1.0, Q)
    y_true = torch.randn(Q, B, 1)

    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    init_loss = None
    for step in range(120):
        opt.zero_grad()
        y_pred = model(x_obs, t_obs, t_query)
        loss = torch.nn.functional.mse_loss(y_pred, y_true)
        if init_loss is None:
            init_loss = loss.item()
        loss.backward()
        opt.step()
    final_loss = loss.item()

    # Sanity check: loss should drop substantially (>50%). The model has
    # the expressivity to drive this further — see Phase 3 for full training
    # diagnostics. The point here is just "gradients are wired correctly."
    assert final_loss < 0.5 * init_loss, (
        f"loss did not decrease enough: init={init_loss:.4f} → final={final_loss:.4f}"
    )
    assert math.isfinite(final_loss)
