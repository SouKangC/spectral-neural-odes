"""Shape / finiteness tests for synthetic time-series generators."""

from __future__ import annotations

import torch

from lsa_node.data.synthetic import (
    forced_vibration,
    glycolytic_oscillator,
    lorenz63,
    lorenz96,
    lotka_volterra,
    periodic_3d,
    poisson_subsample,
    unstable_oscillator,
)


def _check(traj: dict[str, torch.Tensor], expected_n: int, expected_dim: int) -> None:
    assert set(traj.keys()) == {"t", "x"}
    assert traj["t"].shape == (expected_n,)
    assert traj["x"].shape == (expected_n, expected_dim)
    assert torch.isfinite(traj["t"]).all()
    assert torch.isfinite(traj["x"]).all()


def test_periodic_3d_a_shape_and_amplitude() -> None:
    traj = periodic_3d(variant="A", amp=0.05, n_points=400, t_max=10.0)
    _check(traj, 400, 3)
    # Amplitude bounded: sin/cos plus a small amp ⇒ |x| <= ~1.05.
    assert traj["x"].abs().max() < 1.5


def test_periodic_3d_b_shape() -> None:
    traj = periodic_3d(variant="B", amp=0.10, n_points=200, t_max=5.0)
    _check(traj, 200, 3)


def test_lotka_volterra_shape_and_positivity() -> None:
    traj = lotka_volterra(n_points=300, t_max=80.0)
    _check(traj, 300, 2)
    # LV populations stay positive given positive ICs.
    assert (traj["x"] > 0).all()


def test_glycolytic_oscillator_shape() -> None:
    traj = glycolytic_oscillator(n_points=500, t_max=80.0)
    _check(traj, 500, 2)


def test_forced_vibration_growth() -> None:
    traj = forced_vibration(t_max=5.0, dt=0.05)
    expected_n = int(round(5.0 / 0.05)) + 1
    _check(traj, expected_n, 2)
    # Negative damping ⇒ amplitude grows over time.
    early = traj["x"][:20, 0].abs().max().item()
    late = traj["x"][-20:, 0].abs().max().item()
    assert late > early, (early, late)


def test_unstable_oscillator_shape_and_growth() -> None:
    traj = unstable_oscillator(n_points=200, sigma_noise=0.0)
    _check(traj, 200, 1)
    early = traj["x"][:20].abs().max().item()
    late = traj["x"][-20:].abs().max().item()
    assert late > early


def test_lorenz63_shape() -> None:
    traj = lorenz63(t_max=5.0, n_points=500)
    _check(traj, 500, 3)


def test_lorenz96_shape() -> None:
    n_dim = 8
    traj = lorenz96(n_dim=n_dim, t_max=2.0, n_points=200)
    _check(traj, 200, n_dim)


def test_poisson_subsample_reduces_count_and_keeps_pairing() -> None:
    traj = periodic_3d(variant="A", n_points=1000, t_max=20.0)
    sub = poisson_subsample(traj, keep_rate=0.3, seed=0)
    assert sub["t"].shape[0] == sub["x"].shape[0]
    # Expectation 300 with rate 0.3 over 1000; loose check.
    assert 200 < sub["t"].shape[0] < 400
    # Subsampled timestamps are a subset of the originals.
    keep_mask = torch.isin(traj["t"], sub["t"])
    assert keep_mask.sum().item() == sub["t"].shape[0]


def test_poisson_subsample_minimum_two() -> None:
    """With keep_rate near 0 the routine still keeps endpoints."""
    traj = {"t": torch.linspace(0.0, 1.0, 50), "x": torch.zeros(50, 1)}
    sub = poisson_subsample(traj, keep_rate=0.01, seed=0)
    assert sub["t"].shape[0] >= 2
