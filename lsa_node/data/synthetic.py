"""Synthetic time-series generators.

Generators are organized along the 2 × 2 grid used in
``progress.md`` §5.1:

                  | regular sampling     | irregular sampling
    --------------+----------------------+--------------------
    periodic      | periodic_3d, LV, GO  | + poisson_subsample
    aperiodic     | forced_vibration,    | lorenz63, lorenz96
                  | unstable_oscillator  |

Each base generator returns a dense, regularly-sampled trajectory as a
dict with keys ``"t"`` (length-N) and ``"x"`` (N, dim). Pass the result
through :func:`poisson_subsample` to obtain the irregular variant.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
import torch
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _integrate(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    y0: np.ndarray,
    t: np.ndarray,
    method: str = "RK45",
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> np.ndarray:
    """Wrap ``scipy.solve_ivp`` and return ``(N, dim)``."""
    sol = solve_ivp(
        rhs,
        t_span=(float(t[0]), float(t[-1])),
        y0=y0,
        t_eval=t,
        method=method,
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")
    return sol.y.T  # (N, dim)


def _to_tensors(t: np.ndarray, x: np.ndarray) -> dict[str, torch.Tensor]:
    return {
        "t": torch.as_tensor(t, dtype=torch.float32),
        "x": torch.as_tensor(x, dtype=torch.float32),
    }


# ---------------------------------------------------------------------------
# Sampling regularity transform
# ---------------------------------------------------------------------------
def poisson_subsample(
    traj: dict[str, torch.Tensor],
    keep_rate: float = 0.3,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Independently keep each timestamp with probability ``keep_rate``.

    Matches the DIFFODE protocol (Sec. IV-A, Lorenz row, rate 0.3).
    Guarantees at least two timestamps in the output.
    """
    if not (0.0 < keep_rate <= 1.0):
        raise ValueError(f"keep_rate must be in (0, 1], got {keep_rate}")
    t, x = traj["t"], traj["x"]
    N = t.shape[0]
    g = torch.Generator().manual_seed(seed)
    keep = torch.rand(N, generator=g) < keep_rate
    if keep.sum() < 2:
        # Force at least the endpoints to be kept.
        idx = torch.tensor([0, N - 1])
        keep[idx] = True
    return {"t": t[keep], "x": x[keep]}


# ---------------------------------------------------------------------------
# Periodic (regular) — FODE Periodic-3D-A / B (Sec. III-B)
# ---------------------------------------------------------------------------
def periodic_3d(
    variant: str = "A",
    amp: float = 0.05,
    n_points: int = 1000,
    t_max: float = 20.0,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """FODE Periodic-3D-A / Periodic-3D-B.

    Periodic-3D-A:
        x(t) = sin(t)   + amp · sin(20t)
        y(t) = cos(t)   + amp · cos(20t)
        z(t) = sin(2t)  + amp · sin(20t)

    Periodic-3D-B:
        x(t) = sin(2t)  + amp · sin(20t)
        y(t) = cos(2t)  + amp · cos(20t)
        z(t) = cos(5t)  + amp · sin(20t)
    """
    if variant not in {"A", "B"}:
        raise ValueError(f"variant must be 'A' or 'B', got {variant!r}")
    del seed  # deterministic
    t = np.linspace(0.0, t_max, n_points)
    if variant == "A":
        x = np.stack(
            [
                np.sin(t) + amp * np.sin(20 * t),
                np.cos(t) + amp * np.cos(20 * t),
                np.sin(2 * t) + amp * np.sin(20 * t),
            ],
            axis=-1,
        )
    else:
        x = np.stack(
            [
                np.sin(2 * t) + amp * np.sin(20 * t),
                np.cos(2 * t) + amp * np.cos(20 * t),
                np.cos(5 * t) + amp * np.sin(20 * t),
            ],
            axis=-1,
        )
    return _to_tensors(t, x)


# ---------------------------------------------------------------------------
# Periodic (ODE) systems
# ---------------------------------------------------------------------------
def lotka_volterra(
    alpha: float = 0.1,
    beta: float = 0.02,
    gamma: float = 0.3,
    delta: float = 0.01,
    x0: float = 40.0,
    y0: float = 2.0,
    t_max: float = 100.0,
    n_points: int = 500,
) -> dict[str, torch.Tensor]:
    """Lotka–Volterra predator-prey (FODE Sec. III-C, sustained cycles)."""
    def rhs(_t: float, s: np.ndarray) -> np.ndarray:
        x, y = s
        return np.array([alpha * x - beta * x * y,
                         delta * x * y - gamma * y])
    t = np.linspace(0.0, t_max, n_points)
    s = _integrate(rhs, np.array([x0, y0]), t)
    return _to_tensors(t, s)


def glycolytic_oscillator(
    a: float = 0.75,
    b: float = 0.1,
    x0: tuple[float, float] = (1.0, 1.0),
    t_max: float = 100.0,
    n_points: int = 1000,
) -> dict[str, torch.Tensor]:
    """Glycolytic oscillator (FODE Sec. III-C).

    .. math::
        \\dot x_1 = a - b x_1 - x_1 x_2^2
        \\dot x_2 = b x_1 - x_2 + x_1 x_2^2
    """
    def rhs(_t: float, s: np.ndarray) -> np.ndarray:
        x1, x2 = s
        return np.array([a - b * x1 - x1 * x2 ** 2,
                         b * x1 - x2 + x1 * x2 ** 2])
    t = np.linspace(0.0, t_max, n_points)
    s = _integrate(rhs, np.array(x0), t)
    return _to_tensors(t, s)


# ---------------------------------------------------------------------------
# Aperiodic (regular) systems
# ---------------------------------------------------------------------------
def forced_vibration(
    zeta: float = -0.1,           # negative damping ⇒ growing amplitude
    omega_n: float = 2 * math.pi,
    F0: float = 0.1,
    Omega: float = 4.0,
    t_max: float = 5.0,
    dt: float = 0.01,
    x0: float = 0.5,
    v0: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Negatively-damped forced oscillator (FODE Sec. III-C).

    .. math::
        \\ddot x = -2 \\zeta \\omega_n \\dot x - \\omega_n^2 x + F_0 \\cos(\\Omega t)
    """
    def rhs(t_: float, s: np.ndarray) -> np.ndarray:
        x, v = s
        return np.array([v,
                         -2 * zeta * omega_n * v - omega_n ** 2 * x + F0 * np.cos(Omega * t_)])
    t = np.arange(0.0, t_max + dt / 2, dt)
    s = _integrate(rhs, np.array([x0, v0]), t)
    return _to_tensors(t, s)


def unstable_oscillator(
    n_points: int = 629,
    sigma_noise: float = 0.01,
    t_max: float = 2 * math.pi,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """FODE unstable-oscillator signal: exp-growing amplitude with phase shift.

    .. math::
        x(t) = 0.1 e^{0.5 t} ( \\cos(\\pi t + 1) + \\sin(\\pi t - 1) ) + \\eta(t)
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, t_max, n_points)
    base = 0.1 * np.exp(0.5 * t) * (np.cos(np.pi * t + 1.0) + np.sin(np.pi * t - 1.0))
    noise = rng.normal(0.0, sigma_noise, size=t.shape)
    x = (base + noise)[:, None]
    return _to_tensors(t, x)


# ---------------------------------------------------------------------------
# Aperiodic (irregular by default) — Lorenz family
# ---------------------------------------------------------------------------
def lorenz63(
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    x0: tuple[float, float, float] = (1.0, 1.0, 1.0),
    t_max: float = 40.0,
    n_points: int = 4000,
) -> dict[str, torch.Tensor]:
    """Classical Lorenz '63 chaotic attractor."""
    def rhs(_t: float, s: np.ndarray) -> np.ndarray:
        x, y, z = s
        return np.array([sigma * (y - x),
                         x * (rho - z) - y,
                         x * y - beta * z])
    t = np.linspace(0.0, t_max, n_points)
    s = _integrate(rhs, np.array(x0), t)
    return _to_tensors(t, s)


def lorenz96(
    F: float = 8.0,
    n_dim: int = 40,
    t_max: float = 20.0,
    n_points: int = 2000,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Lorenz '96 high-dimensional chaotic system.

    .. math::
        \\dot x_i = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F
    """
    def rhs(_t: float, s: np.ndarray) -> np.ndarray:
        return (np.roll(s, -1) - np.roll(s, 2)) * np.roll(s, 1) - s + F
    rng = np.random.default_rng(seed)
    x0 = F * np.ones(n_dim) + 1e-3 * rng.standard_normal(n_dim)
    t = np.linspace(0.0, t_max, n_points)
    s = _integrate(rhs, x0, t)
    return _to_tensors(t, s)
