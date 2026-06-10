"""Baseline model wrappers — NODE and NCDE.

We restrict to baselines with maintained public code (see
``docs/baselines.md``):

* :class:`NODEBaseline` — vanilla Neural ODE via ``torchdiffeq``.
* :class:`NCDEBaseline` — Neural CDE via ``torchcde``.

Both expose the same ``forward(x_obs, t_obs, t_query)`` signature as
:class:`~lsa_node.models.lsa_node.LSANODE` so the training/eval pipeline
is shared.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# NODE — vanilla Neural ODE
# ---------------------------------------------------------------------------
class _NODEMLPField(nn.Module):
    def __init__(self, hidden_dim: int, mlp_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.SiLU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.SiLU(),
            nn.Linear(mlp_dim, hidden_dim),
        )

    def forward(self, t: Tensor, h: Tensor) -> Tensor:
        del t
        return self.net(h)


class NODEBaseline(nn.Module):
    """Vanilla Neural ODE (Chen et al. 2018).

    Forward:
        z = enc(x_obs[:, -1])               # use the most-recent obs as IC
        h = odeint(field, z, t_query)
        y = dec(h)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        mlp_dim: int = 64,
        solver: str = "dopri5",
        rtol: float = 1e-5,
        atol: float = 1e-7,
        use_adjoint: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = nn.Linear(in_dim, hidden_dim)
        self.field = _NODEMLPField(hidden_dim, mlp_dim)
        self.decoder = nn.Linear(hidden_dim, out_dim)
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.use_adjoint = use_adjoint

    def forward(self, x_obs: Tensor, t_obs: Tensor, t_query: Tensor) -> Tensor:
        del t_obs
        h0 = self.encoder(x_obs[:, -1, :])
        t_query = t_query.to(device=h0.device, dtype=h0.dtype)
        from torchdiffeq import odeint, odeint_adjoint
        solver_fn = odeint_adjoint if self.use_adjoint else odeint
        h = solver_fn(self.field, h0, t_query,
                      method=self.solver, rtol=self.rtol, atol=self.atol)
        return self.decoder(h)                              # (Q, B, out_dim)


# ---------------------------------------------------------------------------
# NCDE — Neural Controlled Differential Equation
# ---------------------------------------------------------------------------
class _NCDEMLPField(nn.Module):
    """``f_θ(z) ∈ R^{hidden × control_dim}`` for ``torchcde``."""

    def __init__(self, hidden_dim: int, control_dim: int, mlp_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.control_dim = control_dim
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.SiLU(),
            nn.Linear(mlp_dim, hidden_dim * control_dim),
            nn.Tanh(),
        )

    def forward(self, t: Tensor, z: Tensor) -> Tensor:
        del t
        return self.net(z).view(*z.shape[:-1], self.hidden_dim, self.control_dim)


class NCDEBaseline(nn.Module):
    """NCDE (Kidger et al. 2020) adapted for *forecasting*.

    Standard NCDE integrates ``dz/dt = f(z) dX/dt`` along a control path
    ``X`` that's defined only inside the observation window. For
    forecasting we need to evaluate the hidden state at query times
    *beyond* the observation window — outside ``X``'s support. We use a
    two-stage rollout:

    1.  **Encode**: build ``X = natural_cubic_spline(t_obs, [t; x_obs])``,
        seed ``z(t_obs[0]) = enc(X(t_obs[0]))``, then ``cdeint(z, f, X)``
        from ``t_obs[0]`` to ``t_obs[-1]``. The final state ``z_T``
        summarizes the (possibly irregularly-sampled) input.
    2.  **Forecast**: from ``z_T`` integrate a *free* Neural ODE
        ``dz/dt = g(z)`` over ``t_query`` and decode each query state.

    This is the standard "encode + extrapolate" hack from the irregular-
    time-series literature; without it, NCDE diverges in any pure
    forecasting setup (see E001 incidents).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        mlp_dim: int = 64,
        solver: str = "rk4",
        rtol: float = 1e-3,
        atol: float = 1e-4,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.control_dim = in_dim + 1                          # data + time
        self.encoder = nn.Linear(self.control_dim, hidden_dim)
        self.field = _NCDEMLPField(hidden_dim, self.control_dim, mlp_dim)
        # Free NODE field used after the CDE-encoded summary.
        self.free_field = _NODEMLPField(hidden_dim, mlp_dim)
        self.decoder = nn.Linear(hidden_dim, out_dim)
        self.solver = solver
        self.rtol = rtol
        self.atol = atol

    @staticmethod
    def _strip_left_pad(t_b: Tensor, x_b: Tensor) -> tuple[Tensor, Tensor]:
        """Remove a constant-time left-pad region from one batch item.

        ``collate_ts_items`` left-pads variable-length irregular inputs
        with the first observation's value/time. ``torchcde``'s spline
        builder requires strictly monotonic ``t``, so we drop the
        constant prefix here.
        """
        if t_b.numel() < 2:
            return t_b, x_b
        # Find first index where t starts increasing. diff[i] = t[i+1]-t[i].
        diff = t_b.diff()
        increases = (diff > 0).nonzero(as_tuple=False)
        if increases.numel() == 0:
            return t_b[:1], x_b[:1]
        first_real = int(increases[0].item())  # t[first_real:] is strictly increasing
        return t_b[first_real:], x_b[first_real:]

    def forward(self, x_obs: Tensor, t_obs: Tensor, t_query: Tensor) -> Tensor:
        import torchcde
        from torchdiffeq import odeint

        B = x_obs.shape[0]

        # Per-window sliding has identical *spacing* across batch items
        # but different *offsets*. NCDE only cares about Δt — zero-base
        # each item so the grid is shared, then propagate the offset
        # into ``t_query`` so the free NODE rollout still asks for the
        # right absolute times.
        t_off = t_obs[:, :1]                                # (B, 1)
        t_obs_rel = t_obs - t_off                            # (B, T)
        t0 = t_obs_rel[0]
        all_same = bool((t_obs_rel == t0).all().item())
        monotone = bool((t0.diff() > 0).all().item()) if t0.numel() > 1 else True

        if all_same and monotone:
            t_grid = t0
            t_b = t_obs_rel.unsqueeze(-1)                    # (B, T, 1)
            x_with_t = torch.cat([t_b, x_obs], dim=-1)
            coeffs = torchcde.natural_cubic_coeffs(x_with_t, t_grid)
            path = torchcde.CubicSpline(coeffs, t=t_grid)
            z0 = self.encoder(path.evaluate(t_grid[0]))                # (B, hidden)
            z_end = torchcde.cdeint(
                X=path, func=self.field, z0=z0,
                t=t_grid[[0, -1]],
                method=self.solver, rtol=self.rtol, atol=self.atol,
            )[:, -1]                                                   # (B, hidden)
            t_end = t_grid[-1]
        else:
            # Per-item path construction. Slower but correct for the
            # irregular-padding case.
            z_ends, t_ends = [], []
            for i in range(B):
                ti, xi = self._strip_left_pad(t_obs[i], x_obs[i])
                if ti.numel() < 2:
                    # degenerate — pretend the path is constant at its single point
                    z_ends.append(self.encoder(
                        torch.cat([ti.unsqueeze(-1), xi[0]], dim=-1)
                    ))
                    t_ends.append(ti[-1])
                    continue
                x_with_t = torch.cat([ti.unsqueeze(-1), xi], dim=-1)    # (Ti, in+1)
                coeffs = torchcde.natural_cubic_coeffs(
                    x_with_t.unsqueeze(0), ti                            # add batch dim
                )
                path = torchcde.CubicSpline(coeffs, t=ti)
                z0 = self.encoder(path.evaluate(ti[0]))                  # (1, hidden)
                z_end_i = torchcde.cdeint(
                    X=path, func=self.field, z0=z0,
                    t=ti[[0, -1]],
                    method=self.solver, rtol=self.rtol, atol=self.atol,
                )[:, -1]                                                 # (1, hidden)
                z_ends.append(z_end_i.squeeze(0))
                t_ends.append(ti[-1])
            z_end = torch.stack(z_ends, dim=0)                           # (B, hidden)
            # All items must terminate before any query time; we use the
            # *latest* end as the common seed time for the free NODE.
            t_end = torch.stack(t_ends).max()

        # ----- Stage 2: free NODE over the query times.
        # We work in relative time (the NCDE encode used relative t;
        # the free NODE is time-invariant). Shift t_query by the same
        # offset used for the path — item-0's offset, since t_query
        # itself is shared across the batch.
        t_query = t_query.to(device=z_end.device, dtype=z_end.dtype)
        t_off0 = t_off[0, 0].to(t_query)
        t_query_rel = t_query - t_off0
        t_end = t_end.to(t_query) - t_off0 if t_end.numel() == 1 else (t_end.to(t_query) - t_off0)
        # Integrate from ``t_end`` forward. If the first query time is
        # already > t_end, prepend t_end as a seed; otherwise (e.g. an
        # *interpolation* protocol where ``t_query[0] == t_end``) skip
        # the prepend so odeint sees a strictly-increasing grid.
        eps = 1e-6
        if t_query_rel[0].item() > t_end.item() + eps:
            t_full = torch.cat([t_end[None], t_query_rel])
            drop = 1
        else:
            t_full = t_query_rel
            drop = 0
        z = odeint(
            self.free_field, z_end, t_full,
            method=self.solver, rtol=self.rtol, atol=self.atol,
        )                                                            # (≥Q, B, hidden)
        z = z[drop:]                                                 # (Q, B, hidden)
        return self.decoder(z)
