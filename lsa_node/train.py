"""Training entry point.

Usage::

    python -m lsa_node.train --config configs/smoke.yaml
    python -m lsa_node.train --config configs/smoke.yaml -o train.lr=5e-4

Results land under ``out_dir = args.out / <exp_id> / <model> / seed<S>``
where ``exp_id`` comes from ``cfg['exp_id']`` (or defaults to
``<data.name>__<data.sampling>``). Each run writes:

* ``hparams.json``           — the resolved config.
* ``metrics.jsonl``          — per-step / per-epoch scalars.
* ``final_metrics.json``     — test-set summary (one line, easy to grep).
* ``checkpoints/ckpt.pt``    — best (lowest val loss) weights.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lsa_node.data import SlidingWindowDataset, collate_ts_items
from lsa_node.data import synthetic as synth_gen
from lsa_node.models.baselines import NCDEBaseline, NODEBaseline
from lsa_node.models.diffode import DIFFODE
from lsa_node.models.fode import FODE
from lsa_node.models.lsa_node import LSANODE
from lsa_node.models.ode_rnn import ODE_RNN
from lsa_node.models.ode_rnn_fode import ODE_RNN_FODE
from lsa_node.models.ode_rnn_gated import ODE_RNN_Gated
from lsa_node.models.ode_rnn_hybrid import ODE_RNN_Hybrid
from lsa_node.models.ode_rnn_hybrid_notime import ODE_RNN_Hybrid_NoTime
from lsa_node.models.stft_fode import STFTFODE
from lsa_node.utils.config import add_common_args, load_yaml, merge_overrides
from lsa_node.utils.logging import RunLogger
from lsa_node.utils.seed import seed_everything


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LSA-NODE trainer")
    add_common_args(p)
    p.add_argument("--model", default=None,
                   help="override cfg.model.name from the command line")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------
_GENERATOR_REGISTRY = {
    "periodic_3d_a": dict(fn=synth_gen.periodic_3d, kwargs={"variant": "A"}),
    "periodic_3d_b": dict(fn=synth_gen.periodic_3d, kwargs={"variant": "B"}),
    "lotka_volterra": dict(fn=synth_gen.lotka_volterra, kwargs={}),
    "glycolytic": dict(fn=synth_gen.glycolytic_oscillator, kwargs={}),
    "forced_vibration": dict(fn=synth_gen.forced_vibration, kwargs={}),
    "unstable_oscillator": dict(fn=synth_gen.unstable_oscillator, kwargs={}),
    "lorenz63": dict(fn=synth_gen.lorenz63, kwargs={}),
    "lorenz96": dict(fn=synth_gen.lorenz96, kwargs={}),
}


def build_datasets(cfg_data: dict[str, Any]) -> tuple[SlidingWindowDataset, SlidingWindowDataset]:
    """Return ``(train_ds, test_ds)`` split chronologically (80/20).

    ``cfg_data`` keys:

    * ``name``                — generator name (see ``_GENERATOR_REGISTRY``).
    * ``sampling``            — ``"regular"`` or ``"irregular"`` (default regular).
    * ``in_len``, ``out_len`` — sliding-window sizes.
    * ``train_frac``          — split point (default 0.8).
    * any extra keys are passed to the generator.
    """
    name = cfg_data["name"]
    if name not in _GENERATOR_REGISTRY:
        raise ValueError(f"unknown dataset {name!r}; have {list(_GENERATOR_REGISTRY)}")
    spec = _GENERATOR_REGISTRY[name]

    gen_kwargs = {**spec["kwargs"]}
    # Pass through any extra keys ("amp", "n_points", "t_max", ...).
    for k, v in cfg_data.items():
        if k in {"name", "sampling", "in_len", "out_len", "train_frac",
                 "irregular_keep_rate", "seed_offset"}:
            continue
        gen_kwargs[k] = v
    traj = spec["fn"](**gen_kwargs)

    train_frac = float(cfg_data.get("train_frac", 0.8))
    N = traj["t"].shape[0]
    split = int(round(train_frac * N))
    train_traj = {"t": traj["t"][:split], "x": traj["x"][:split]}
    test_traj = {"t": traj["t"][split:], "x": traj["x"][split:]}

    sampling = cfg_data.get("sampling", "regular").lower()
    if sampling not in {"regular", "irregular"}:
        raise ValueError(f"sampling must be 'regular' or 'irregular', got {sampling!r}")
    is_irregular = sampling == "irregular"

    common = dict(
        in_len=int(cfg_data["in_len"]),
        out_len=int(cfg_data["out_len"]),
        irregular=is_irregular,
        irregular_keep_rate=float(cfg_data.get("irregular_keep_rate", 0.3)),
    )
    train_ds = SlidingWindowDataset(
        train_traj, **common, irregular_seed_offset=int(cfg_data.get("seed_offset", 0)),
    )
    test_ds = SlidingWindowDataset(
        test_traj, **common, irregular_seed_offset=int(cfg_data.get("seed_offset", 10000)),
    )
    return train_ds, test_ds


def build_model(cfg_model: dict[str, Any], in_dim: int, out_dim: int) -> torch.nn.Module:
    name = cfg_model["name"]
    if name == "lsa_node":
        return LSANODE(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=int(cfg_model["hidden_dim"]),
            n_fft=int(cfg_model["n_fft"]),
            hop_length=int(cfg_model["hop_length"]),
            d_att=int(cfg_model["d_att"]),
            n_heads=int(cfg_model.get("n_heads", 4)),
            time_emb_dim=int(cfg_model.get("time_emb_dim", 16)),
            encoder_hidden=int(cfg_model.get("encoder_hidden", 64)),
            decoder_hidden=int(cfg_model.get("decoder_hidden", 64)),
            solver=cfg_model.get("solver", "dopri5"),
            rtol=float(cfg_model.get("rtol", 1e-5)),
            atol=float(cfg_model.get("atol", 1e-7)),
            use_adjoint=bool(cfg_model.get("use_adjoint", True)),
        )
    if name == "node":
        return NODEBaseline(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=int(cfg_model["hidden_dim"]),
            mlp_dim=int(cfg_model.get("mlp_dim", 64)),
            solver=cfg_model.get("solver", "dopri5"),
            rtol=float(cfg_model.get("rtol", 1e-5)),
            atol=float(cfg_model.get("atol", 1e-7)),
            use_adjoint=bool(cfg_model.get("use_adjoint", True)),
        )
    if name == "ncde":
        return NCDEBaseline(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=int(cfg_model["hidden_dim"]),
            mlp_dim=int(cfg_model.get("mlp_dim", 64)),
            solver=cfg_model.get("solver", "rk4"),
            rtol=float(cfg_model.get("rtol", 1e-3)),
            atol=float(cfg_model.get("atol", 1e-4)),
        )
    if name == "stft_fode":
        return STFTFODE(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=int(cfg_model["hidden_dim"]),
            n_fft=int(cfg_model["n_fft"]),
            hop_length=int(cfg_model["hop_length"]),
            mlp_dim=int(cfg_model.get("mlp_dim", 128)),
            depth=int(cfg_model.get("depth", 3)),
            solver=cfg_model.get("solver", "dopri5"),
            rtol=float(cfg_model.get("rtol", 1e-3)),
            atol=float(cfg_model.get("atol", 1e-4)),
            use_adjoint=bool(cfg_model.get("use_adjoint", False)),
        )
    if name == "ode_rnn":
        return ODE_RNN(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=int(cfg_model["hidden_dim"]),
            mlp_dim=int(cfg_model.get("mlp_dim", 64)),
            solver=cfg_model.get("solver", "dopri5"),
            rtol=float(cfg_model.get("rtol", 1e-3)),
            atol=float(cfg_model.get("atol", 1e-4)),
            use_adjoint=bool(cfg_model.get("use_adjoint", False)),
        )
    if name == "fode":
        return FODE(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=int(cfg_model["hidden_dim"]),
            mlp_dim=int(cfg_model.get("fode_mlp_dim", 16)),
            depth=int(cfg_model.get("fode_depth", 3)),
            solver=cfg_model.get("solver", "dopri5"),
            rtol=float(cfg_model.get("rtol", 1e-3)),
            atol=float(cfg_model.get("atol", 1e-4)),
            use_adjoint=bool(cfg_model.get("use_adjoint", False)),
        )
    if name == "ode_rnn_gated":
        return ODE_RNN_Gated(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=int(cfg_model["hidden_dim"]),
            time_mlp_dim=int(cfg_model.get("time_mlp_dim", 64)),
            fode_mlp_dim=int(cfg_model.get("fode_mlp_dim", 16)),
            fode_depth=int(cfg_model.get("fode_depth", 3)),
            time_emb_dim=int(cfg_model.get("time_emb_dim", 16)),
            gate_init=float(cfg_model.get("gate_init", -6.0)),
            solver=cfg_model.get("solver", "dopri5"),
            rtol=float(cfg_model.get("rtol", 1e-3)),
            atol=float(cfg_model.get("atol", 1e-4)),
            use_adjoint=bool(cfg_model.get("use_adjoint", False)),
        )
    if name == "ode_rnn_hybrid_notime":
        return ODE_RNN_Hybrid_NoTime(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=int(cfg_model["hidden_dim"]),
            time_mlp_dim=int(cfg_model.get("time_mlp_dim", 64)),
            fode_mlp_dim=int(cfg_model.get("fode_mlp_dim", 16)),
            fode_depth=int(cfg_model.get("fode_depth", 3)),
            solver=cfg_model.get("solver", "dopri5"),
            rtol=float(cfg_model.get("rtol", 1e-3)),
            atol=float(cfg_model.get("atol", 1e-4)),
            use_adjoint=bool(cfg_model.get("use_adjoint", False)),
        )
    if name == "ode_rnn_hybrid":
        return ODE_RNN_Hybrid(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=int(cfg_model["hidden_dim"]),
            time_mlp_dim=int(cfg_model.get("time_mlp_dim", 64)),
            fode_mlp_dim=int(cfg_model.get("fode_mlp_dim", 16)),
            fode_depth=int(cfg_model.get("fode_depth", 3)),
            time_emb_dim=int(cfg_model.get("time_emb_dim", 16)),
            solver=cfg_model.get("solver", "dopri5"),
            rtol=float(cfg_model.get("rtol", 1e-3)),
            atol=float(cfg_model.get("atol", 1e-4)),
            use_adjoint=bool(cfg_model.get("use_adjoint", False)),
        )
    if name == "ode_rnn_fode":
        return ODE_RNN_FODE(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=int(cfg_model["hidden_dim"]),
            fode_mlp_dim=int(cfg_model.get("fode_mlp_dim", 16)),
            fode_depth=int(cfg_model.get("fode_depth", 3)),
            solver=cfg_model.get("solver", "dopri5"),
            rtol=float(cfg_model.get("rtol", 1e-3)),
            atol=float(cfg_model.get("atol", 1e-4)),
            use_adjoint=bool(cfg_model.get("use_adjoint", False)),
        )
    if name == "diffode":
        return DIFFODE(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=int(cfg_model["hidden_dim"]),
            mlp_dim=int(cfg_model.get("mlp_dim", 64)),
            time_emb_dim=int(cfg_model.get("time_emb_dim", 16)),
            encoder_hidden=int(cfg_model.get("encoder_hidden", 64)),
            solver=cfg_model.get("solver", "dopri5"),
            rtol=float(cfg_model.get("rtol", 1e-3)),
            atol=float(cfg_model.get("atol", 1e-4)),
            use_adjoint=bool(cfg_model.get("use_adjoint", False)),
        )
    if name == "faithful_node":
        from lsa_node.models.faithful_node import FaithfulNODE
        return FaithfulNODE(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=int(cfg_model["hidden_dim"]),
            mlp_dim=int(cfg_model.get("mlp_dim", 64)),
            solver=cfg_model.get("solver", "dopri5"),
            rtol=float(cfg_model.get("rtol", 1e-3)),
            atol=float(cfg_model.get("atol", 1e-4)),
            use_adjoint=bool(cfg_model.get("use_adjoint", False)),
        )
    if name == "faithful_fode":
        from lsa_node.models.faithful_fode import FaithfulFODE
        return FaithfulFODE(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=int(cfg_model.get("hidden_dim", 16)),
            mlp_dim=int(cfg_model.get("mlp_dim", 16)),
            depth=int(cfg_model.get("depth", 3)),
            solver=cfg_model.get("solver", "dopri5"),
            rtol=float(cfg_model.get("rtol", 1e-3)),
            atol=float(cfg_model.get("atol", 1e-4)),
            use_adjoint=bool(cfg_model.get("use_adjoint", False)),
        )
    if name == "faithful_ode_rnn":
        from lsa_node.models.faithful_ode_rnn import FaithfulODE_RNN
        return FaithfulODE_RNN(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=int(cfg_model["hidden_dim"]),
            mlp_dim=int(cfg_model.get("mlp_dim", 64)),
            solver=cfg_model.get("solver", "dopri5"),
            rtol=float(cfg_model.get("rtol", 1e-3)),
            atol=float(cfg_model.get("atol", 1e-4)),
            use_adjoint=bool(cfg_model.get("use_adjoint", False)),
        )
    if name == "faithful_diffode":
        from lsa_node.models.faithful_diffode import FaithfulDIFFODE
        return FaithfulDIFFODE(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=int(cfg_model["hidden_dim"]),
            mlp_dim=int(cfg_model.get("mlp_dim", 64)),
            time_emb_dim=int(cfg_model.get("time_emb_dim", 16)),
            encoder_hidden=int(cfg_model.get("encoder_hidden", 64)),
            solver=cfg_model.get("solver", "dopri5"),
            rtol=float(cfg_model.get("rtol", 1e-3)),
            atol=float(cfg_model.get("atol", 1e-4)),
            use_adjoint=bool(cfg_model.get("use_adjoint", False)),
        )
    raise ValueError(
        f"unknown model {name!r} — have lsa_node, node, ncde, stft_fode, ode_rnn, "
        f"ode_rnn_fode, ode_rnn_hybrid, ode_rnn_hybrid_notime, ode_rnn_gated, fode, "
        f"diffode, faithful_diffode"
    )


# ---------------------------------------------------------------------------
# Loop helpers
# ---------------------------------------------------------------------------
def _to_device(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def _forward_loss(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    # batch.x_query: (B, Q, in_dim); model output: (Q, B, out_dim)
    y_pred = model(batch["x_obs"], batch["t_obs"], batch["t_query"])
    y_true = batch["x_query"].transpose(0, 1)                       # (Q, B, in_dim)
    loss = F.mse_loss(y_pred, y_true)
    # DIFFODE-paper Hoyer sparsity term: subtract lambda * Hoyer(p_t) so
    # the model is pushed toward sparse attention. Active only for models
    # that expose `last_hoyer`.
    h = getattr(model, "last_hoyer", None)
    if h is not None:
        loss = loss - 1.0e-2 * h
    return loss


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    total_se, total_ae, total_pct, n = 0.0, 0.0, 0.0, 0
    for batch in loader:
        batch = _to_device(batch, device)
        y_pred = model(batch["x_obs"], batch["t_obs"], batch["t_query"])
        y_true = batch["x_query"].transpose(0, 1)
        diff = (y_pred - y_true)
        total_se += diff.pow(2).sum().item()
        total_ae += diff.abs().sum().item()
        denom = y_true.abs().clamp(min=1e-8)
        total_pct += (diff.abs() / denom).sum().item() * 100.0
        n += diff.numel()
    return {
        "mse": total_se / max(n, 1),
        "mae": total_ae / max(n, 1),
        "mape": total_pct / max(n, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    cfg = merge_overrides(cfg, args.override)
    if args.model is not None:
        cfg.setdefault("model", {})["name"] = args.model

    seed_everything(args.seed, deterministic=False)         # speed > bit-exactness for training

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve experiment id (used as the first level under out/).
    sampling = cfg.get("data", {}).get("sampling", "regular")
    exp_id = cfg.get("exp_id") or f"{cfg['data']['name']}__{sampling}"
    out_dir = (args.out / exp_id / cfg["model"]["name"] / f"seed{args.seed}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stamp the resolved cfg + run metadata.
    hparams = {**cfg, "seed": args.seed, "device": device, "exp_id": exp_id}

    train_ds, test_ds = build_datasets(cfg["data"])
    in_dim = train_ds[0].x_obs.shape[-1]
    out_dim = train_ds[0].x_query.shape[-1]

    batch_size = int(cfg["train"]["batch_size"])
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_ts_items, drop_last=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_ts_items, drop_last=False,
    )

    model = build_model(cfg["model"], in_dim=in_dim, out_dim=out_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    hparams["n_params"] = n_params
    print(f"[train] exp={exp_id} model={cfg['model']['name']} seed={args.seed} "
          f"device={device} n_params={n_params:,} "
          f"|train|={len(train_ds)} |test|={len(test_ds)}")

    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["train"]["lr"]),
                           weight_decay=float(cfg["train"].get("weight_decay", 0.0)))
    grad_clip = float(cfg["train"].get("grad_clip", 0.0))
    epochs = int(cfg["train"]["epochs"])
    log_every = int(cfg["train"].get("log_every", 50))

    best_test_mse = math.inf
    step = 0
    with RunLogger(out_dir, hparams=hparams) as log:
        for epoch in range(epochs):
            model.train()
            running = 0.0
            n_batches = 0
            for batch in train_loader:
                batch = _to_device(batch, device)
                loss = _forward_loss(model, batch)
                opt.zero_grad()
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
                running += loss.item()
                n_batches += 1
                step += 1
                if step % log_every == 0:
                    log.scalar("train/loss_step", loss.item(), step)
            train_loss = running / max(n_batches, 1)
            log.scalar("train/loss_epoch", train_loss, epoch)

            # Eval each epoch (cheap for tiny synthetic datasets).
            metrics = evaluate(model, test_loader, device)
            log.scalars({f"test/{k}": v for k, v in metrics.items()}, step=epoch)

            if metrics["mse"] < best_test_mse:
                best_test_mse = metrics["mse"]
                log.save_checkpoint(model, opt, step=epoch, extra={
                    "epoch": epoch, "test_mse": metrics["mse"],
                })

            if epoch % max(1, epochs // 20) == 0 or epoch == epochs - 1:
                print(f"[train] epoch {epoch:4d}/{epochs}  "
                      f"train_loss={train_loss:.4e}  "
                      f"test_mse={metrics['mse']:.4e}  "
                      f"test_mape={metrics['mape']:.3f}%")

        # Final eval + dump.
        final = evaluate(model, test_loader, device)
        final["best_test_mse"] = best_test_mse
        final["n_params"] = n_params
        final["epochs_run"] = epochs
        (out_dir / "final_metrics.json").write_text(json.dumps(final, indent=2))
        print(f"[train] DONE  final={final}")


if __name__ == "__main__":
    main()
