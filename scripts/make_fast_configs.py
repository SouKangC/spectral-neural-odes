"""Generate a slim 'fast iteration' config set for the course project.

Six representative cells (one per 2×2 quadrant + 2 sanity-check):
    periodic_3d_a__regular     — sanity-check vs E003
    periodic_3d_a__irregular   — periodic × irregular witness
    lotka_volterra__regular    — periodic × regular witness (longer period)
    unstable_oscillator__regular — aperiodic × regular witness
    lorenz63__irregular        — aperiodic × irregular (the hard corner)
    glycolytic__regular        — sanity-check (smooth, NODE saturates here)

Knobs vs make_configs.py defaults:
    epochs:  1000  → 300
    rtol:    1e-5  → 1e-3
    atol:    1e-7  → 1e-4
    adjoint: True  → False  (direct backprop is faster on small windows)

Writes ``code/configs/fast_<cell>.yaml``. Submit with the fast sbatch
template that has ``--time=00:30:00``.
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1] / "configs"

CELLS = [
    ("periodic_3d_a__regular",     "periodic_3d_a", "regular",   3, {"amp": 0.05, "n_points": 1000, "t_max": 20.0}),
    ("periodic_3d_a__irregular",   "periodic_3d_a", "irregular", 3, {"amp": 0.05, "n_points": 1000, "t_max": 20.0}),
    ("lotka_volterra__regular",    "lotka_volterra","regular",   2, {"n_points": 1000, "t_max": 100.0}),
    ("unstable_oscillator__regular","unstable_oscillator","regular",1, {"n_points": 629}),
    ("lorenz63__irregular",        "lorenz63",      "irregular", 3, {"n_points": 4000, "t_max": 40.0}),
    ("glycolytic__regular",        "glycolytic",    "regular",   2, {"n_points": 1000, "t_max": 100.0}),
]


def make(cell: str, gen: str, sampling: str, in_dim: int, data_extra: dict) -> None:
    cfg = {
        "exp_id": f"fast_{cell}",
        "model": {
            "name": "lsa_node",
            "in_dim": in_dim,
            "out_dim": in_dim,
            "hidden_dim": 64,
            "n_fft": 16,
            "hop_length": 4,
            "d_att": 64,
            "n_heads": 4,
            "time_emb_dim": 16,
            "encoder_hidden": 64,
            "decoder_hidden": 64,
            "solver": "dopri5",
            "rtol": 1.0e-3,
            "atol": 1.0e-4,
            "use_adjoint": False,
            "mlp_dim": 128,
        },
        "train": {
            "epochs": 300,
            "batch_size": 32,
            "lr": 1.0e-3,
            "weight_decay": 0.0,
            "grad_clip": 1.0,
            "log_every": 20,
        },
        "data": {
            "name": gen,
            "sampling": sampling,
            "in_len": 30,
            "out_len": 30,
            "train_frac": 0.8,
            **data_extra,
        },
    }
    if sampling == "irregular":
        cfg["data"]["irregular_keep_rate"] = 0.3
    out = ROOT / f"fast_{cell}.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"wrote {out.relative_to(ROOT.parent.parent)}")


for c in CELLS:
    make(*c)
