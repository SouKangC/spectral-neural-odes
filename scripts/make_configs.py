"""Generate per-cell YAML configs for the Phase 5 2×2 synthetic grid.

Reads ``code/configs/_base_<periodic|aperiodic>.yaml`` (a template) and
writes one ``<generator>__<sampling>.yaml`` per cell. Run once after
editing the templates; the generated files are committed.
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1] / "configs"


# (generator, periodicity, default sampling regimes)
PERIODIC_REGULAR = [
    ("periodic_3d_a", {}),
    ("periodic_3d_b", {}),
    ("lotka_volterra", {}),
    ("glycolytic", {}),
]
APERIODIC_REGULAR = [
    ("forced_vibration", {}),
    ("unstable_oscillator", {}),
]
APERIODIC_IRREGULAR = [
    ("lorenz63", {}),
    ("lorenz96", {"n_dim": 8}),       # keep small to start
]


def _write(out_path: Path, payload: dict) -> None:
    out_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    print(f"wrote {out_path.relative_to(ROOT.parent.parent)}")


def _base_model() -> dict:
    return {
        "name": "lsa_node",
        "in_dim": 3,
        "out_dim": 3,
        "hidden_dim": 64,
        "n_fft": 16,
        "hop_length": 4,
        "d_att": 64,
        "n_heads": 4,
        "time_emb_dim": 16,
        "encoder_hidden": 64,
        "decoder_hidden": 64,
        "solver": "dopri5",
        "rtol": 1.0e-5,
        "atol": 1.0e-7,
        "use_adjoint": True,
        "mlp_dim": 128,
    }


def _base_train(epochs: int = 1000) -> dict:
    return {
        "epochs": epochs,
        "batch_size": 32,
        "lr": 1.0e-3,
        "weight_decay": 0.0,
        "grad_clip": 1.0,
        "log_every": 50,
    }


def make(gen_name: str, sampling: str, in_dim: int, gen_extra: dict) -> None:
    cfg = {
        "exp_id": f"{gen_name}__{sampling}",
        "model": {**_base_model(), "in_dim": in_dim, "out_dim": in_dim},
        "train": _base_train(),
        "data": {
            "name": gen_name,
            "sampling": sampling,
            "in_len": 30,
            "out_len": 30,
            "train_frac": 0.8,
            **gen_extra,
        },
    }
    if sampling == "irregular":
        cfg["data"]["irregular_keep_rate"] = 0.3

    # Generator-specific data overrides.
    if gen_name in {"periodic_3d_a", "periodic_3d_b"}:
        cfg["data"].update(amp=0.05, n_points=1000, t_max=20.0)
    elif gen_name == "lotka_volterra":
        cfg["data"].update(n_points=1000, t_max=80.0)
    elif gen_name == "glycolytic":
        cfg["data"].update(n_points=1000, t_max=80.0)
    elif gen_name == "forced_vibration":
        # 1601 points at dt=0.01, t_max=16 → 80/20 = 1281/320 train/test,
        # comfortably wider than the 60+60 forecasting window. The longer
        # horizon also reveals more of the negative-damping transient.
        cfg["data"].update(t_max=16.0, dt=0.01)
        cfg["data"]["in_len"] = 60
        cfg["data"]["out_len"] = 60
    elif gen_name == "unstable_oscillator":
        cfg["data"].update(n_points=629)
    elif gen_name == "lorenz63":
        cfg["data"].update(n_points=4000, t_max=40.0)
    elif gen_name == "lorenz96":
        cfg["data"].update(n_points=2000, t_max=20.0)

    out = ROOT / f"{gen_name}__{sampling}.yaml"
    _write(out, cfg)


def main() -> None:
    for gen, extra in PERIODIC_REGULAR:
        make(gen, "regular",  3, extra)
        make(gen, "irregular", 3, extra)
    for gen, extra in APERIODIC_REGULAR:
        in_dim = 2 if gen == "forced_vibration" else 1   # forced has (x, v); unstable has scalar
        make(gen, "regular",   in_dim, extra)
        make(gen, "irregular", in_dim, extra)
    for gen, extra in APERIODIC_IRREGULAR:
        in_dim = extra.get("n_dim", 3)
        make(gen, "irregular", in_dim, extra)
        # regular variant of Lorenz isn't in the 2x2 (treated as
        # naturally chaotic + irregular in our protocol), but keep one
        # available in case we want it.
        make(gen, "regular",   in_dim, extra)


if __name__ == "__main__":
    main()
