"""Generate 1000-epoch tighter-tolerance configs for the cells where
ODE-RNN-Hybrid-NoTime is close to ODE-RNN's win.

For each cell we generate ``long_<cell>.yaml`` — identical to
``fast_<cell>.yaml`` but with:
* 1000 epochs (vs 300)
* dopri5 rtol=1e-5, atol=1e-7 (vs 1e-3 / 1e-4) — matches FODE paper
"""

from __future__ import annotations

from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1] / "configs"


def upgrade(cell: str) -> None:
    fast = ROOT / f"fast_{cell}.yaml"
    cfg = yaml.safe_load(fast.read_text())
    cfg["exp_id"] = f"long_{cell}"
    cfg["model"]["rtol"] = 1.0e-5
    cfg["model"]["atol"] = 1.0e-7
    cfg["train"]["epochs"] = 1000
    out = ROOT / f"long_{cell}.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"wrote {out.relative_to(ROOT.parent.parent)}")


for cell in [
    "periodic_3d_a__regular",
    "periodic_3d_a__irregular",
    "periodic_3d_b__regular",
    "lotka_volterra__regular",
    "glycolytic__regular",
    "unstable_oscillator__regular",
    "lorenz63__irregular",
    "lorenz96__irregular",
]:
    upgrade(cell)
