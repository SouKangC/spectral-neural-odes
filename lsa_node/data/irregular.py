"""Irregular **real-world** datasets.

Lorenz63 / Lorenz96 are generated synthetically in
``lsa_node.data.synthetic`` and then run through
``synthetic.poisson_subsample``; they are *not* loaded here.

Loaders below:
    - USHCN daily climate (drop 50% randomly).
    - PhysioNet 2012 Challenge (round to 6-min, drop ~30%).
"""

from __future__ import annotations

from pathlib import Path

import torch


def load_ushcn(
    data_dir: str | Path,
    drop_frac: float = 0.5,
    seed: int = 0,
) -> dict[str, torch.utils.data.Dataset]:
    raise NotImplementedError("scaffold only")


def load_physionet2012(
    data_dir: str | Path,
    round_minutes: int = 6,
    seed: int = 0,
) -> dict[str, torch.utils.data.Dataset]:
    raise NotImplementedError("scaffold only")
