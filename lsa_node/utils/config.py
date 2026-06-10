"""YAML config loader + CLI overrides.

Convention: every entry in ``code/configs/<name>.yaml`` is a nested dict
with top-level keys ``model``, ``train``, ``data``. The CLI accepts
``--key value`` pairs where ``key`` is a dotted path
(``--model.n_fft 32``) and ``value`` is parsed as YAML so types are
preserved.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())


def merge_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``key=value`` or ``key value`` pairs to a nested config in-place.

    Args:
        cfg: nested dict from :func:`load_yaml`.
        overrides: list of "dotted.key=value" strings, e.g.
            ``["model.n_fft=32", "train.lr=5e-4"]``.

    Each ``value`` is parsed with ``yaml.safe_load`` so ``"true"`` becomes
    ``True``, ``"32"`` becomes ``32`` (int), etc.
    """
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got {item!r}")
        key, raw_val = item.split("=", 1)
        keys = key.split(".")
        node = cfg
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = yaml.safe_load(raw_val)
    return cfg


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True, help="YAML config")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None,
                        help="cuda / cpu (default: cuda if available)")
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument(
        "-o", "--override", action="append", default=[],
        help='dotted overrides, e.g. -o train.lr=1e-4 -o model.n_fft=16',
    )
