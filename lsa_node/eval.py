"""Evaluation entry point. Loads a trained checkpoint and dumps metrics +
attention visualizations.

Usage::

    python -m lsa_node.eval --ckpt results/.../ckpt.pt --config configs/...yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from lsa_node.train import build_dataset, build_model, load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LSA-NODE evaluator")
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save-attn", action="store_true", help="dump (τ, ω) attention maps")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    raise NotImplementedError("scaffold only")


if __name__ == "__main__":
    main()
