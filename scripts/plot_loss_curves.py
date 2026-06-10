"""Plot training / test loss curves for one experiment.

Walks ``results/<exp_id>/<model>/seed<S>/metrics.jsonl`` and renders one
PNG per metric tag, with one line per (model, seed).

Usage::

    python code/scripts/plot_loss_curves.py results/periodic_3d_a__regular
    # writes results/periodic_3d_a__regular/figs/{train,test}_*.png
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_metrics(jsonl: Path) -> dict[str, list[tuple[int, float]]]:
    series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    if not jsonl.exists():
        return series
    for line in jsonl.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        series[rec["tag"]].append((rec["step"], rec["value"]))
    return series


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("exp_root", type=Path)
    p.add_argument("--tags", nargs="*", default=None,
                   help="tags to plot (default: all)")
    p.add_argument("--logy", action="store_true",
                   help="use log scale on y-axis")
    args = p.parse_args()

    fig_dir = args.exp_root / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)

    by_tag_model: dict[str, dict[str, list[tuple[int, float]]]] = defaultdict(dict)
    for run_dir in sorted(args.exp_root.glob("*/seed*")):
        model = run_dir.parent.name
        seed = run_dir.name
        series = _load_metrics(run_dir / "metrics.jsonl")
        for tag, pts in series.items():
            by_tag_model[tag][f"{model}/{seed}"] = pts

    tags = args.tags or sorted(by_tag_model)
    for tag in tags:
        runs = by_tag_model.get(tag, {})
        if not runs:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        for label, pts in sorted(runs.items()):
            xs, ys = zip(*pts)
            ax.plot(xs, ys, label=label, linewidth=1.4)
        ax.set_title(f"{args.exp_root.name} — {tag}")
        ax.set_xlabel("step" if "step" in tag else "epoch")
        ax.set_ylabel(tag)
        if args.logy:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        safe_tag = tag.replace("/", "_")
        out = fig_dir / f"{safe_tag}.png"
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
