"""Aggregate per-run ``final_metrics.json`` files into a single table.

Walks ``results/<exp_id>/<model>/seed<S>/final_metrics.json``, pulls the
headline metrics + the ``slurm-<job>.out`` wall-clock if present, and
prints a Markdown table ready to paste into the matching
``docs/experiments/E###_*.md`` doc.

Usage::

    python code/scripts/aggregate_results.py results/periodic_3d_a__regular
    python code/scripts/aggregate_results.py results/periodic_3d_a__regular --csv out.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def _read_json(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _slurm_walltime(out_dir: Path, exp_id: str, model: str) -> tuple[str, str]:
    """Return (wall_clock, job_id) by grepping the SLURM .out files in
    the same project's ``results/`` dir."""
    results_root = out_dir.parent.parent  # results/<exp_id>/<model>/seedX/ → results/
    pattern = f"slurm-*{model}-*.out"
    candidates = sorted(results_root.glob(pattern))
    if not candidates:
        return "—", "—"
    log = candidates[-1].read_text(errors="replace")
    m = re.search(r"slurm-[^-]+-(\d+)\.out", str(candidates[-1].name))
    job_id = m.group(1) if m else "—"
    # Wall-clock isn't in the .out itself; ask sacct instead via env (skip
    # if not available). For now just return job_id.
    return "—", job_id


def collect(exp_root: Path) -> list[dict]:
    rows: list[dict] = []
    for run_dir in sorted(exp_root.glob("*/seed*")):
        if not run_dir.is_dir():
            continue
        model = run_dir.parent.name
        seed_str = run_dir.name
        seed = int(seed_str.replace("seed", ""))
        final = _read_json(run_dir / "final_metrics.json")
        if final is None:
            status = "running" if (run_dir / "metrics.jsonl").exists() else "missing"
        else:
            status = "done"
        hp = _read_json(run_dir / "hparams.json") or {}
        wallclock, job_id = _slurm_walltime(run_dir, exp_root.name, model)
        rows.append({
            "model": model,
            "seed": seed,
            "status": status,
            "n_params": (final or {}).get("n_params", "—"),
            "best_test_mse": (final or {}).get("best_test_mse", "—"),
            "test_mse": (final or {}).get("mse", "—"),
            "test_mae": (final or {}).get("mae", "—"),
            "test_mape": (final or {}).get("mape", "—"),
            "epochs_run": (final or {}).get("epochs_run", "—"),
            "wallclock": wallclock,
            "job_id": job_id,
            "config": hp.get("model", {}).get("solver", "—"),
        })
    return rows


def _fmt(v):
    if isinstance(v, float):
        if v == 0 or 1e-3 <= abs(v) < 1e4:
            return f"{v:.4f}"
        return f"{v:.3e}"
    return str(v)


def _print_md(rows: list[dict], title: str) -> None:
    cols = ["model", "seed", "status", "n_params", "best_test_mse",
            "test_mse", "test_mape", "epochs_run", "job_id"]
    print(f"\n### Results — `{title}`\n")
    print("| " + " | ".join(cols) + " |")
    print("| " + " | ".join("---" for _ in cols) + " |")
    for r in rows:
        print("| " + " | ".join(_fmt(r.get(c, "—")) for c in cols) + " |")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("exp_root", type=Path, help="results/<exp_id> directory")
    p.add_argument("--csv", type=Path, help="optional CSV output")
    args = p.parse_args()

    rows = collect(args.exp_root)
    if not rows:
        print(f"no runs found under {args.exp_root}")
        return
    _print_md(rows, args.exp_root.name)
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote CSV → {args.csv}")


if __name__ == "__main__":
    main()
