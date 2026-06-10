"""Master comparison table for the Phase 5 2×2 grid.

Walks ``results/<cell>/<model>/seed<S>/final_metrics.json`` for every
cell × model × seed and emits:

* a Markdown table aggregated by (cell, model) with mean ± std over
  seeds (drops into ``docs/experiments/`` write-ups);
* a LaTeX-ready table grouped by (periodicity × sampling) → drops into
  ``paper/main.tex``;
* an aggregated CSV at ``results/master_table.csv``.

Usage::

    python code/scripts/master_table.py
    python code/scripts/master_table.py --csv results/master_table.csv
    python code/scripts/master_table.py --latex
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]

GRID = {
    ("periodic",  "regular"):   ["periodic_3d_a__regular", "periodic_3d_b__regular",
                                 "lotka_volterra__regular", "glycolytic__regular"],
    ("periodic",  "irregular"): ["periodic_3d_a__irregular", "periodic_3d_b__irregular",
                                 "lotka_volterra__irregular", "glycolytic__irregular"],
    ("aperiodic", "regular"):   ["forced_vibration__regular", "unstable_oscillator__regular"],
    ("aperiodic", "irregular"): ["lorenz63__irregular", "lorenz96__irregular"],
}
MODELS = ["lsa_node", "node", "ncde"]
DISPLAY_MODEL = {"lsa_node": "LSA-NODE", "node": "NODE", "ncde": "NCDE"}


def _load(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def collect() -> list[dict]:
    rows: list[dict] = []
    for (periodicity, sampling), cells in GRID.items():
        for cell in cells:
            for model in MODELS:
                seed_metrics: dict[int, dict] = {}
                for sd in (PROJECT / "results" / cell / model).glob("seed*"):
                    seed_n = int(sd.name.replace("seed", ""))
                    final = _load(sd / "final_metrics.json")
                    if final is not None:
                        seed_metrics[seed_n] = final
                if not seed_metrics:
                    continue
                mses = [m["best_test_mse"] for m in seed_metrics.values()]
                mapes = [m["mape"] for m in seed_metrics.values()]
                row = {
                    "periodicity": periodicity,
                    "sampling": sampling,
                    "cell": cell,
                    "model": model,
                    "n_seeds": len(seed_metrics),
                    "mse_mean": statistics.fmean(mses),
                    "mse_std":  statistics.stdev(mses) if len(mses) > 1 else 0.0,
                    "mape_mean": statistics.fmean(mapes),
                    "mape_std":  statistics.stdev(mapes) if len(mapes) > 1 else 0.0,
                    "n_params": next(iter(seed_metrics.values())).get("n_params", "—"),
                }
                rows.append(row)
    return rows


def _fmt_mse(mean: float, std: float, n: int) -> str:
    if mean == 0:
        return "0"
    s = f"{mean:.3e}"
    if n > 1:
        s += f" ± {std:.0e}"
    return s


def print_md(rows: list[dict]) -> None:
    print("\n## Phase 5.1 — synthetic 2×2 grid (mean ± std over seeds)\n")
    print("| cell | model | seeds | best test MSE | MAPE (%) | n_params |")
    print("| --- | --- | --- | --- | --- | --- |")
    by_cell = {}
    for r in rows:
        by_cell.setdefault(r["cell"], []).append(r)
    for cell, group in sorted(by_cell.items()):
        for r in group:
            print(f"| {cell} | {DISPLAY_MODEL[r['model']]} | {r['n_seeds']} | "
                  f"{_fmt_mse(r['mse_mean'], r['mse_std'], r['n_seeds'])} | "
                  f"{r['mape_mean']:.2f}{' ± ' + format(r['mape_std'], '.2f') if r['n_seeds'] > 1 else ''} | "
                  f"{r['n_params']:,} |")


def print_latex(rows: list[dict]) -> None:
    print("\n% --- Phase 5.1 main results table ---")
    print(r"\begin{tabular}{llrr}")
    print(r"\toprule")
    print(r"Cell & Model & best test MSE & MAPE (\%) \\")
    print(r"\midrule")
    by_cell = {}
    for r in rows:
        by_cell.setdefault(r["cell"], []).append(r)
    for cell in sorted(by_cell):
        for j, r in enumerate(by_cell[cell]):
            cell_str = cell.replace("_", r"\_") if j == 0 else ""
            mse = _fmt_mse(r["mse_mean"], r["mse_std"], r["n_seeds"]).replace("±", r"$\pm$")
            mape = f"{r['mape_mean']:.2f}"
            if r["n_seeds"] > 1:
                mape += rf" $\pm$ {r['mape_std']:.2f}"
            print(rf"  {cell_str} & {DISPLAY_MODEL[r['model']]} & {mse} & {mape} \\")
        print(r"  \midrule")
    print(r"\bottomrule")
    print(r"\end{tabular}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path)
    p.add_argument("--latex", action="store_true")
    args = p.parse_args()

    rows = collect()
    if not rows:
        print("no completed runs found yet")
        return

    print_md(rows)
    if args.latex:
        print_latex(rows)
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
