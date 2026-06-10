#!/usr/bin/env python3
"""Aggregate the committed run results into human-readable tables.

Walks ``results/<run>/<model>/seed<S>/final_metrics.json`` and summarizes the
``best_test_mse`` of every run, averaged over seeds. Produces:

* ``results/summary_main.csv``      — headline benchmark (7 models x 8 cells)
* ``results/summary_ablation.csv``  — fusion-ladder ablation (fast protocol)

and prints both tables as Markdown to stdout.

No training is required: it reads only the small JSON summaries already in
``results/``. Run from anywhere:

    python scripts/summarize_results.py
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SEEDS = (0, 1, 2)

# --- headline benchmark: one model per directory prefix (1000 epochs, rtol 1e-3) ---
HEADLINE = [  # (display name, dir prefix, model subdir)
    ("NODE",      "fnode_long",  "faithful_node"),
    ("ODE-RNN",   "frnn_long",   "faithful_ode_rnn"),
    ("FODE",      "ff_long",     "faithful_fode"),
    ("DIFFODE",   "fd_long",     "faithful_diffode"),
    ("Hybrid",    "hyb_long",    "ode_rnn_hybrid"),
    ("Hyb-NoT",   "hybnt_long",  "ode_rnn_hybrid_notime"),
    ("LSA-NODE",  "lsa_long",    "lsa_node"),
]
CELLS = [  # display order = report Table 1
    "periodic_3d_a__regular",
    "periodic_3d_a__irregular",
    "periodic_3d_b__regular",
    "lotka_volterra__regular",
    "glycolytic__regular",
    "unstable_oscillator__regular",
    "lorenz63__irregular",
    "lorenz96__irregular",
]
# fusion ladder used in the ablation figure (left = "vocabulary only", right = restored)
LADDER = [
    ("LSA-NODE",     "lsa_node"),
    ("STFT-FODE",    "stft_fode"),
    ("ODE-RNN-FODE", "ode_rnn_fode"),
    ("Gated",        "ode_rnn_gated"),
    ("Hybrid",       "ode_rnn_hybrid"),
    ("Hyb-NoT",      "ode_rnn_hybrid_notime"),
]


def best_mse(run_dir: Path, model: str):
    """Mean best_test_mse over the available seeds, or None if no runs found."""
    vals = []
    for s in SEEDS:
        f = run_dir / model / f"seed{s}" / "final_metrics.json"
        if f.exists():
            vals.append(json.loads(f.read_text())["best_test_mse"])
    return sum(vals) / len(vals) if vals else None


def fmt(v):
    return "n/a" if v is None else f"{v:.2e}"


def md_table(rows, header):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def main():
    # ---- headline benchmark ----
    grid = {}  # (model, cell) -> mean mse
    for disp, prefix, model in HEADLINE:
        for cell in CELLS:
            grid[(disp, cell)] = best_mse(RESULTS / f"{prefix}_{cell}", model)

    # best (lowest) model per cell
    best_per_cell = {}
    for cell in CELLS:
        cands = [(grid[(d, cell)], d) for d, _, _ in HEADLINE if grid[(d, cell)] is not None]
        best_per_cell[cell] = min(cands)[1] if cands else None

    short = {c: c.replace("__regular", " (reg)").replace("__irregular", " (irr)")
             .replace("periodic_3d_", "periodic-").replace("lotka_volterra", "lotka")
             .replace("unstable_oscillator", "unstable").replace("glycolytic", "glyco")
             .replace("lorenz", "lorenz") for c in CELLS}

    md_rows = []
    for cell in CELLS:
        row = [short[cell]]
        for disp, _, _ in HEADLINE:
            v = grid[(disp, cell)]
            cell_str = fmt(v)
            if disp == best_per_cell[cell]:
                cell_str = f"**{cell_str}**"
            row.append(cell_str)
        md_rows.append(row)
    headline_md = md_table(md_rows, ["cell"] + [d for d, _, _ in HEADLINE])

    with open(RESULTS / "summary_main.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cell"] + [d for d, _, _ in HEADLINE])
        for cell in CELLS:
            w.writerow([cell] + [("" if grid[(d, cell)] is None else f"{grid[(d, cell)]:.6e}")
                                 for d, _, _ in HEADLINE])

    # ---- ablation (fast protocol) ----
    abl_cells = ["periodic_3d_a__regular", "periodic_3d_a__irregular"]
    abl_rows = []
    with open(RESULTS / "summary_ablation.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cell"] + [d for d, _ in LADDER])
        for cell in abl_cells:
            row_csv, row_md = [], [short[cell]]
            for _, model in LADDER:
                v = best_mse(RESULTS / f"fast_{cell}", model)
                row_csv.append("" if v is None else f"{v:.6e}")
                row_md.append(fmt(v))
            w.writerow([cell] + row_csv)
            abl_rows.append(row_md)
    ablation_md = md_table(abl_rows, ["cell"] + [d for d, _ in LADDER])

    print("## Headline benchmark — best test MSE (mean over 3 seeds)\n")
    print(headline_md)
    print("\n_Bold = best model in that cell._\n")
    print("## Ablation (fast 300-epoch protocol) — best test MSE\n")
    print(ablation_md)
    print(f"\nWrote {RESULTS/'summary_main.csv'} and {RESULTS/'summary_ablation.csv'}")


if __name__ == "__main__":
    main()
