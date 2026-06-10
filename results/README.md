# `results/` — committed run summaries

This directory holds the per-run result summaries that back the report, so the
numbers can be inspected without retraining. See [`../RESULTS.md`](../RESULTS.md)
for the aggregated headline tables.

## What is (and isn't) here

To keep the repository small, each run keeps only its two lightweight summary
files. The large artifacts produced during training are **not** committed:

| Committed | Excluded |
|---|---|
| `final_metrics.json` — final/best metrics for the run | `metrics.jsonl` — per-epoch curves |
| `hparams.json` — the exact resolved config | `checkpoints/*.pt` — model weights |
|  | `tb/` — TensorBoard event files, SLURM logs |

Only the two result sets used in the report are included: the **headline
benchmark** (`*_long_*`) and the **ablation** (`fast_*`). Earlier proposal-era
runs at a tighter solver tolerance are omitted.

## Layout

```
results/
└── <run>/<model>/seed<S>/
    ├── final_metrics.json
    └── hparams.json
```

Each leaf is one `(run, model, seed)` cell. `<S>` ∈ {0, 1, 2}.

### Run-directory naming

A cell name is `<system>__<sampling>`, e.g. `glycolytic__regular` or
`lorenz63__irregular`.

**Headline benchmark** — `<prefix>_long_<cell>/`, 1000 epochs, solver tolerance
`rtol=1e-3, atol=1e-4`. Each directory holds exactly one model; the prefix names
which one:

| Prefix | Model subdir | Report name |
|---|---|---|
| `fnode_long_` | `faithful_node` | NODE |
| `frnn_long_`  | `faithful_ode_rnn` | ODE-RNN |
| `ff_long_`    | `faithful_fode` | FODE |
| `fd_long_`    | `faithful_diffode` | DIFFODE |
| `hyb_long_`   | `ode_rnn_hybrid` | Hybrid |
| `hybnt_long_` | `ode_rnn_hybrid_notime` | Hyb-NoT |
| `lsa_long_`   | `lsa_node` | LSA-NODE |

These eight cells appear under every prefix: `periodic_3d_a__regular`,
`periodic_3d_a__irregular`, `periodic_3d_b__regular`, `lotka_volterra__regular`,
`glycolytic__regular`, `unstable_oscillator__regular`, `lorenz63__irregular`,
`lorenz96__irregular`.

**Ablation** — `fast_<cell>/`, the shorter 300-epoch protocol used for the fusion
ladder. Unlike the headline dirs, each `fast_` cell contains many model subdirs
(`lsa_node`, `stft_fode`, `ode_rnn_fode`, `ode_rnn_gated`, `ode_rnn_hybrid`,
`ode_rnn_hybrid_notime`, and the plain baselines).

## `final_metrics.json` schema

```json
{
  "mse": 0.0557,            // final-epoch test MSE
  "mae": 0.2116,            // final-epoch test MAE
  "mape": 38.67,            // final-epoch test MAPE (%)
  "best_test_mse": 0.0481,  // best test MSE over training  <-- used in all tables
  "n_params": 27939,        // trainable parameter count
  "epochs_run": 1000
}
```

`hparams.json` is the fully resolved config for the run (model block, training
block, data block), i.e. the YAML in `configs/` after defaults are applied.

## Regenerating the summary tables

```bash
python scripts/summarize_results.py
```

Reads every `final_metrics.json` here, averages `best_test_mse` over seeds, and
writes `summary_main.csv` (headline) and `summary_ablation.csv`, printing both as
Markdown. No training or GPU required.

## Reproducing the runs from scratch

Train a single cell (writes a fresh `results/<exp_id>/seed<S>/`):

```bash
python -m lsa_node.train --config configs/glycolytic__regular.yaml --seed 0
```

Or launch the full sweeps with `make main` / `make ablation` (see the top-level
README). Reproducing every run requires a GPU and many hours.
