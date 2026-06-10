# LSA-NODE: Localized Spectral Attention for Neural ODEs

Code for our ECE 228 project on putting a spectral inductive bias inside a Neural
ODE. We propose **LSA-NODE**, a Neural ODE whose vector field runs attention over
the short-time spectrum of the state, and study why it fails: the attention output
is trapped in the span of a spectral dictionary, so it memorizes training spectra
instead of learning a generalizing field. The diagnosis motivates a working fix,
**ODE-RNN-Hybrid**, which sums a time-domain MLP with a dense spectral MLP inside
an ODE-RNN encoder.

The written report is submitted separately; this repository contains the code,
experiment configs, and tests.

## Layout

```
.
├── lsa_node/            Python package
│   ├── models/          NODE, ODE-RNN, FODE, DIFFODE, LSA-NODE, the Hybrid variants
│   ├── data/            synthetic generators (Lotka-Volterra, glycolytic, Lorenz, ...)
│   ├── train.py         training entry point  (python -m lsa_node.train)
│   ├── eval.py          evaluation entry point
│   ├── losses.py        MSE and Hoyer-sparsity losses
│   └── utils/           seeding, config, logging, plotting helpers
├── configs/             one YAML per (dataset × sampling × model) cell
├── scripts/             reproduction drivers and config generators
├── tests/               unit tests (pytest)
├── results/             committed per-run metric summaries (no retraining needed)
├── RESULTS.md           headline tables, generated from results/
├── pyproject.toml       package metadata and dependencies
├── requirements.txt     pinned dependencies
└── Makefile             install / test / reproduce shortcuts
```

## Results

You can read the numbers without running anything: see **[RESULTS.md](RESULTS.md)**
for the headline benchmark and ablation tables. They are generated from the small
per-run summaries committed under `results/` (see
[results/README.md](results/README.md) for the layout and JSON schema):

```bash
python scripts/summarize_results.py     # rebuilds results/summary_*.csv + prints the tables
```

## Install

Python ≥ 3.10 with PyTorch (CUDA optional). From the repository root:

```bash
pip install -e ".[dev,log]"      # or: pip install -r requirements.txt
```

## Quickstart

```bash
# fast smoke run (a few epochs, CPU is fine)
python -m lsa_node.train --config configs/smoke.yaml

# train one model on one cell; results land in results/<exp_id>/seed<seed>/
python -m lsa_node.train --config configs/glycolytic__regular.yaml --seed 0

# swap the model without editing the config
python -m lsa_node.train --config configs/glycolytic__regular.yaml \
    --model ode_rnn_hybrid_notime --seed 0

# run the unit tests
pytest -q tests
```

Each config sets the dataset, sampling regime (`regular` / `irregular`), model, and
training budget. Model names match the files in `lsa_node/models/`
(`node`, `ode_rnn`, `fode`, `diffode`, `lsa_node`, `ode_rnn_hybrid`,
`ode_rnn_hybrid_notime`, ...).

## Reproducing the benchmark

`configs/` holds the eight benchmark cells (periodic_3d_a/b, lotka_volterra,
glycolytic, unstable_oscillator, lorenz63, lorenz96) under regular and irregular
sampling, plus the per-model run sets used for the tables. The headline numbers use
a single solver tolerance (`rtol=1e-3`, `atol=1e-4`), 1000 epochs, and seeds
`{0,1,2}`:

```bash
make main        # bash scripts/reproduce_table_main.sh
make ablation    # bash scripts/reproduce_table_ablation.sh
```

> The `scripts/sbatch_*.sh` launchers and a few helpers target the SLURM-based UIUC
> Delta cluster and contain machine-specific paths; adapt them for your environment.

## Team

Jinshi He, Nishant Arya, Yuan Liu — Department of Electrical and Computer
Engineering, UC San Diego.
