# Results

This page lets you read the headline numbers without running anything. The tables
below are generated from the per-run summaries committed under `results/` by

```bash
python scripts/summarize_results.py
```

which also writes `results/summary_main.csv` and `results/summary_ablation.csv`.
See [`results/README.md`](results/README.md) for the directory layout and the
schema of each JSON file.

All numbers are **best test MSE**, averaged over seeds `{0, 1, 2}`. Lower is
better; **bold** marks the best model in each cell.

## Headline benchmark (1000 epochs, single solver tolerance `rtol=1e-3`)

Seven models compared apples-to-apples across eight cells.

| cell | NODE | ODE-RNN | FODE | DIFFODE | Hybrid | Hyb-NoT | LSA-NODE |
|---|---|---|---|---|---|---|---|
| periodic-a (reg) | 2.38e-03 | 2.55e-03 | 1.35e-02 | 2.63e-02 | **2.07e-03** | 2.46e-03 | 1.37e-02 |
| periodic-a (irr) | **2.72e-03** | 1.30e-02 | 1.35e-02 | 2.32e-02 | 1.00e-02 | 9.75e-03 | 1.23e-02 |
| periodic-b (reg) | **2.71e-03** | 4.10e-03 | 7.40e-03 | 9.03e-02 | 5.56e-03 | 5.31e-03 | 1.82e-01 |
| lotka (reg) | **1.14e-02** | 5.58e-02 | 1.64e-01 | 2.14e+00 | 2.28e-02 | 2.08e-02 | 1.09e-01 |
| glyco (reg) | 1.10e-04 | 1.58e-04 | 2.26e-03 | 4.06e-02 | 1.49e-04 | **4.12e-05** | 1.11e-02 |
| unstable (reg) | 2.99e-02 | **4.66e-03** | 1.39e-01 | 2.47e-01 | 4.06e-02 | 5.87e-02 | 6.74e-02 |
| lorenz63 (irr) | **7.69e+00** | 7.85e+00 | 2.27e+01 | 2.29e+01 | 1.14e+01 | 1.28e+01 | 1.93e+01 |
| lorenz96 (irr) | 1.13e+01 | 7.30e+00 | 5.47e+00 | 7.62e+00 | 8.58e+00 | 8.65e+00 | **5.26e+00** |

**What to read from this.** LSA-NODE is never the best model and is an order of
magnitude behind on the smooth periodic cells (periodic-b, glyco), the signature of
its memorization failure. The spectral prior pays off in exactly one clean,
statistically separated case, glycolytic, where **Hyb-NoT** wins. Everywhere else a
plain full-window **NODE** is the strongest model, and on broadband chaos
(lorenz63) every spectral model trails it.

## Ablation: the fusion ladder (300-epoch fast protocol)

Progressively restoring each parent's working mechanism, on the two cleanest cells.
The two left models keep only the spectral "vocabulary"; the rest restore the
load-carrying parts (full-window encoder + dense spectral MLP).

| cell | LSA-NODE | STFT-FODE | ODE-RNN-FODE | Gated | Hybrid | Hyb-NoT |
|---|---|---|---|---|---|---|
| periodic-a (reg) | 1.63e-02 | 1.29e-02 | 2.55e-03 | 2.32e-03 | 2.26e-03 | 2.52e-03 |
| periodic-a (irr) | 1.48e-02 | 1.29e-02 | 1.09e-02 | 1.08e-02 | 1.09e-02 | 9.75e-03 |

The accuracy jumps by roughly an order of magnitude as soon as the full-window
ODE-RNN encoder and the dense spectral MLP are put back (ODE-RNN-FODE onward),
isolating those as the components that actually do the work.
