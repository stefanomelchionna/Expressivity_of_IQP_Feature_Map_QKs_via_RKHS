# Expressivity of IQP Feature-Map Quantum Kernels via Reproducing Kernel Hilbert Spaces

Code, data generators and results for the paper

> V. Repetto, S. Melchionna, R. D'Aurizio, *Expressivity of IQP Feature-Map Quantum Kernels via Reproducing
> Kernel Hilbert Spaces*, [arXiv:2610.23692](https://arxiv.org/abs/2610.23692).

The paper characterises the RKHS of quantum kernels (QKs) built on the IQP feature map and shows that the bandwidth
`b` moves the kernel through three regimes: Gaussian-like (small `b`), multimodal (intermediate `b`) and periodic
(large `b`). The numerical experiments of Section IV test this on two families of synthetic data, periodic and
multimodal, for two tasks, supervised classification (SVM) and unsupervised anomaly detection (One-Class SVM),
and compare the QKs with a classical Gaussian (RBF) kernel. This repository reproduces all of them.

## Repository structure

```
.
├── src/                        library code
│   ├── QK.py                   IQP feature-map fidelity quantum kernel (PennyLane + JAX)
│   ├── data.py                 generators of the periodic (Eq. 6) and multimodal (Eq. 7) datasets
│   └── utils.py                classical RBF kernel with default gamma, kernel diagnostics
├── Data_gen/                   (optional) export the datasets of the 4 experiments to CSV
├── experiments/
│   ├── QK_computation/         step 1: generate the data and compute the quantum kernels
│   └── evaluation/             step 2: train/evaluate SVM / One-Class SVM on the kernels -> output/<Experiment>/*.csv
├── gamma_study/
│   └── compute_gamma_decades.py  classical RBF baseline for gamma in {0.01, 0.1, 1, 10, 100} and the default gamma
├── code_for_papers_plots/
│   └── make_report_plots.py    all the figures of the paper, from the CSVs in output/
├── config/                     configs of Data_gen/ (datasets used in the paper)
├── configuration_files/        configs of experiments/ (kernel grids and evaluation paths)
├── output/                     results used in the paper (CSV)
│   ├── Classification_periodic/classification_periodic_results.csv
│   ├── AD_periodic/AD_periodic_results.csv
│   ├── Classification_multimodal/classification_multimodal_results.csv
│   ├── AD_multimodal/AD_results.csv
│   └── RBF_gamma_scan/         classical RBF scan (gamma_decades_*.csv) and its frozen reference
└── requirements.txt
```

Generated datasets and kernels go to `Simulated_Data/` and figures to `images/report_plots/`; both are git-ignored.

## The four experiments

| Experiment (paper section) | Data | Task / model | Kernel grid | Scripts | Results |
|---|---|---|---|---|---|
| Periodic classification (IV A, Fig. 9, S1) | uniform in `[0,1]^D`, labels by Eq. (6), `ω ∈ {1,2,4,6,8}`; train seed 0, test seed 1 (500 points) | SVM, test F1 | `b ∈ {1,2,4,6,8}` | `QK_computation_classification_periodic.py`, `Classification_periodic_eval.py` | `output/Classification_periodic/` |
| Periodic anomaly detection (IV A, Fig. 10, S2) | inliers = class 0 of Eq. (6), + 20% uniform outliers | One-Class SVM (`nu=0.2`), F1 | `b ∈ {1,2,4,6,8}` | `QK_computation_AD_periodic.py`, `AD_periodic_eval.py` | `output/AD_periodic/` |
| Multimodal classification (IV B, Fig. 11a, S3) | two Gaussian mixtures with `C ∈ {2,4,8}` centres each; 500 test points | SVM, test F1 | `b ∈ {1/4,1/2,1,3/2,2}` | `QK_computation_classification_multimodal.py`, `Classification_multimodal_eval.py` | `output/Classification_multimodal/` |
| Multimodal anomaly detection (IV B, Fig. 11b, S4) | Gaussian mixture with `C ∈ {2,4,8}` centres, + 20% uniform outliers | One-Class SVM (`nu=0.2`), F1 | `b ∈ {1/4,1/2,1,3/2,2}` | `QK_computation_AD_multimodal.py`, `AD_multimodal_eval.py` | `output/AD_multimodal/` |

All experiments use `D ∈ {2,4,6}` features (= number of qubits), `r ∈ {1,2,3}` repetitions of the feature map and
linear or full entanglement. The bandwidth is applied by rescaling the data: periodic data `x ↦ b·π·x`,
multimodal data min-max scaled to `[0, b·π]`. The figures use training sets of 1000 points (`Sampling = 1000`);
the result CSVs and the default configs also contain smaller (and, for the multimodal data, larger) training sets.

Classical baseline: an RBF kernel with scikit-learn's default `γ = 1/(D·Var(X))`, plus
`γ ∈ {0.01, 0.1, 1, 10, 100}` for the min-max bands of Figs. 9(c), 10(c) and 11
(`gamma_study/compute_gamma_decades.py`, same datasets, same models).

### Figures

`code_for_papers_plots/make_report_plots.py` writes every figure to `images/report_plots/` (PNG and PDF):

| Paper | File |
|---|---|
| Fig. 1 – Gaussian RKHS generators | `RKHS_RBF` |
| Fig. 6 – quantum RKHS generators | `RKHS_quantum` |
| Fig. 7 – Fourier transform of the RKHS generators | `RKHS_fourier_transform_heatmaps` |
| Fig. 8 – datasets | `datasets` |
| Fig. 9 – periodic classification | `combined_periodic_classification` |
| Fig. 10 – periodic anomaly detection | `combined_periodic_AD` |
| Fig. 11 – multimodal data | `Trend_multimodal_gammaband` |
| Figs. S1–S4 – per repetition and entanglement | `supplementary_S1` … `supplementary_S4` |

Figs. 2–5 are circuit diagrams and are not generated by code. The individual panels of Figs. 9–11 are also written
as separate files.

## Installation

Python 3.12:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Note: `jax >= 0.8` is not compatible with `pennylane 0.45`; keep the pinned versions.

## How to run

All commands are run from the repository root with `PYTHONPATH=.`.

### Figures only (from the committed results, ~1 min)

```bash
python code_for_papers_plots/make_report_plots.py
```

### Full reproduction

For each experiment, compute the kernels, then evaluate them (example: periodic classification):

```bash
export PYTHONPATH=.

# 1. data + quantum kernels  -> Simulated_Data/Classification_periodic/Kernels/...
python experiments/QK_computation/QK_computation_classification_periodic.py \
    --config configuration_files/config_classification_periodic.json

# 2. SVM on quantum and classical kernels -> output/Classification_periodic/classification_periodic_results.csv
python experiments/evaluation/Classification_periodic_eval.py \
    --config configuration_files/config_classification_periodic_eval.json
```

and the same for the other three experiments:

| Experiment | Step 1 | Step 2 |
|---|---|---|
| Periodic classification | `QK_computation_classification_periodic.py` + `config_classification_periodic.json` | `Classification_periodic_eval.py` + `config_classification_periodic_eval.json` |
| Periodic AD | `QK_computation_AD_periodic.py` + `config_AD_periodic.json` | `AD_periodic_eval.py` + `config_AD_periodic_eval.json` |
| Multimodal classification | `QK_computation_classification_multimodal.py` + `config_classification_multimodal.json` | `Classification_multimodal_eval.py` + `config_classification_multimodal_eval.json` |
| Multimodal AD | `QK_computation_AD_multimodal.py` + `config_AD_multimodal.json` | `AD_multimodal_eval.py` + `config_AD_multimodal_eval.json` |

The `--config` defaults are the files above, so `--config` can be omitted. Then the classical baseline and the figures:

```bash
# 3. classical RBF over the gamma grid -> output/RBF_gamma_scan/gamma_decades_scan_f1_ts.csv (~1 min)
python gamma_study/compute_gamma_decades.py --clf-metric f1_ts
#    optional: test-accuracy version, checked against the frozen reference (prints VERIFY: OK)
python gamma_study/compute_gamma_decades.py --verify

# 4. figures
python code_for_papers_plots/make_report_plots.py
```

Optionally, `Data_gen/data_gen_<experiment>.py --config config/config_<experiment>.json` exports the datasets
alone. The kernel scripts regenerate the same data (same generators and seeds), so this step is not needed.

**Cost.** With `Sampling = 1000` only, steps 1–2 for all four experiments take about 1 hour on an 8-core laptop
(CPU, two experiments in parallel). A 1000 × 1000 kernel takes 0.5 s (2 qubits) to 5 s (6 qubits) to simulate. The default configs include
all the training-set sizes of the result CSVs and take correspondingly longer. The kernels need several GB of disk
in `Simulated_Data/`. To compute only what the figures use, set `n_samples` / `n_samples_train` to `[1000]` in
`configuration_files/`.

### Configuration files

`configuration_files/config_<experiment>.json` (step 1):

| key | meaning |
|---|---|
| `Data_dir` / `output_path` | where data and kernels are written (`Simulated_Data/<Experiment>`) |
| `n_samples_train` / `n_samples` | training-set sizes (dataset sizes for AD) |
| `n_samples_test` | test-set size (classification) |
| `n_qubits` | `D`, number of features = number of qubits |
| `oscillations` / `n_clusters` | `ω` (periodic) / `C` (multimodal) |
| `n_repeats`, `entanglement`, `bandwidth` | `r`, entanglement pattern, `b` |

`configuration_files/config_<experiment>_eval.json` (step 2): `Kernel_dir` (kernels of step 1) and `Output_dir`
(results CSV). `config/config_<experiment>.json` are the corresponding dataset configs of `Data_gen/`.

### Result CSVs

One row per kernel configuration and kernel type (`K_type` = `Quantum` or `Classical`, the latter with default
`γ`). Classification: `acc_tr/ts`, `f1_tr/ts` (weighted F1, used in the figures), `prec_tr/ts`. Anomaly detection:
`F1` (anomaly = positive class, used in the figures), `Accuracy`, `Precision`, `Recall` and anomaly-score
statistics. `K_concentration` (variance of the off-diagonal kernel entries) and `GD` (geometric difference between
the classical and quantum kernels) are diagnostics not used in the paper.

## Reproducibility notes

- Re-running the pipeline reproduces the committed quantum classification results to within ±0.002
  (446/450 periodic and 268/270 multimodal configurations bit-for-bit). The RBF scan reproduces bit-for-bit.
- Anomaly detection is numerically more sensitive. Many points lie on the One-Class SVM decision boundary, so
  float32 differences between library versions change individual F1 values by up to ~0.04, although the kernels
  agree to 1e-7. Averages over configurations change by < 0.01.
- Anomaly detection follows an in-sample protocol: the One-Class SVM is fitted without labels on all points and
  evaluated on the same points.
- With 2 qubits, linear and full entanglement give the same circuit. The committed periodic AD results contain
  only `full` for `D = 2`. The default config computes both, so the average over configurations in Fig. 10(b)
  can differ slightly after a re-run.
- The committed CSVs were produced with an earlier version of `src/utils.py`. For the classical rows of the
  classification CSVs, the test kernel used `γ` computed on the test set (now on the training set; difference
  ≤ 0.006), and `GD` used a randomised norm estimate (now exact). Neither enters any figure: the classical curves
  of the figures come from `output/RBF_gamma_scan/`.
