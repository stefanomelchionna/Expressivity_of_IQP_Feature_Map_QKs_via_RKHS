#!/usr/bin/env python
"""Classical RBF metrics on the manuscript's 48 configurations, for five decade gammas.

Classical baseline of Figures 9(c), 10(c) and 11 of the paper. Depends only on numpy / pandas /
scikit-learn and on the dataset generators in ``src/data.py``, so the datasets are exactly those
of the quantum experiments.

What it computes, for every configuration of the manuscript's grid
(4 tasks x 3 dimensions x 5 or 3 structure values = 48 datasets):

* the classical RBF metric at ``gamma in (0.01, 0.1, 1, 10, 100)``;
* the same metric at the manuscript's default ``gamma = 1/(D * Var(X))``
  (scikit-learn's ``gamma='scale'``, taken over the flattened training matrix).

Protocol (identical to the reference run whose result table is kept in
``output/RBF_gamma_scan/reference_scan_5gammas.csv``):

* classification -- ``SVC(kernel='precomputed', random_state=42)`` fit on the training
  kernel, scored by test-set accuracy (or test-set weighted F1 with ``--clf-metric f1_ts``, the
  same ``average='weighted'`` F1 as ``f1_ts`` in ``experiments/evaluation/Classification_*_eval.py``);
  the gamma is applied to both kernels;
* anomaly detection -- ``OneClassSVM(kernel='precomputed', nu=0.2)`` fit and scored on
  the same points (the manuscript's transductive protocol), binary F1 with the anomaly
  as the positive class.

Usage::

    python gamma_study/compute_gamma_decades.py --clf-metric f1_ts   # used by the paper figures, writes *_f1_ts.csv
    python gamma_study/compute_gamma_decades.py --verify             # test accuracy + check against the reference
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, OneClassSVM

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "output" / "RBF_gamma_scan"
SCAN_CSV = OUT_DIR / "gamma_decades_scan.csv"
SUMMARY_CSV = OUT_DIR / "gamma_decades_summary.csv"
#: Frozen reference results at these same five gammas, used only by ``--verify``.
REF_DIR = OUT_DIR

#: The five gamma values under study.
GAMMAS = (0.01, 0.1, 1.0, 10.0, 100.0)

#: The manuscript's structural sweeps and sample sizes (as in ``config/config_*.json``).
DIMENSIONS = (2, 4, 6)
OSCILLATIONS = (1, 2, 4, 6, 8)
N_CLUSTERS = (2, 4, 8)
N_SAMPLES = 1000
N_SAMPLES_TEST = 500


# ---------------------------------------------------------------------------
# datasets: rebuilt with the parent repository's generators, verbatim
# ---------------------------------------------------------------------------


def parent_data():
    """Import the dataset generators of ``src/data.py``."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    import src.data as data

    return data


def build(task: str, n_features: int, structure: int):
    """Return ``(X_train, y_train, X_test, y_test)`` for one configuration.

    ``X_test`` is ``None`` for the anomaly-detection tasks, whose protocol is
    transductive. Every seed and split matches ``Data_gen/data_gen_*.py``.
    """
    data = parent_data()

    if task == "Classification_periodic":
        x_tr, y_tr = data.compute_periodic_data_classification(
            oscillations=structure, sample_size=N_SAMPLES, seed=0,
            number_of_features=n_features,
        )
        x_ts, y_ts = data.compute_periodic_data_classification(
            oscillations=structure, sample_size=N_SAMPLES_TEST, seed=1,
            number_of_features=n_features,
        )
        return np.asarray(x_tr, float), np.asarray(y_tr), np.asarray(x_ts, float), np.asarray(y_ts)

    if task == "Classification_multimodal":
        df = data.create_blob_multimodal_classification(
            n_features=n_features, sample_size=N_SAMPLES + N_SAMPLES_TEST,
            number_of_blobs=structure,
        )
        X, y = df.drop(columns=["label"]).values, df["label"].values
        x_tr, x_ts, y_tr, y_ts = train_test_split(
            X, y, test_size=N_SAMPLES_TEST, random_state=42
        )
        return np.asarray(x_tr, float), y_tr, np.asarray(x_ts, float), y_ts

    if task == "AD_periodic":
        df = data.create_periodic_dataset_AD(
            oscillations=structure, sample_size=N_SAMPLES, number_of_features=n_features
        )
    elif task == "AD_multimodal":
        df = data.create_blob_with_outliers(
            n_features=n_features, number_of_blobs=structure,
            sample_size=N_SAMPLES, ratio_outliers=0.2,
        )
    else:
        raise ValueError(f"unknown task: {task}")

    return np.asarray(df.drop(columns=["label"]).values, float), df["label"].values, None, None


#: task -> (structure sweep, structure column name in the committed result CSVs)
TASKS = {
    "Classification_periodic": (OSCILLATIONS, "oscillations"),
    "Classification_multimodal": (N_CLUSTERS, "N_clusters"),
    "AD_periodic": (OSCILLATIONS, "oscillations"),
    "AD_multimodal": (N_CLUSTERS, "N_clusters"),
}


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def default_gamma(X: np.ndarray) -> float:
    """scikit-learn's ``gamma='scale'``: ``1/(D * Var(X))`` over the flattened matrix."""
    return 1.0 / (X.shape[1] * float(X.var()))


def evaluate(y_train, y_test, d2_train, d2_test, gamma: float,
             clf_metric: str = "acc_ts") -> tuple[float, str]:
    """Metric of the classical RBF kernel at one gamma.
    Classification is scored by test accuracy (``acc_ts``) or test weighted F1 (``f1_ts``),
    matching the quantum results' ``f1_ts``."""
    if d2_test is None:  # anomaly detection, transductive
        K = np.exp(-gamma * d2_train)
        model = OneClassSVM(kernel="precomputed", nu=0.2).fit(K)
        preds = ((model.predict(K) - 1) * (-0.5)).astype(int)
        return float(f1_score(y_train, preds, zero_division=0)), "F1"

    model = SVC(kernel="precomputed", random_state=42).fit(np.exp(-gamma * d2_train), y_train)
    preds = model.predict(np.exp(-gamma * d2_test))
    if clf_metric == "f1_ts":
        return float(f1_score(y_test, preds, average="weighted", zero_division=0)), "f1_ts"
    return float(accuracy_score(y_test, preds)), "acc_ts"


def scan(gammas=GAMMAS, verbose: bool = True, clf_metric: str = "acc_ts") -> pd.DataFrame:
    """Evaluate every configuration at every gamma, plus its default gamma."""
    rows = []
    for task, (sweep, _) in TASKS.items():
        for d in DIMENSIONS:
            for s in sweep:
                X_tr, y_tr, X_ts, y_ts = build(task, d, s)
                if verbose:
                    print(f"  {task:28s} D={d} s={s}", flush=True)
                # Squared distances are computed once and reused for every gamma.
                d2_tr = np.maximum(euclidean_distances(X_tr, X_tr, squared=True), 0.0)
                d2_ts = (
                    None if X_ts is None
                    else np.maximum(euclidean_distances(X_ts, X_tr, squared=True), 0.0)
                )
                g_default = default_gamma(X_tr)
                for gamma, is_default in [(g, False) for g in gammas] + [(g_default, True)]:
                    metric, name = evaluate(y_tr, y_ts, d2_tr, d2_ts, gamma, clf_metric)
                    rows.append({
                        "Task": task, "N_qubits": d, "structure": s,
                        "Gamma": float(gamma), "is_default": is_default,
                        "default_gamma": g_default, "metric": metric, "metric_name": name,
                    })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """One row per configuration: the band edges over the five gammas and the default."""
    swept = df[~df.is_default]
    keys = ["Task", "N_qubits", "structure"]
    agg = (
        swept.groupby(keys)["metric"]
        .agg(band_low="min", band_high="max", band_median="median")
        .reset_index()
    )
    best = swept.loc[swept.groupby(keys)["metric"].idxmax()][keys + ["Gamma"]]
    best = best.rename(columns={"Gamma": "best_gamma"})
    dflt = df[df.is_default][keys + ["metric", "Gamma"]].rename(
        columns={"metric": "default_metric", "Gamma": "default_gamma"}
    )
    out = agg.merge(best, on=keys).merge(dflt, on=keys)
    out["headroom"] = out["band_high"] - out["default_metric"]
    return out.sort_values(keys).reset_index(drop=True)


# ---------------------------------------------------------------------------
# cross-check against the results already committed in the repository
# ---------------------------------------------------------------------------


def verify(df: pd.DataFrame) -> int:
    """Compare this scan with ``reference_scan_5gammas.csv``, an earlier independent run of the
    same protocol (test accuracy for classification) at the same five gammas. Every point must
    agree to floating-point noise. Returns 0 on success.
    """
    ok = True

    band_csv = REF_DIR / "reference_scan_5gammas.csv"
    if band_csv.exists():
        ref = pd.read_csv(band_csv)
        keys = ["Task", "N_qubits", "structure", "is_default"]
        mine = df.copy()
        for frame in (ref, mine):
            frame["gkey"] = frame["Gamma"].round(9)
        m = mine.merge(ref, on=keys + ["gkey"], suffixes=("", "_ref"))
        m["diff"] = (m["metric"] - m["metric_ref"]).abs()
        print(f"vs {band_csv.name}: {len(m)} shared points "
              f"({df.shape[0]} computed here), max |diff| = {m['diff'].max():.3e}")
        print(m.groupby("Task")["diff"].agg(["size", "max"]).to_string())
        expected = 48 * (len(GAMMAS) + 1)  # 48 configurations x (5 gammas + default)
        ok &= bool(len(m) == expected and m["diff"].max() < 1e-12)
        if len(m) != expected:
            print(f"  ! expected {expected} shared points, got {len(m)}")
    else:
        print(f"! {band_csv} not found, skipping the cross-check")
        ok = False

    print("\nVERIFY:", "OK" if ok else "MISMATCH")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", action="store_true",
                    help="cross-check the scan against the committed reference table")
    ap.add_argument("--clf-metric", choices=("acc_ts", "f1_ts"), default="acc_ts",
                    help="classification metric; f1_ts writes to *_f1_ts.csv and leaves the default outputs untouched")
    args = ap.parse_args()
    if args.verify and args.clf_metric != "acc_ts":
        ap.error("--verify compares against accuracy references; use it with the default --clf-metric")

    suffix = "" if args.clf_metric == "acc_ts" else f"_{args.clf_metric}"
    scan_csv = SCAN_CSV.with_name(SCAN_CSV.stem + suffix + ".csv")
    summary_csv = SUMMARY_CSV.with_name(SUMMARY_CSV.stem + suffix + ".csv")

    print(f"scanning 48 configurations at gamma in {GAMMAS} + default "
          f"(classification metric: {args.clf_metric})...")
    df = scan(clf_metric=args.clf_metric)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(scan_csv, index=False)
    summary = summarize(df)
    summary.to_csv(summary_csv, index=False)
    print(f"\nwrote {scan_csv.relative_to(REPO)} ({len(df)} rows)")
    print(f"wrote {summary_csv.relative_to(REPO)} ({len(summary)} rows)")

    print("\nheadroom (best of the five gammas - default gamma):")
    print(summary.groupby("Task")["headroom"].agg(["mean", "max"]).round(4).to_string())

    if args.verify:
        print()
        return verify(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
