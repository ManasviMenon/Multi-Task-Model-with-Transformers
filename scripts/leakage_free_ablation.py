"""Leakage-free single split and the section 10 ablation, retrained on lagged features.

leakage_free_retrain.py covers the five walk-forward folds for the 5-head MTL model. This
covers what is left before the accuracy table is clean:

  1. the single 80/20 split, 5-head MTL      (sections 1 and 2)
  2. the single 80/20 split, single-task VaR (section 10, table 1)
  3. all five folds, single-task VaR         (section 10, table 2)

Same clean feature set as leakage_free_retrain: each of the 8 day-t features replaced by its
one-day lag, nothing else touched. The single-task model is the one from
single_task_var_ablation.py - identical transformer body, one Dense(1) VaR head, pinball loss,
same optimiser and early stopping.

TWO CHANGES IN THE SINGLE SPLIT, NOT ONE - stated up front so the comparison is not
oversold. The original single split (verify_loss_bug_impact.build_single_split) reproduces the
notebook's documented quirk of fitting the FEATURE scaler on the whole dataset rather than
train-only. That is a second, separate leak. Since the point here is a clean accuracy table
rather than a controlled one-variable ablation, this fits the feature scaler on the training
portion only, matching what the walk-forward folds already do. So the leakage-free single-split
numbers differ from the originals in two respects - feature timing and scaler scope - both
corrections. The fold numbers, which are the paper's primary evidence, change in one respect
only, since the folds always fit train-only.

PRE-REGISTERED, before running:
  - Single-split MTL accuracy degrades in the same shape as fold5 did: log_return and VaR_5
    much worse, yz_vol and CVaR_5 roughly unchanged or better.
  - The section 10 direction (single-task calibrates better than 5-head MTL) may or may not
    survive. It was always a modest effect concentrated where MTL was already near the
    boundary, and the leakage-free VaR target is materially harder to reconstruct
    (R^2 0.99998 -> 0.5937). A reversal is a real possible outcome and gets reported as one.
  - Calibration verdicts are checked against the model-free raw-target backtest from
    leakage_audit.py. The thesis predicts they should keep agreeing.
"""
import json
import os
import sys

import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from leakage_free_retrain import build_clean_frame, build_fold_arrays
from mtl_var.calibration import fixed_offset_calibration
from mtl_var.data import SEQUENCE_LENGTH, TARGETS, load_and_engineer_features, make_sequences
from mtl_var.model import build_transformer_model, quantile_loss_05, train_model  # noqa: F401
from mtl_var.pipeline import FOLDS
from mtl_var.seeding import set_seeds
from single_task_var_ablation import build_single_task_model, train_single_task

RESULTS_PATH = "results/leakage_free_ablation_results.json"


def build_single_split_clean(data, feature_names):
    """80/20 chronological split with the feature scaler fit on the TRAIN portion only."""
    y_raw = {t: data[t].values[SEQUENCE_LENGTH - 1:] for t in TARGETS}
    n_seq = len(data) - SEQUENCE_LENGTH + 1
    split_index = int(n_seq * 0.8)

    # The train rows are those feeding sequences before split_index; a sequence at index j
    # spans rows j..j+4, so the last training row is split_index + SEQUENCE_LENGTH - 2.
    train_rows = data.iloc[:split_index + SEQUENCE_LENGTH - 1]

    fs = StandardScaler().fit(train_rows[feature_names])
    X_seq = make_sequences(fs.transform(data[feature_names]), SEQUENCE_LENGTH)

    X_train, X_test = X_seq[:split_index], X_seq[split_index:]
    y_train_raw = {t: y_raw[t][:split_index] for t in TARGETS}
    y_test_raw = {t: y_raw[t][split_index:] for t in TARGETS}

    scalers, y_train_z, y_test_z = {}, {}, {}
    for t in TARGETS:
        ts = StandardScaler()
        y_train_z[t] = ts.fit_transform(y_train_raw[t].reshape(-1, 1)).flatten()
        y_test_z[t] = ts.transform(y_test_raw[t].reshape(-1, 1)).flatten()
        scalers[t] = ts

    return X_train, X_test, y_train_z, y_test_z, y_test_raw, scalers


def run_mtl_single_split(data, feature_names):
    print("\n===== single split, 5-head MTL (leakage-free) =====", flush=True)
    tf.keras.backend.clear_session()
    set_seeds()

    X_train, X_test, y_train_z, y_test_z, y_test_raw, scalers = build_single_split_clean(
        data, feature_names)
    print(f"n_train={len(X_train)}  n_test={len(X_test)}  n_features={len(feature_names)}", flush=True)

    model, epochs, best_val = train_model(X_train, y_train_z, X_test, y_test_z, len(feature_names))
    print(f"trained {epochs} epochs, best val_loss={best_val:.6f}", flush=True)

    preds = model.predict(X_test, verbose=0)
    native = {t: scalers[t].inverse_transform(np.asarray(p).reshape(-1, 1)).flatten()
              for t, p in zip(TARGETS, preds)}

    accuracy = {t: {"mae": float(mean_absolute_error(y_test_raw[t], native[t])),
                    "rmse": float(np.sqrt(mean_squared_error(y_test_raw[t], native[t])))}
                for t in TARGETS}
    calib = fixed_offset_calibration(y_test_raw["log_return"], native["VaR_5"])

    return {"case": "single_split_mtl", "epochs_run": epochs, "best_val_loss": best_val,
            "n_train": len(X_train), "n_test": len(X_test),
            "accuracy": accuracy, "var_calibration": calib}


def run_single_task_single_split(data, feature_names):
    print("\n===== single split, single-task VaR only (leakage-free) =====", flush=True)
    X_train, X_test, y_train_z, y_test_z, y_test_raw, scalers = build_single_split_clean(
        data, feature_names)

    model, epochs, best_val = train_single_task(
        X_train, y_train_z["VaR_5"], X_test, y_test_z["VaR_5"], len(feature_names))
    print(f"trained {epochs} epochs, best val_loss={best_val:.6f}", flush=True)

    pred_z = np.asarray(model.predict(X_test, verbose=0)).reshape(-1, 1)
    var_pred = scalers["VaR_5"].inverse_transform(pred_z).flatten()
    calib = fixed_offset_calibration(y_test_raw["log_return"], var_pred)
    mae = float(mean_absolute_error(y_test_raw["VaR_5"], var_pred))

    return {"case": "single_split_single_task", "epochs_run": epochs, "best_val_loss": best_val,
            "var_mae": mae, "var_calibration": calib}


def run_single_task_fold(fold_cfg, data, feature_names):
    print(f"\n===== {fold_cfg['name']}, single-task VaR only (leakage-free) =====", flush=True)
    X_train, X_test, y_train_z, y_test_z, y_test_raw, scalers = build_fold_arrays(
        fold_cfg, data, feature_names)
    print(f"n_train={len(X_train)}  n_test={len(X_test)}", flush=True)

    model, epochs, best_val = train_single_task(
        X_train, y_train_z["VaR_5"], X_test, y_test_z["VaR_5"], len(feature_names))
    print(f"trained {epochs} epochs, best val_loss={best_val:.6f}", flush=True)

    pred_z = np.asarray(model.predict(X_test, verbose=0)).reshape(-1, 1)
    var_pred = scalers["VaR_5"].inverse_transform(pred_z).flatten()
    calib = fixed_offset_calibration(y_test_raw["log_return"], var_pred)
    mae = float(mean_absolute_error(y_test_raw["VaR_5"], var_pred))

    return {"case": f"single_task_{fold_cfg['name']}", "fold": fold_cfg["name"],
            "epochs_run": epochs, "best_val_loss": best_val,
            "var_mae": mae, "var_calibration": calib}


def main():
    print(__doc__, flush=True)
    only = set(sys.argv[1:])

    data = load_and_engineer_features("SPX500-18Y_scaled_y.csv")
    clean, names = build_clean_frame(data)
    print(f"clean frame: {len(clean)} rows, {len(names)} features\n", flush=True)

    results = []
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            results = json.load(f)
        print(f"resuming: {sorted(r['case'] for r in results)} already done", flush=True)
    done = {r["case"] for r in results}

    def maybe(case, fn):
        if case in done or (only and case not in only):
            return
        res = fn()
        results.append(res)
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)
        cc = res["var_calibration"]["calibrated"]
        p = cc["kupiec_p"]
        verdict = "PASS" if (p is not None and not np.isnan(p) and p >= 0.05) else "FAIL"
        print(f"  -> {case}: cal coverage={cc['coverage']*100:.2f}%  p={p}  {verdict}", flush=True)

    maybe("single_split_mtl", lambda: run_mtl_single_split(clean, names))
    maybe("single_split_single_task", lambda: run_single_task_single_split(clean, names))
    for fold_cfg in FOLDS:
        maybe(f"single_task_{fold_cfg['name']}",
              lambda fc=fold_cfg: run_single_task_fold(fc, clean, names))

    print(f"\nwrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
