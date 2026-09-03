"""
Ablation: does the 5-head multi-task structure help, hurt, or leave VaR calibration
unchanged, versus a single-task VaR-only model? Everything else held identical - same
transformer body, same features, same target, same loss, same optimizer/schedule, same
calibration procedure. Only the head count changes.

Runs on the CORRECTED loss (the shape-squeeze fix + shape assertions from RESULTS.md
section 0) - both the MTL reference numbers below and the single-task model built here use
the fixed weighted_loss / quantile_loss_05. Comparing a fixed single-task model against a
buggy MTL baseline (or vice versa) would confound head-count with loss-correctness; that
risk no longer applies since MTL was already re-run on the corrected loss.

Two comparisons:
  1. Single 80/20 split (matches MTL_Improved.ipynb's methodology exactly). MTL's corrected
     numbers are already established (verify_loss_bug_impact.py / sanity_check_patched_loss.py,
     recorded in RESULTS.md section 2): uncal 48.20% -> cal 2.76%, Kupiec p=0.015156.
  2. Walk-forward, all 5 folds - reuses walk_forward.py's FOLDS, feature/target scaling
     pattern, and kupiec_from_counts UNCHANGED. MTL's corrected numbers are read directly
     from walk_forward_results.json (var_calibration.calibrated) rather than hardcoded, to
     avoid a transcription error confounding the comparison.

Does NOT modify MTL_Improved.ipynb or walk_forward.py. Writes a new JSON only
(single_task_ablation_results.json).

--------------------------------------------------------------------------------------
PRE-REGISTERED INTERPRETATION (written before any results were computed or looked at):

  - Single-task VaR calibrates BETTER than MTL -> the multi-task burden costs VaR
    calibration; the other four heads compete for capacity/gradient signal that would
    otherwise go toward getting VaR right.

  - SAME (within run-to-run noise) -> MTL is neutral for calibration; the joint
    representation neither helps nor hurts VaR specifically.

  - Single-task VaR calibrates WORSE than MTL -> the shared multi-task representation
    benefits VaR calibration - e.g. by regularizing against overfitting to VaR's own noisy
    tail signal, or by the other heads supplying related information VaR alone can't
    extract from 18 features and 5-day sequences.

  All three are valid findings. Training has run-to-run variance even with fixed seeds
  (CPU multi-threading order effects survive tf.config.experimental.enable_op_determinism()
  in practice, as already observed between runs in this investigation) - sub-1% coverage
  differences on the single split alone are not over-interpreted as a discovery. Only a
  difference consistent in DIRECTION across the single split AND a majority of the 5
  walk-forward folds is reported as a directional finding.
--------------------------------------------------------------------------------------
"""
import json
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from walk_forward import (
    FEATURES, SEQUENCE_LENGTH, FOLDS,
    quantile_loss_05, transformer_block,
    load_and_engineer_features, kupiec_from_counts, make_sequences, set_seeds,
)
from verify_loss_bug_impact import build_single_split

CSV_PATH = "SPX500-18Y_scaled_y.csv"
RESULTS_PATH = "results/single_task_ablation_results.json"

# Verified single-split MTL reference (RESULTS.md section 2 / verify_loss_bug_impact.py,
# corrected loss, reproduced exactly twice already - not re-run here to avoid a third
# ~20min training pass of a number already pinned down to 5 decimal places).
MTL_SINGLE_SPLIT = {"uncal_coverage": 0.4820, "cal_coverage": 0.0276, "cal_kupiec_p": 0.015156}


def build_single_task_model(seq_len, n_features):
    inp = Input(shape=(seq_len, n_features))
    x = layers.Dense(64, activation=None)(inp)
    for _ in range(4):
        x = transformer_block(x)  # identical body to the MTL model: 4 heads, key_dim=ff_dim=128
    x = layers.GlobalAveragePooling1D()(x)
    out_var = layers.Dense(1, name="VaR_5")(x)
    return Model(inp, out_var)


def train_single_task(X_train, y_train_z, X_test, y_test_z, n_features,
                       batch_size=32, max_epochs=50, patience=5):
    tf.keras.backend.clear_session()
    set_seeds()

    model = build_single_task_model(SEQUENCE_LENGTH, n_features)
    optimizer = tf.keras.optimizers.Adam(1e-4)

    train_ds = tf.data.Dataset.from_tensor_slices(
        (X_train, tf.cast(y_train_z, tf.float32))
    ).batch(batch_size)
    val_ds = tf.data.Dataset.from_tensor_slices(
        (X_test, tf.cast(y_test_z, tf.float32))
    ).batch(batch_size)

    best_val_loss = float('inf')
    wait = 0
    best_weights = None
    epochs_run = 0

    for epoch in range(max_epochs):
        for xb, yb in train_ds:
            with tf.GradientTape() as tape:
                pred = tf.squeeze(model(xb, training=True), axis=-1)
                loss = quantile_loss_05(yb, pred)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        val_losses = []
        for xb, yb in val_ds:
            pred = tf.squeeze(model(xb, training=False), axis=-1)
            val_losses.append(float(quantile_loss_05(yb, pred)))
        val_loss = float(np.mean(val_losses))
        epochs_run = epoch + 1

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = model.get_weights()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.set_weights(best_weights)
    return model, epochs_run, best_val_loss


def calibrate_and_evaluate(returns, var_pred, calib_n=None):
    """Identical procedure to walk_forward.py's run_fold / the notebook's calibration cell:
    split in half chronologically (or at a given calib_n), fit a 5th-percentile-of-residuals
    offset on the first half, Kupiec-test the untouched second half, before and after."""
    if calib_n is None:
        calib_n = len(var_pred) // 2

    returns_calib, returns_eval = returns[:calib_n], returns[calib_n:]
    var_calib, var_eval = var_pred[:calib_n], var_pred[calib_n:]
    offset = float(np.quantile(returns_calib - var_calib, 0.05))
    var_eval_cal = var_eval + offset

    def kupiec_eval(returns_e, var_e):
        T = len(var_e)
        N = int(np.sum(returns_e < var_e))
        return N, T, kupiec_from_counts(N, T)

    N0, T0, p0 = kupiec_eval(returns_eval, var_eval)
    N1, T1, p1 = kupiec_eval(returns_eval, var_eval_cal)
    return {
        "calib_n": calib_n, "eval_n": T0, "offset": offset,
        "uncalibrated": {"breaches": N0, "coverage": N0 / T0, "kupiec_p": p0},
        "calibrated": {"breaches": N1, "coverage": N1 / T1, "kupiec_p": p1},
    }


def run_single_split():
    print("=== Single 80/20 split ===")
    X_train, X_test, y_train_raw, y_test_raw, y_train_z, y_test_z, target_scalers = build_single_split()
    n_features = len(FEATURES)
    print(f"n_train={len(X_train)}  n_test={len(X_test)}")

    model, epochs_run, best_val_loss = train_single_task(
        X_train, y_train_z["VaR_5"], X_test, y_test_z["VaR_5"], n_features
    )
    print(f"  trained {epochs_run} epochs, best val_loss={best_val_loss:.6f}")

    pred = model.predict(X_test, verbose=0)
    var_pred = target_scalers["VaR_5"].inverse_transform(pred.reshape(-1, 1)).flatten()
    mae = float(mean_absolute_error(y_test_raw["VaR_5"], var_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test_raw["VaR_5"], var_pred)))

    result = calibrate_and_evaluate(y_test_raw["log_return"], var_pred)
    result["mae"] = mae
    result["rmse"] = rmse
    result["epochs_run"] = epochs_run
    return result


def build_fold_data(fold_cfg, data):
    """Mirrors walk_forward.py's run_fold data-prep exactly (per-fold feature/target
    scalers fit on train only), duplicated here rather than imported since run_fold isn't
    factored into a separate data-prep function and this task is not to modify that file."""
    dates = data['observation_date']
    train_mask = (dates >= fold_cfg['train_start']) & (dates <= fold_cfg['train_end'])
    test_mask = (dates >= fold_cfg['test_start']) & (dates <= fold_cfg['test_end'])
    train_df = data.loc[train_mask].reset_index(drop=True)
    test_df = data.loc[test_mask].reset_index(drop=True)

    feat_scaler = StandardScaler()
    X_train_scaled = feat_scaler.fit_transform(train_df[FEATURES])
    X_test_scaled = feat_scaler.transform(test_df[FEATURES])
    X_train = make_sequences(X_train_scaled, SEQUENCE_LENGTH)
    X_test = make_sequences(X_test_scaled, SEQUENCE_LENGTH)

    y_train_var_raw = train_df["VaR_5"].values[SEQUENCE_LENGTH - 1:]
    y_test_var_raw = test_df["VaR_5"].values[SEQUENCE_LENGTH - 1:]
    y_test_return_raw = test_df["log_return"].values[SEQUENCE_LENGTH - 1:]

    ts = StandardScaler()
    y_train_var_z = ts.fit_transform(y_train_var_raw.reshape(-1, 1)).flatten()
    y_test_var_z = ts.transform(y_test_var_raw.reshape(-1, 1)).flatten()

    return X_train, X_test, y_train_var_z, y_test_var_z, y_test_var_raw, y_test_return_raw, ts


def run_walk_forward():
    print("\n=== Walk-forward, 5 folds ===")
    data = load_and_engineer_features(CSV_PATH)

    results = {}
    for fold_cfg in FOLDS:
        fold = fold_cfg["name"]
        print(f"\n--- {fold} ---")
        (X_train, X_test, y_train_var_z, y_test_var_z,
         y_test_var_raw, y_test_return_raw, ts) = build_fold_data(fold_cfg, data)
        n_features = len(FEATURES)
        print(f"n_train={len(X_train)}  n_test={len(X_test)}")

        model, epochs_run, best_val_loss = train_single_task(
            X_train, y_train_var_z, X_test, y_test_var_z, n_features
        )
        print(f"  trained {epochs_run} epochs, best val_loss={best_val_loss:.6f}")

        pred = model.predict(X_test, verbose=0)
        var_pred = ts.inverse_transform(pred.reshape(-1, 1)).flatten()
        mae = float(mean_absolute_error(y_test_var_raw, var_pred))

        result = calibrate_and_evaluate(y_test_return_raw, var_pred)
        result["mae"] = mae
        result["epochs_run"] = epochs_run
        results[fold] = result
        print(f"  uncal={result['uncalibrated']['coverage']*100:.2f}% "
              f"cal={result['calibrated']['coverage']*100:.2f}% "
              f"p={result['calibrated']['kupiec_p']}")

    return results


def main():
    print(__doc__)

    single_split_result = run_single_split()
    walk_forward_results = run_walk_forward()

    with open(RESULTS_PATH, "w") as f:
        json.dump({"single_split": single_split_result, "walk_forward": walk_forward_results}, f, indent=2)

    # --- Table 1: single split ---
    print("\n" + "=" * 90)
    print("TABLE 1: Single 80/20 split — MTL (5-head) vs single-task (VaR-only)")
    print(f"{'':16s} {'uncal cov':>10s} {'cal cov':>10s} {'cal Kupiec p':>14s}")
    print(f"{'MTL (ref.)':16s} {MTL_SINGLE_SPLIT['uncal_coverage']*100:>9.2f}% "
          f"{MTL_SINGLE_SPLIT['cal_coverage']*100:>9.2f}% {MTL_SINGLE_SPLIT['cal_kupiec_p']:>14.5g}")
    ss = single_split_result
    print(f"{'Single-task':16s} {ss['uncalibrated']['coverage']*100:>9.2f}% "
          f"{ss['calibrated']['coverage']*100:>9.2f}% {ss['calibrated']['kupiec_p']:>14.5g}")

    # --- Table 2: walk-forward ---
    print("\n" + "=" * 90)
    print("TABLE 2: Walk-forward, 5 folds — MTL vs single-task (calibrated coverage / Kupiec p)")
    mtl_wf = json.load(open("results/walk_forward_results.json"))
    mtl_by_fold = {r["fold"]: r["var_calibration"]["calibrated"] for r in mtl_wf}

    print(f"{'fold':22s} {'MTL cov':>10s} {'MTL p':>12s} {'Single cov':>12s} {'Single p':>12s}  direction")
    directions = []
    for fold_cfg in FOLDS:
        fold = fold_cfg["name"]
        mtl = mtl_by_fold[fold]
        st = walk_forward_results[fold]["calibrated"]
        mtl_p = mtl["kupiec_p"] if mtl["kupiec_p"] is not None else float("nan")
        st_p = st["kupiec_p"] if st["kupiec_p"] is not None else float("nan")
        # "better calibrated" = coverage closer to the 5% target
        mtl_dist = abs(mtl["coverage"] - 0.05)
        st_dist = abs(st["coverage"] - 0.05)
        direction = "single BETTER" if st_dist < mtl_dist else ("single WORSE" if st_dist > mtl_dist else "same")
        directions.append(direction)
        print(f"{fold:22s} {mtl['coverage']*100:>9.2f}% {mtl_p:>12.4g} "
              f"{st['coverage']*100:>11.2f}% {st_p:>12.4g}  {direction}")

    # --- Pre-registered read ---
    ss_dist_mtl = abs(MTL_SINGLE_SPLIT['cal_coverage'] - 0.05)
    ss_dist_single = abs(ss['calibrated']['coverage'] - 0.05)
    single_split_direction = ("single BETTER" if ss_dist_single < ss_dist_mtl
                               else ("single WORSE" if ss_dist_single > ss_dist_mtl else "same"))

    print("\n" + "=" * 90)
    print("PRE-REGISTERED READ")
    print(f"Single split direction: {single_split_direction}")
    print(f"Walk-forward directions by fold: {directions}")
    n_better = directions.count("single BETTER")
    n_worse = directions.count("single WORSE")
    n_same = directions.count("same")
    print(f"Walk-forward: {n_better} better, {n_worse} worse, {n_same} same (of 5 folds)")
    majority = "BETTER" if n_better > 2 else ("WORSE" if n_worse > 2 else "no clear majority")
    consistent = (single_split_direction.startswith("single BETTER") and majority == "BETTER") or \
                 (single_split_direction.startswith("single WORSE") and majority == "WORSE")
    print(f"Walk-forward majority direction: {majority}")
    print(f"Consistent with single-split direction: {consistent}")


if __name__ == "__main__":
    main()
