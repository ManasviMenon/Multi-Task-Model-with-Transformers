"""
Single 80/20 split, 5-head MTL, leakage-free features, retrained with early stopping on the last 15%
of the training sequences (RESULTS.md section 15). The section 13 single-split model
(leakage_free_ablation.run_mtl_single_split) used the test portion for early stopping. This is the
only remaining reported neural model that had not been retrained. It feeds the "Single split" column
of the accuracy figure (fig07).

Identical to run_mtl_single_split apart from the early-stopping data: same build_single_split_clean
(feature scaler fit on the train portion), architecture, losses, weights, optimiser and seed 42. Also
records the naive return baselines (predict zero, persistence) on this exact test window.
Output: results/cleanval_single_split_results.json.

`--single-task` instead retrains the single-task VaR head (pinball q=0.05 on the VaR_5 proxy; the
ablation's single-split cell) with the same split and clean early stopping, and writes
results/cleanval_single_split_single_task_results.json.
"""
import sys
import json

import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error

from leakage_free_ablation import build_single_split_clean
from leakage_free_retrain import build_clean_frame
from mtl_var.backtest import full_backtest
from mtl_var.calibration import fixed_offset_calibration
from mtl_var.data import SEQUENCE_LENGTH, TARGETS, load_and_engineer_features
from mtl_var.model import fit_with_train_tail_validation, train_single_task
from mtl_var.seeding import set_seeds

OUT = "results/cleanval_single_split_results.json"


def main():
    data = load_and_engineer_features("SPX500-18Y_scaled_y.csv")
    clean, names = build_clean_frame(data)
    tf.keras.backend.clear_session()
    set_seeds()
    X_train, X_test, y_train_z, _unused, y_test_raw, scalers = build_single_split_clean(clean, names)
    model, epochs, best_val, n_fit, n_val = fit_with_train_tail_validation(X_train, y_train_z, len(names))
    print(f"trained {epochs} epochs on {n_fit}, best val_loss={best_val:.6f} on last {n_val}", flush=True)

    preds = model.predict(X_test, verbose=0)
    native = {t: scalers[t].inverse_transform(np.asarray(p).reshape(-1, 1)).flatten() for t, p in zip(TARGETS, preds)}
    acc = {t: {"mae": float(mean_absolute_error(y_test_raw[t], native[t])),
               "rmse": float(np.sqrt(mean_squared_error(y_test_raw[t], native[t])))} for t in TARGETS}

    r = y_test_raw["log_return"]
    # previous day's return for every test row: the row before each target row in the clean frame
    n_seq = len(clean) - SEQUENCE_LENGTH + 1
    split = int(n_seq * 0.8)
    rows = np.arange(split, n_seq) + SEQUENCE_LENGTH - 1
    prev = clean["log_return"].values[rows - 1]
    assert np.array_equal(clean["log_return"].values[rows], r)
    naive = {"predict_zero": float(np.abs(r).mean()), "persistence": float(np.abs(r - prev).mean())}

    calib = fixed_offset_calibration(r, native["VaR_5"])
    cn = calib["calib_n"]
    bt = full_backtest(r[cn:], native["VaR_5"][cn:] + calib["offset"])
    dates = clean["observation_date"].values[rows]
    res = {"case": "single_split_mtl_cleanval", "epochs_run": epochs, "best_val_loss": best_val,
           "n_fit": n_fit, "n_val": n_val, "n_test": int(len(X_test)),
           "test_window": [str(dates[0])[:10], str(dates[-1])[:10]],
           "accuracy": acc, "naive_baselines": naive, "var_calibration": calib, "frozen_backtest": bt}
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({"accuracy": acc, "naive": naive, "frozen": [bt["N"], bt["T"], bt["p_pof"]]}, indent=1))


def main_single_task():
    data = load_and_engineer_features("SPX500-18Y_scaled_y.csv")
    clean, names = build_clean_frame(data)
    X_train, X_test, y_train_z, _unused, y_test_raw, scalers = build_single_split_clean(clean, names)
    model, epochs, best_val, n_fit, n_val = fit_with_train_tail_validation(
        X_train, y_train_z["VaR_5"], len(names), trainer=train_single_task)
    print(f"trained {epochs} epochs on {n_fit}, best val_loss={best_val:.6f} on last {n_val}", flush=True)
    pred_z = np.asarray(model.predict(X_test, verbose=0)).reshape(-1, 1)
    var = scalers["VaR_5"].inverse_transform(pred_z).flatten()
    r = y_test_raw["log_return"]
    calib = fixed_offset_calibration(r, var)
    cn = calib["calib_n"]
    bt = full_backtest(r[cn:], var[cn:] + calib["offset"])
    assert bt["N"] == calib["calibrated"]["breaches"]
    res = {"case": "single_split_single_task_cleanval", "epochs_run": epochs, "best_val_loss": best_val,
           "n_fit": n_fit, "n_val": n_val, "n_test": int(len(X_test)),
           "var_mae": float(mean_absolute_error(y_test_raw["VaR_5"], var)),
           "var_calibration": calib, "frozen_backtest": bt}
    with open("results/cleanval_single_split_single_task_results.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"frozen {bt['N']}/{bt['T']} p_pof={bt['p_pof']:.4g} p_cc={bt['p_cc']:.4g}", flush=True)


if __name__ == "__main__":
    main_single_task() if "--single-task" in sys.argv else main()
