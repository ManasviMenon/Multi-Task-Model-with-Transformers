"""What does removing the leakage actually cost? Retrain a fold with all day-t features lagged.

leakage_audit.py established that 8 of 18 features embed day-t information, that
`log_close_ret` IS the `log_return` target, and that the calibration finding reproduces
without any model at all. This measures the other half of the question: how much of the
reported accuracy was real.

The leakage-free variant changes exactly one thing - information timing. Same architecture,
same losses, same equal 0.2 weights, same optimiser and early stopping, same folds, same
calibration protocol, same seed. Each contaminated feature is replaced by its one-day lag, so
the model sees only what was knowable before day t while still predicting day t.

(One redundancy is accepted rather than papered over: log_close_ret lagged one day is exactly
ret_lag1, which is already in the set. Keeping it preserves the "18 features, shifted" framing
and a duplicate column is harmless to the model.)

PRE-REGISTERED EXPECTATIONS, written before running:
  - log_return MAE degrades badly. Its linear R^2 collapses from 1.0000 to 0.0190 once the
    day-t features are removed, so essentially all apparent return skill was the target being
    handed to the model as an input. Expect the leaky 0.00386 to move toward the naive level.
  - yz_vol MAE barely moves. Its R^2 only falls 0.9960 -> 0.9937: volatility is persistent and
    yz_vol_lag1 already carries almost all of it. This head was never leakage-dependent.
  - VaR_5 MAE degrades moderately. R^2 falls 0.99998 -> 0.5937, so roughly half the target is
    genuinely available from lagged information.
  - The Kupiec verdict is genuinely uncertain and is NOT the success criterion here. This
    measures the accuracy cost; the calibration thesis was already shown to be
    model-independent, so a verdict change either way would not overturn it.

Writes a new JSON. Does not modify any existing result, script, or the notebook.
"""
import json
import os
import sys

import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from mtl_var.calibration import fixed_offset_calibration
from mtl_var.data import FEATURES, SEQUENCE_LENGTH, TARGETS, load_and_engineer_features, make_sequences
from mtl_var.model import train_model
from mtl_var.pipeline import FOLDS
from mtl_var.seeding import set_seeds

RESULTS_PATH = "results/leakage_free_retrain_results.json"
WALK_FORWARD_PATH = "results/walk_forward_results.json"

CONTAMINATED = ["log_open_ret", "log_high_ret", "log_low_ret", "log_close_ret",
                "drawdown", "ret_min20", "ret_max20", "ret_skew20"]


def build_clean_frame(data):
    """Replace each day-t feature with its one-day lag. Returns (frame, feature_names)."""
    out = data.copy()
    names = []
    for f in FEATURES:
        if f in CONTAMINATED:
            lagged = f"{f}_lag1"
            out[lagged] = out[f].shift(1)
            names.append(lagged)
        else:
            names.append(f)
    return out.dropna().reset_index(drop=True), names


def build_fold_arrays(fold_cfg, data, feature_names):
    dates = data["observation_date"]
    train = data.loc[(dates >= fold_cfg["train_start"]) & (dates <= fold_cfg["train_end"])].reset_index(drop=True)
    test = data.loc[(dates >= fold_cfg["test_start"]) & (dates <= fold_cfg["test_end"])].reset_index(drop=True)

    fs = StandardScaler()
    X_train = make_sequences(fs.fit_transform(train[feature_names]), SEQUENCE_LENGTH)
    X_test = make_sequences(fs.transform(test[feature_names]), SEQUENCE_LENGTH)

    y_train_raw = {t: train[t].values[SEQUENCE_LENGTH - 1:] for t in TARGETS}
    y_test_raw = {t: test[t].values[SEQUENCE_LENGTH - 1:] for t in TARGETS}

    scalers, y_train_z, y_test_z = {}, {}, {}
    for t in TARGETS:
        ts = StandardScaler()
        y_train_z[t] = ts.fit_transform(y_train_raw[t].reshape(-1, 1)).flatten()
        y_test_z[t] = ts.transform(y_test_raw[t].reshape(-1, 1)).flatten()
        scalers[t] = ts

    return X_train, X_test, y_train_z, y_test_z, y_test_raw, scalers


def run_fold_clean(fold_cfg, data, feature_names, seed=42):
    print(f"\n===== {fold_cfg['name']} (leakage-free, {len(feature_names)} lagged features) =====", flush=True)
    tf.keras.backend.clear_session()
    set_seeds(seed)

    X_train, X_test, y_train_z, y_test_z, y_test_raw, scalers = build_fold_arrays(
        fold_cfg, data, feature_names)
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

    return {"fold": fold_cfg["name"], "seed": seed, "n_features": len(feature_names),
            "epochs_run": epochs, "best_val_loss": best_val,
            "accuracy": accuracy, "var_calibration": calib}


def main():
    folds = sys.argv[1:] or ["fold5_2022_bear"]
    print(__doc__, flush=True)

    data = load_and_engineer_features("SPX500-18Y_scaled_y.csv")
    clean, names = build_clean_frame(data)
    print(f"rows: {len(data)} -> {len(clean)} after lagging (one row lost to the shift)")
    print(f"lagged features: {', '.join(n for n in names if n.endswith('_lag1') and n not in FEATURES)}\n")

    with open(WALK_FORWARD_PATH) as f:
        leaky = {r["fold"]: r for r in json.load(f) if "error" not in r}

    # Resume rather than overwrite: these are multi-hour runs and fold5 was done separately.
    results = []
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            results = json.load(f)
        print(f"resuming: {sorted(r['fold'] for r in results)} already done")

    done = {r["fold"] for r in results}
    for fold_cfg in FOLDS:
        if fold_cfg["name"] not in folds or fold_cfg["name"] in done:
            continue
        res = run_fold_clean(fold_cfg, clean, names)
        results.append(res)

        ref = leaky.get(res["fold"], {})
        print(f"\n--- {res['fold']}: leaky (day-t features) vs leakage-free (all lagged) ---")
        print(f"  {'task':12s} {'leaky MAE':>14s} {'clean MAE':>14s} {'ratio':>9s}")
        for t in TARGETS:
            a = ref.get("accuracy", {}).get(t, {}).get("mae")
            b = res["accuracy"][t]["mae"]
            ratio = (b / a) if a else float("nan")
            print(f"  {t:12s} {a:14.6f} {b:14.6f} {ratio:8.2f}x")

        rc = ref.get("var_calibration", {}).get("calibrated", {})
        cc = res["var_calibration"]["calibrated"]
        verdict = lambda p: "PASS" if (p is not None and not np.isnan(p) and p >= 0.05) else "FAIL"  # noqa: E731
        print(f"\n  VaR calibration   leaky: {(rc.get('coverage') or 0)*100:.2f}% "
              f"p={rc.get('kupiec_p')} {verdict(rc.get('kupiec_p'))}")
        print(f"                    clean: {cc['coverage']*100:.2f}% "
              f"p={cc['kupiec_p']} {verdict(cc['kupiec_p'])}")

        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nwrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
