"""Walk-forward validation across five regime-aligned folds.

Fold orchestration moved from scripts/walk_forward.py. The per-fold work is unchanged: fresh
scalers fit on that fold's train rows only, a freshly seeded model trained from scratch, then
the test window split in half - first half fits the VaR offset, second half reports coverage
and Kupiec.

Fold design (expanding train window, non-overlapping 2-3yr test blocks covering the regimes
that actually matter: 2008 GFC, 2011-13 EU debt crisis, 2015-17 China slowdown + Aug'15
selloff, 2019-20 COVID crash, 2022 rate-hike bear market):

  fold 1: train 2004-2007        test 2008-2009
  fold 2: train 2004-2010        test 2011-2013
  fold 3: train 2004-2014        test 2015-2017
  fold 4: train 2004-2018        test 2019-2020
  fold 5: train 2004-2021        test 2022

Two feature sets (data.FEATURE_SETS). "original" is the default and reproduces every recorded
result; it contains target leakage (RESULTS.md section 13). "leakage_free" lags every day-t
feature by one day. Each set writes its own results file and prediction cache, so a
leakage-free run can never overwrite the recorded ones.

Results are written after every fold, so a crash partway through doesn't lose completed folds.
"""
import gc
import json
import os
import traceback

import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from . import tracking
from .calibration import fixed_offset_calibration, rolling_calibration_analysis
from .data import (
    DEFAULT_CSV,
    FEATURE_SETS,
    FEATURES,
    SEQUENCE_LENGTH,
    TARGETS,
    frame_for_feature_set,
    load_and_engineer_features,
    make_sequences,
)
from .model import WEIGHTS, train_model
from .seeding import SEED, set_seeds

RESULTS_PATH = "results/walk_forward_results.json"
PREDICTIONS_DIR = "results/predictions"

FOLDS = [
    {"name": "fold1_gfc",        "train_start": "2004-01-01", "train_end": "2007-12-31", "test_start": "2008-01-01", "test_end": "2009-12-31"},
    {"name": "fold2_eu_crisis",  "train_start": "2004-01-01", "train_end": "2010-12-31", "test_start": "2011-01-01", "test_end": "2013-12-31"},
    {"name": "fold3_2015_selloff","train_start": "2004-01-01", "train_end": "2014-12-31", "test_start": "2015-01-01", "test_end": "2017-12-31"},
    {"name": "fold4_covid",      "train_start": "2004-01-01", "train_end": "2018-12-31", "test_start": "2019-01-01", "test_end": "2020-12-31"},
    {"name": "fold5_2022_bear",  "train_start": "2004-01-01", "train_end": "2021-12-31", "test_start": "2022-01-01", "test_end": "2022-12-31"},
]


def results_path(feature_set="original"):
    """Original keeps its historical path; every other feature set gets its own file."""
    return RESULTS_PATH if feature_set == "original" else f"results/walk_forward_results_{feature_set}.json"


def predictions_dir(feature_set="original"):
    return PREDICTIONS_DIR if feature_set == "original" else f"results/predictions_{feature_set}"


def build_fold_data(fold_cfg, data, feature_names=FEATURES):
    """Per-fold split, feature scaling, sequencing, and target standardisation - scalers fit
    on that fold's train rows only."""
    dates = data['observation_date']
    train_mask = (dates >= fold_cfg['train_start']) & (dates <= fold_cfg['train_end'])
    test_mask = (dates >= fold_cfg['test_start']) & (dates <= fold_cfg['test_end'])

    train_df = data.loc[train_mask].reset_index(drop=True)
    test_df = data.loc[test_mask].reset_index(drop=True)

    feat_scaler = StandardScaler()
    X_train_scaled = feat_scaler.fit_transform(train_df[feature_names])
    X_test_scaled = feat_scaler.transform(test_df[feature_names])

    X_train = make_sequences(X_train_scaled, SEQUENCE_LENGTH)
    X_test = make_sequences(X_test_scaled, SEQUENCE_LENGTH)

    y_train_raw = {t: train_df[t].values[SEQUENCE_LENGTH - 1:] for t in TARGETS}
    y_test_raw = {t: test_df[t].values[SEQUENCE_LENGTH - 1:] for t in TARGETS}

    target_scalers = {}
    y_train_z, y_test_z = {}, {}
    for t in TARGETS:
        ts = StandardScaler()
        y_train_z[t] = ts.fit_transform(y_train_raw[t].reshape(-1, 1)).flatten()
        y_test_z[t] = ts.transform(y_test_raw[t].reshape(-1, 1)).flatten()
        target_scalers[t] = ts

    return {
        "train_df": train_df, "test_df": test_df,
        "X_train": X_train, "X_test": X_test,
        "y_train_z": y_train_z, "y_test_z": y_test_z,
        "y_train_raw": y_train_raw, "y_test_raw": y_test_raw,
        "target_scalers": target_scalers,
        "feat_scaler": feat_scaler,
    }


def run_fold(fold_cfg, data, seed=SEED, feature_set="original", save_predictions=True):
    """Train and evaluate one fold. Emits one MLflow run per (fold, seed) when tracking is on.

    Tracking calls are deliberately kept off the seeding/training boundary: params go in after
    the (RNG-free) data prep, metrics and the model artifact only after training has finished.
    Verified inert - fold5 reproduces its recorded metrics bit-for-bit with tracking enabled.
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unknown feature set {feature_set!r}; expected one of {sorted(FEATURE_SETS)}")

    run_name = f"{fold_cfg['name']}_seed{seed}" + ("" if feature_set == "original" else f"_{feature_set}")
    with tracking.start_run(
        run_name=run_name,
        tags={"fold": fold_cfg["name"], "seed": seed, "stage": "walk-forward", "feature_set": feature_set},
    ):
        return _run_fold_impl(fold_cfg, data, seed, feature_set, save_predictions)


def _run_fold_impl(fold_cfg, data, seed, feature_set="original", save_predictions=True):
    feature_names = FEATURE_SETS[feature_set]
    data = frame_for_feature_set(data, feature_set)

    label = "" if feature_set == "original" else f"  [{feature_set}]"
    print(f"\n===== {fold_cfg['name']}  train[{fold_cfg['train_start']}:{fold_cfg['train_end']}]  "
          f"test[{fold_cfg['test_start']}:{fold_cfg['test_end']}]{label} =====", flush=True)

    tf.keras.backend.clear_session()
    gc.collect()
    set_seeds(seed)

    fd = build_fold_data(fold_cfg, data, feature_names)
    X_train, X_test = fd["X_train"], fd["X_test"]
    y_train_z, y_test_z = fd["y_train_z"], fd["y_test_z"]
    y_test_raw, target_scalers = fd["y_test_raw"], fd["target_scalers"]
    test_df = fd["test_df"]

    n_features = len(feature_names)
    print(f"n_train={len(X_train)}  n_test={len(X_test)}  n_features={n_features}", flush=True)

    tracking.log_params(tracking.fold_params(
        fold_cfg, seed, len(X_train), len(X_test), n_features, SEQUENCE_LENGTH, TARGETS, WEIGHTS,
    ))
    tracking.log_params({"feature_set": feature_set})
    model, epochs_run, best_val_loss = train_model(X_train, y_train_z, X_test, y_test_z, n_features)
    print(f"trained {epochs_run} epochs, best val_loss={best_val_loss:.6f}", flush=True)

    preds = model.predict(X_test, verbose=0)
    native_preds = {}
    for t, p in zip(TARGETS, preds):
        native_preds[t] = target_scalers[t].inverse_transform(p.reshape(-1, 1)).flatten()

    def rmse(a, b):
        return float(np.sqrt(mean_squared_error(a, b)))

    accuracy = {}
    for t in TARGETS:
        y_true = y_test_raw[t]
        y_pred = native_preds[t]
        accuracy[t] = {"mae": float(mean_absolute_error(y_true, y_pred)), "rmse": rmse(y_true, y_pred)}

    returns = y_test_raw['log_return']
    var_pred = native_preds['VaR_5']

    var_calibration = fixed_offset_calibration(returns, var_pred)
    calib_n = var_calibration["calib_n"]
    T0 = var_calibration["eval_n"]
    N0 = var_calibration["uncalibrated"]["breaches"]
    p0 = var_calibration["uncalibrated"]["kupiec_p"]
    N1 = var_calibration["calibrated"]["breaches"]
    p1 = var_calibration["calibrated"]["kupiec_p"]
    T1 = T0

    test_dates = test_df['observation_date'].values[SEQUENCE_LENGTH - 1:]

    # A one-off training run (mtl-var train) must not overwrite the committed prediction caches the
    # research results were computed from.
    if save_predictions:
        pred_dir = predictions_dir(feature_set)
        os.makedirs(pred_dir, exist_ok=True)
        np.savez(
            os.path.join(pred_dir, f"{fold_cfg['name']}.npz"),
            dates=test_dates.astype("datetime64[D]").astype(str),
            returns=returns,
            var_pred=var_pred,
            calib_n=calib_n,
        )

    rolling = rolling_calibration_analysis(test_dates, returns, var_pred, eval_start_idx=calib_n,
                                            trigger_on_elevated=False)
    rolling_triggered = rolling_calibration_analysis(test_dates, returns, var_pred, eval_start_idx=calib_n,
                                                       trigger_on_elevated=True)
    if rolling is not None:
        rc = rolling["eval_slice_only"]
        print(f"Rolling scheduled-only (eval slice) coverage={rc['coverage']:.4f} (kupiec p={rc['kupiec_p']})  "
              f"[fixed offset was coverage={N1/T1:.4f}, p={p1}]", flush=True)
        mon = rolling["monitor"]
        print(f"Monitor: breach rate when ratio elevated={mon['elevated_breach_rate']}  "
              f"vs normal={mon['normal_breach_rate']}  |  "
              f"breaches preceded by an elevated reading={mon['breach_preceded_by_elevated_rate']}  "
              f"vs baseline={mon['baseline_preceded_by_elevated_rate']}", flush=True)
    if rolling_triggered is not None:
        rc2 = rolling_triggered["eval_slice_only"]
        rf = rolling_triggered["refreshes"]
        print(f"Rolling event-triggered (eval slice) coverage={rc2['coverage']:.4f} (kupiec p={rc2['kupiec_p']})  "
              f"refreshes: scheduled={rf['scheduled']} triggered={rf['triggered']}", flush=True)

    result = {
        "fold": fold_cfg["name"],
        "feature_set": feature_set,
        "train_start": fold_cfg["train_start"],
        "train_end": fold_cfg["train_end"],
        "test_start": fold_cfg["test_start"],
        "test_end": fold_cfg["test_end"],
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "epochs_run": epochs_run,
        "best_val_loss": best_val_loss,
        "accuracy": accuracy,
        "var_calibration": var_calibration,
        "rolling_calibration": rolling,
        "rolling_calibration_triggered": rolling_triggered,
    }

    print(f"VaR uncalibrated coverage={N0/T0:.4f} (kupiec p={p0})  "
          f"calibrated coverage={N1/T1:.4f} (kupiec p={p1})", flush=True)

    tracking.log_metrics(tracking.fold_metrics(result))
    model_uri = tracking.log_model(model, input_example=X_test[:2])
    if model_uri:
        result["mlflow_model_uri"] = model_uri
        print(f"logged model to mlflow: {model_uri}", flush=True)

    return result


def main(csv_path=DEFAULT_CSV, feature_set="original"):
    print(f"Loading data and engineering features (feature set: {feature_set})...", flush=True)
    data = load_and_engineer_features(csv_path)
    print(f"{len(data)} usable rows, {data['observation_date'].min().date()} to "
          f"{data['observation_date'].max().date()}", flush=True)

    path = results_path(feature_set)
    results = []
    if os.path.exists(path):
        with open(path) as f:
            results = json.load(f)
        done = {r["fold"] for r in results}
        print(f"Resuming: {len(results)} fold(s) already completed: {sorted(done)}", flush=True)
    else:
        done = set()

    for fold_cfg in FOLDS:
        if fold_cfg["name"] in done:
            continue
        try:
            result = run_fold(fold_cfg, data, feature_set=feature_set)
            results.append(result)
        except Exception as e:
            print(f"FOLD {fold_cfg['name']} FAILED: {e}", flush=True)
            traceback.print_exc()
            results.append({"fold": fold_cfg["name"], "error": str(e)})

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(results, f, indent=2)

    print("\n===== walk-forward complete =====", flush=True)
    for r in results:
        if "error" in r:
            print(f"{r['fold']}: ERROR - {r['error']}")
            continue
        vc = r["var_calibration"]
        line = (f"{r['fold']}: test={r['test_start']}..{r['test_end']}  "
                f"n_test={r['n_test']}  "
                f"return_MAE={r['accuracy']['log_return']['mae']:.5f}  "
                f"fixed_offset: uncal={vc['uncalibrated']['coverage']:.3f} "
                f"cal={vc['calibrated']['coverage']:.3f} p={vc['calibrated']['kupiec_p']:.3f}")
        rc = r.get("rolling_calibration")
        if rc:
            es = rc["eval_slice_only"]
            mon = rc["monitor"]
            line += (f"  |  rolling: cov={es['coverage']:.3f} p={es['kupiec_p']}  "
                     f"monitor: elevated_breach_rate={mon['elevated_breach_rate']} "
                     f"normal_breach_rate={mon['normal_breach_rate']} "
                     f"lead={mon['breach_preceded_by_elevated_rate']} "
                     f"vs baseline={mon['baseline_preceded_by_elevated_rate']}")
        rct = r.get("rolling_calibration_triggered")
        if rct:
            es2 = rct["eval_slice_only"]
            rf = rct["refreshes"]
            line += (f"  |  triggered: cov={es2['coverage']:.3f} p={es2['kupiec_p']} "
                     f"refreshes(sched={rf['scheduled']},trig={rf['triggered']})")
        print(line)


if __name__ == "__main__":
    main()
