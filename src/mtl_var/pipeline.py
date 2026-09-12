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

Results are written to results/walk_forward_results.json after every fold, so a crash partway
through doesn't lose completed folds.
"""
import gc
import json
import os
import traceback

import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from .calibration import fixed_offset_calibration, rolling_calibration_analysis
from .data import DEFAULT_CSV, FEATURES, SEQUENCE_LENGTH, TARGETS, load_and_engineer_features, make_sequences
from .model import train_model
from .seeding import set_seeds

RESULTS_PATH = "results/walk_forward_results.json"
PREDICTIONS_DIR = "results/predictions"

FOLDS = [
    {"name": "fold1_gfc",        "train_start": "2004-01-01", "train_end": "2007-12-31", "test_start": "2008-01-01", "test_end": "2009-12-31"},
    {"name": "fold2_eu_crisis",  "train_start": "2004-01-01", "train_end": "2010-12-31", "test_start": "2011-01-01", "test_end": "2013-12-31"},
    {"name": "fold3_2015_selloff","train_start": "2004-01-01", "train_end": "2014-12-31", "test_start": "2015-01-01", "test_end": "2017-12-31"},
    {"name": "fold4_covid",      "train_start": "2004-01-01", "train_end": "2018-12-31", "test_start": "2019-01-01", "test_end": "2020-12-31"},
    {"name": "fold5_2022_bear",  "train_start": "2004-01-01", "train_end": "2021-12-31", "test_start": "2022-01-01", "test_end": "2022-12-31"},
]


def build_fold_data(fold_cfg, data):
    """Per-fold split, feature scaling, sequencing, and target standardisation - scalers fit
    on that fold's train rows only."""
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
    }


def run_fold(fold_cfg, data):
    print(f"\n===== {fold_cfg['name']}  train[{fold_cfg['train_start']}:{fold_cfg['train_end']}]  "
          f"test[{fold_cfg['test_start']}:{fold_cfg['test_end']}] =====", flush=True)

    tf.keras.backend.clear_session()
    gc.collect()
    set_seeds()

    fd = build_fold_data(fold_cfg, data)
    X_train, X_test = fd["X_train"], fd["X_test"]
    y_train_z, y_test_z = fd["y_train_z"], fd["y_test_z"]
    y_test_raw, target_scalers = fd["y_test_raw"], fd["target_scalers"]
    test_df = fd["test_df"]

    print(f"n_train={len(X_train)}  n_test={len(X_test)}  n_features={len(FEATURES)}", flush=True)

    n_features = len(FEATURES)
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

    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    np.savez(
        os.path.join(PREDICTIONS_DIR, f"{fold_cfg['name']}.npz"),
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

    return result


def main(csv_path=DEFAULT_CSV):
    print("Loading data and engineering features...", flush=True)
    data = load_and_engineer_features(csv_path)
    print(f"{len(data)} usable rows, {data['observation_date'].min().date()} to "
          f"{data['observation_date'].max().date()}", flush=True)

    results = []
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            results = json.load(f)
        done = {r["fold"] for r in results}
        print(f"Resuming: {len(results)} fold(s) already completed: {sorted(done)}", flush=True)
    else:
        done = set()

    for fold_cfg in FOLDS:
        if fold_cfg["name"] in done:
            continue
        try:
            result = run_fold(fold_cfg, data)
            results.append(result)
        except Exception as e:
            print(f"FOLD {fold_cfg['name']} FAILED: {e}", flush=True)
            traceback.print_exc()
            results.append({"fold": fold_cfg["name"], "error": str(e)})

        with open(RESULTS_PATH, "w") as f:
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
