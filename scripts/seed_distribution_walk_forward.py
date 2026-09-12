"""
Multi-seed distribution of the fixed-offset walk-forward baseline (RESULTS.md section 3).
The single-split seed-sensitivity check (verify_seed_sensitivity.py) showed 3 of 4 alternate
seeds PASS where the reference seed=42 FAILS - a single point estimate per fold isn't a
robust claim. This converts "fold X fails" into "fold X fails in N/10 seeds", turning a
fragile point estimate into a distribution, per the plan agreed before running this.

Scope, deliberately bounded: the fixed-offset baseline only, not the rolling/triggered/
threshold/window-length sweeps built on top of it - each of those would need its own
multi-seed treatment, and this alone is already a ~10x compute multiplier on section 3
(10 seeds x 5 folds = 50 trainings instead of 5). If this baseline is robust across seeds,
the sweeps built on the seed=42 predictions are on solid ground; if it isn't, that's the
headline finding and the sweeps need the same treatment as a follow-up, not bundled in here.

Runs sequentially, writes results incrementally after EVERY single training (not just at
the end) so a multi-hour run stays resumable if interrupted.

Does not modify walk_forward.py. Writes a new JSON only (seed_distribution_results.json).
"""
import json
import os

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from walk_forward import (
    FEATURES, SEQUENCE_LENGTH, FOLDS, TARGETS,
    weighted_loss, load_and_engineer_features, kupiec_from_counts, make_sequences,
)
from verify_seed_sensitivity import set_seeds_custom, train_once_no_reseed
from mtl_var import tracking

SEEDS = list(range(42, 52))  # 10 seeds: 42 (the original reference) through 51
RESULTS_PATH = "results/seed_distribution_results.json"


def build_fold_data(fold_cfg, data):
    """Mirrors walk_forward.py's run_fold data-prep exactly (per-fold feature/target
    scalers fit on train only) - duplicated here since run_fold isn't factored into a
    separate data-prep function and this task is not to modify that file."""
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

    return X_train, X_test, y_train_z, y_test_z, y_train_raw, y_test_raw, target_scalers


def evaluate_fixed_offset(y_test_raw, native_preds):
    """Identical procedure to walk_forward.py's run_fold: split test in half chronologically,
    fit a 5th-percentile-of-residuals offset on the first half, Kupiec-test the second half,
    before and after."""
    returns = y_test_raw['log_return']
    var_pred = native_preds['VaR_5']
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
        "uncalibrated": {"breaches": N0, "coverage": N0 / T0, "kupiec_p": p0},
        "calibrated": {"breaches": N1, "coverage": N1 / T1, "kupiec_p": p1},
        "offset": offset,
    }


def main():
    print(__doc__, flush=True)
    data = load_and_engineer_features("SPX500-18Y_scaled_y.csv")

    results = []
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            results = json.load(f)
        done = {(r["fold"], r["seed"]) for r in results}
        print(f"Resuming: {len(results)} (fold,seed) combos already done", flush=True)
    else:
        done = set()

    for fold_cfg in FOLDS:
        fold = fold_cfg["name"]
        X_train, X_test, y_train_z, y_test_z, y_train_raw, y_test_raw, target_scalers = build_fold_data(fold_cfg, data)
        n_features = len(FEATURES)

        for seed in SEEDS:
            if (fold, seed) in done:
                continue
            print(f"\n--- {fold}  seed={seed} ---", flush=True)
            tf.keras.backend.clear_session()
            set_seeds_custom(seed)
            model = train_once_no_reseed(weighted_loss, X_train, y_train_z, X_test, y_test_z, n_features)

            preds = model.predict(X_test, verbose=0)
            native_preds = {t: target_scalers[t].inverse_transform(p.reshape(-1, 1)).flatten()
                             for t, p in zip(TARGETS, preds)}

            r = evaluate_fixed_offset(y_test_raw, native_preds)
            row = {
                "fold": fold, "seed": seed,
                "uncalibrated_coverage": r["uncalibrated"]["coverage"],
                "calibrated_coverage": r["calibrated"]["coverage"],
                "calibrated_kupiec_p": r["calibrated"]["kupiec_p"],
            }

            # Added in the MLflow phase: one run per (fold, seed) so the campaign is
            # comparable in the UI. Deliberately a single call placed AFTER training rather
            # than a context manager around the loop body - nothing may sit between
            # set_seeds_custom and train_once_no_reseed above.
            tracking.log_standalone_run(
                run_name=f"{fold}_seed{seed}",
                tags={"fold": fold, "seed": seed, "campaign": "seed-distribution",
                      "stage": "fixed-offset-baseline"},
                params={"fold": fold, "seed": seed, "n_train": len(X_train), "n_test": len(X_test),
                        "n_features": n_features, **tracking.ARCHITECTURE, **tracking.TRAINING_DEFAULTS},
                metrics={"var_uncal_coverage": r["uncalibrated"]["coverage"],
                         "var_uncal_kupiec_p": r["uncalibrated"]["kupiec_p"],
                         "var_cal_coverage": r["calibrated"]["coverage"],
                         "var_cal_kupiec_p": r["calibrated"]["kupiec_p"],
                         "var_offset": r["offset"]},
            )
            results.append(row)
            with open(RESULTS_PATH, "w") as f:
                json.dump(results, f, indent=2)

            p = r["calibrated"]["kupiec_p"]
            verdict = "PASS" if (p is not None and not np.isnan(p) and p >= 0.05) else "FAIL"
            print(f"  cal_coverage={r['calibrated']['coverage']*100:.2f}%  p={p}  verdict={verdict}", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("SUMMARY: pass rate by fold", flush=True)
    for fold_cfg in FOLDS:
        fold = fold_cfg["name"]
        rows = [r for r in results if r["fold"] == fold]
        passes = sum(1 for r in rows if r["calibrated_kupiec_p"] is not None
                     and not np.isnan(r["calibrated_kupiec_p"]) and r["calibrated_kupiec_p"] >= 0.05)
        covs = [r["calibrated_coverage"] for r in rows]
        print(f"{fold:22s}: {passes}/{len(rows)} pass  "
              f"coverage mean={np.mean(covs)*100:.2f}% std={np.std(covs)*100:.2f}% "
              f"min={np.min(covs)*100:.2f}% max={np.max(covs)*100:.2f}%", flush=True)


if __name__ == "__main__":
    main()
