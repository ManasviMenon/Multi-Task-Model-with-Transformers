"""
Step 7 of the early-stopping fix (RESULTS.md section 15): retrain the two single-task models on
every fold with early stopping on the training tail instead of the test window.

  var_proxy        single-task head, pinball q=0.05 on the constructed VaR_5 target
                   (the section 10/13 ablation; old numbers in leaky_earlystop/leakage_free_ablation_results.json)
  return_quantile  same head, pinball q=0.05 on the realised log return
                   (section 14.4 control; old numbers in leaky_earlystop/return_quantile_control_results.json)

Both use mtl_var.model.train_single_task (identical loop to the scripts that produced the old
numbers) via fit_with_train_tail_validation: the last 15% of training sequences, chronological,
decide early stopping. The test window is first used by model.predict after training ends.
Pre-registered readings: scripts/cleanval_report.py (P4).
Outputs: results/cleanval_single_task_results.json and
results/predictions_{var_proxy,return_quantile}_cleanval/*.npz. Resumes per fold.
"""
import json
import os
import sys

import numpy as np

from leakage_free_retrain import build_clean_frame, build_fold_arrays
from mtl_var.backtest import full_backtest
from mtl_var.calibration import fixed_offset_calibration, rolling_calibration_analysis
from mtl_var.data import SEQUENCE_LENGTH, load_and_engineer_features
from mtl_var.model import fit_with_train_tail_validation, train_single_task
from mtl_var.pipeline import FOLDS

OUT = "results/cleanval_single_task_results.json"
MTL_PRED_DIR = "results/predictions_leakage_free_cleanval"
VARIANTS = {"var_proxy": "VaR_5", "return_quantile": "log_return"}


def evaluate(dates, returns, var, calib_n):
    fo = fixed_offset_calibration(returns, var)
    r_e, v_e = returns[calib_n:], var[calib_n:]
    res = {"offset": fo["offset"], "raw": full_backtest(r_e, v_e),
           "frozen": full_backtest(r_e, v_e + fo["offset"])}
    assert res["frozen"]["N"] == fo["calibrated"]["breaches"]
    rc = rolling_calibration_analysis(dates, returns, var, eval_start_idx=calib_n)
    L = rc["params"]["L"]
    idx = np.arange(L, len(returns))
    offs, cur, refresh = np.empty(len(idx)), None, set(rc["refresh_days"])
    for k, t in enumerate(idx):
        if t in refresh:
            cur = float(np.quantile(returns[t - L:t] - var[t - L:t], 0.05))
        offs[k] = cur
    m = idx >= calib_n
    res["rolling"] = full_backtest(returns[idx[m]], var[idx[m]] + offs[m])
    assert res["rolling"]["N"] == rc["eval_slice_only"]["breaches"]
    return res


def run(fold_cfg, clean, names, variant):
    target = VARIANTS[variant]
    print(f"\n===== {fold_cfg['name']} / {variant} (target {target}) =====", flush=True)
    X_train, X_test, y_train_z, _y_test_z_unused, y_test_raw, scalers = build_fold_arrays(fold_cfg, clean, names)
    model, epochs, best_val, n_fit, n_val = fit_with_train_tail_validation(
        X_train, y_train_z[target], len(names), trainer=train_single_task)
    print(f"trained {epochs} epochs on {n_fit}, best val_loss={best_val:.6f} on last {n_val}", flush=True)

    var = scalers[target].inverse_transform(np.asarray(model.predict(X_test, verbose=0)).reshape(-1, 1)).flatten()
    returns = y_test_raw["log_return"]
    dates = clean.loc[(clean["observation_date"] >= fold_cfg["test_start"]) &
                      (clean["observation_date"] <= fold_cfg["test_end"]),
                      "observation_date"].values[SEQUENCE_LENGTH - 1:].astype("datetime64[D]")
    calib_n = len(var) // 2
    ref = np.load(os.path.join(MTL_PRED_DIR, f"{fold_cfg['name']}.npz"), allow_pickle=True)
    assert np.array_equal(ref["dates"].astype("datetime64[D]"), dates)
    assert np.array_equal(ref["returns"], returns) and int(ref["calib_n"]) == calib_n

    d = f"results/predictions_{variant}_cleanval"
    os.makedirs(d, exist_ok=True)
    np.savez(os.path.join(d, f"{fold_cfg['name']}.npz"), dates=dates.astype(str), returns=returns,
             var_pred=var, calib_n=calib_n)
    res = evaluate(dates, returns, var, calib_n)
    res.update({"epochs_run": epochs, "best_val_loss": best_val, "n_fit": n_fit, "n_val": n_val})
    for k in ("raw", "frozen", "rolling"):
        b = res[k]
        print(f"  {k:8s} {b['N']}/{b['T']}  p_pof={b['p_pof']:.3g}  p_cc={b['p_cc']:.3g}", flush=True)
    return res


def main():
    data = load_and_engineer_features("SPX500-18Y_scaled_y.csv")
    clean, names = build_clean_frame(data)
    results = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for fold_cfg in FOLDS:
        for variant in VARIANTS:
            if variant in results.get(fold_cfg["name"], {}) or (sys.argv[1:] and fold_cfg["name"] not in sys.argv[1:]):
                continue
            results.setdefault(fold_cfg["name"], {})[variant] = run(fold_cfg, clean, names, variant)
            with open(OUT, "w") as f:
                json.dump(results, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
