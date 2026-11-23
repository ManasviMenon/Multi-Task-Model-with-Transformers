"""
ACAI 2026 review, revision 2 control: train the VaR head directly on realised returns.

The reviewer's point: the VaR head is trained with pinball loss (q=0.05) against a constructed
target, VaR_5 = EWMA_10(log_return) - 1.2 * yz_vol, so pinball loss estimates the conditional 5%
quantile of that PROXY, not of the return. This control trains the identical single-task model
(single_task_var_ablation.build_single_task_model / train_single_task: same transformer body,
one Dense(1) head, pinball q=0.05, Adam 1e-4, batch 32, patience 5, seed 42) on the leakage-free
features, with ONE change: the target is the realised log return of the same row. Every
feature in row t is known before day t, so the head estimates the conditional 5% quantile of
r_t given information through t-1.

Everything downstream is the same as the primary results: fixed_offset_calibration (half/half
frozen offset) and rolling recalibration (L=125, R=21), backtested with mtl_var.backtest.full_backtest.
Predictions are cached to results/predictions_return_quantile/ so later sweeps need no retraining.

Known protocol property, inherited unchanged from every training run in this project and kept
here so the comparison is like-for-like: early stopping monitors loss on the fold's test window
(train_single_task / train_model pass X_test as validation data). Stated, not fixed, here.

------------------------------------------------------------------------------------------
PRE-REGISTERED READINGS (written before training):
  C1. Raw coverage. If the head is a genuine conditional-quantile estimator, raw (uncalibrated)
      evaluation-half coverage should be far closer to 5% than the proxy head's 30-41%. Reported
      per fold with N/T; no pass/fail attached.
  C2. The question the reviewer asked: under the FROZEN offset, is the return-quantile head
      rejected (Kupiec or LR_cc, 5%) in at least 3 of the 4 folds where the proxy-trained model is
      rejected (GFC, EU crisis, 2015 selloff, COVID)? If yes: the regime-conditioned failure
      persists when the head targets the return quantile directly. If no: the failure depends on
      the constructed target and the paper must say so.
  C3. Secondary: does the frozen offset make the return-quantile head worse than its own raw
      output (raw not rejected, frozen rejected) in any fold? Reported as in revision 4 (R4c).
------------------------------------------------------------------------------------------
"""
import json
import os
import sys

import numpy as np
from sklearn.metrics import mean_absolute_error

from leakage_free_retrain import build_clean_frame, build_fold_arrays
from mtl_var.backtest import full_backtest
from mtl_var.calibration import fixed_offset_calibration, rolling_calibration_analysis
from mtl_var.data import SEQUENCE_LENGTH, load_and_engineer_features
from mtl_var.pipeline import FOLDS
from single_task_var_ablation import train_single_task

OUT = "results/return_quantile_control_results.json"
PRED_DIR = "results/predictions_return_quantile"
LF_PRED_DIR = "results/predictions_leakage_free"


def evaluate(dates, returns, var, calib_n):
    fo = fixed_offset_calibration(returns, var)
    r_e, v_e = returns[calib_n:], var[calib_n:]
    res = {"offset": fo["offset"], "raw": full_backtest(r_e, v_e),
           "frozen": full_backtest(r_e, v_e + fo["offset"])}
    rc = rolling_calibration_analysis(dates, returns, var, eval_start_idx=calib_n)
    if rc is not None:
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


def run_fold(fold_cfg, clean, names):
    print(f"\n===== {fold_cfg['name']}: single-task head, pinball q=0.05 on realised log return =====",
          flush=True)
    X_train, X_test, y_train_z, y_test_z, y_test_raw, scalers = build_fold_arrays(fold_cfg, clean, names)
    model, epochs, best_val = train_single_task(
        X_train, y_train_z["log_return"], X_test, y_test_z["log_return"], len(names))
    print(f"trained {epochs} epochs, best val_loss={best_val:.6f}", flush=True)

    pred_z = np.asarray(model.predict(X_test, verbose=0)).reshape(-1, 1)
    var = scalers["log_return"].inverse_transform(pred_z).flatten()
    returns = y_test_raw["log_return"]

    dates_df = clean.loc[(clean["observation_date"] >= fold_cfg["test_start"]) &
                         (clean["observation_date"] <= fold_cfg["test_end"]), "observation_date"]
    dates = dates_df.values[SEQUENCE_LENGTH - 1:].astype("datetime64[D]")
    calib_n = len(var) // 2

    # same calendar and returns as the primary leakage-free model, by construction - checked
    lf = np.load(os.path.join(LF_PRED_DIR, f"{fold_cfg['name']}.npz"), allow_pickle=True)
    assert np.array_equal(lf["dates"].astype("datetime64[D]"), dates), fold_cfg["name"]
    assert np.allclose(lf["returns"], returns, rtol=0, atol=0), fold_cfg["name"]
    assert int(lf["calib_n"]) == calib_n

    os.makedirs(PRED_DIR, exist_ok=True)
    np.savez(os.path.join(PRED_DIR, f"{fold_cfg['name']}.npz"), dates=dates.astype(str),
             returns=returns, var_pred=var, calib_n=calib_n)

    res = evaluate(dates, returns, var, calib_n)
    res.update({"fold": fold_cfg["name"], "epochs_run": epochs, "best_val_loss": best_val,
                "n_train": int(len(X_train)), "n_test": int(len(X_test)),
                "proxy_mae_vs_VaR5_target": float(mean_absolute_error(y_test_raw["VaR_5"], var))})
    for k in ("raw", "frozen", "rolling"):
        bt = res.get(k)
        if bt:
            print(f"  {k:8s} {bt['N']}/{bt['T']} = {bt['coverage']*100:.2f}%  p_pof={bt['p_pof']:.3g}  "
                  f"p_ind={bt['p_ind']:.3g}  p_cc={bt['p_cc']:.3g}", flush=True)
    return res


def main():
    print(__doc__, flush=True)
    data = load_and_engineer_features("SPX500-18Y_scaled_y.csv")
    clean, names = build_clean_frame(data)
    results = {}
    if os.path.exists(OUT):
        with open(OUT) as f:
            results = json.load(f)
    for fold_cfg in FOLDS:
        if fold_cfg["name"] in results or (sys.argv[1:] and fold_cfg["name"] not in sys.argv[1:]):
            continue
        results[fold_cfg["name"]] = run_fold(fold_cfg, clean, names)
        with open(OUT, "w") as f:
            json.dump(results, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
