"""
Classical baselines for Table 1 (paper's "Final Model vs. Classical Baselines" table).

Table 1 originally shipped with four baseline rows (Persistence, ARIMA(1,0,1), GARCH(1,1),
EWMA VaR) that do not exist anywhere in this codebase - grepped the notebook, every script,
and every results JSON for "arima"/"garch"/"persistence" and found zero matches. Those numbers
had no source. This script actually builds and evaluates the three classical methods the table
needs, on the identical single 80/20 chronological split the notebook's single-split MTL numbers
come from, so Table 1 becomes real instead of fabricated.

Reuses, UNMODIFIED, from walk_forward.py:
  - load_and_engineer_features: same feature/target engineering, same NaN cascade
  - kupiec_from_counts: the exact same Kupiec POF test used everywhere else in this project

Does NOT modify walk_forward.py, the notebook, or any existing results file. Writes a new
JSON only (results/classical_baselines_results.json).

------------------------------------------------------------------------------------------
SCOPE DECISIONS, fixed before running (so nothing gets tuned after seeing numbers):

Test period: the LAST 941 rows of the cleaned dataset. This is the exact same calendar
window as the notebook's single-split test set (confirmed: cell 10 of MTL_Improved.ipynb
prints "Train samples: 3760, Test samples: 941" out of 4701 sequences from a 4705-row
cleaned dataset - X_seq loses 4 rows to 5-day windowing, so the sequence-based test set and
"last 941 raw rows" are the same 941 calendar days). Train = everything before that (3764
raw rows for these row-wise classical models, vs 3760 sequences for the MTL model - a
4-row difference from windowing, immaterial, noted here rather than engineered around).

1. Persistence (naive): forecast_t = actual return_{t-1} (the standard "tomorrow = today"
   persistence forecast), not "predict zero". Only Ret. MAE is reported - Table 1 leaves the
   other two cells blank for this row and there's no principled persistence-based VaR to add.

2. ARIMA(1,0,1) on log_return: fit ONCE on the train partition, parameters then FROZEN -
   matching how the MTL model is trained once and evaluated frozen on test, so the comparison
   is fair on that axis. One-step-ahead forecasts across the test window come from extending
   the fitted state with true past observations (statsmodels `.append(refit=False)`), never
   refitting. VaR: ARIMA's one-step mean forecast plus a static residual-quantile offset
   (5th percentile of TRAIN residuals) - report both RAW (offset fit once on train, applied to
   the whole test set) and the same half/half calibrate-then-evaluate protocol used everywhere
   else in this project, exactly like empirical_var_comparison.py's raw/calibrated pattern.

3. GARCH(1,1) on log_return: fit ONCE on train (zero-mean, GARCH(1,1) variance), parameters
   frozen; one-step-ahead conditional-volatility forecasts across test via the arch package's
   fixed-parameter recursive forecast. Vol. MAE is evaluated against the same yz_vol actual
   values the MTL model targets - GARCH natively forecasts return volatility, not an OHLC-range
   proxy, so this is a disclosed cross-definition comparison, not a claim that GARCH targets
   yz_vol. It's what Table 1's generic "Vol. MAE" column asks every row to be measured against.

4. EWMA VaR (RiskMetrics-style, lambda=0.94, zero mean, z_0.05=-1.6449): sigma_t^2 =
   lambda*sigma_{t-1}^2 + (1-lambda)*r_{t-1}^2, seeded with train-period variance; VaR_t =
   z_0.05 * sigma_t. This is the standard RiskMetrics EWMA-VaR baseline - NOT to be confused
   with this project's own VaR_5 target, which uses EWMA_10 only on the *mean* of returns, not
   on volatility. Report RAW (EWMA is inherently adaptive, no offset needed by design) AND the
   same half/half calibrated variant, again matching the empirical-VaR script's pattern, so
   nothing is cherry-picked between the two.

PRE-REGISTERED INTERPRETATION (written before any of this was run or looked at):
  - If a classical baseline passes Kupiec on this split while the corrected MTL model fails
    (2.76% cov, p=0.0152, per RESULTS.md section 2) -> report that plainly as a real weakness
    of MTL's single-split calibration, not something to bury under "but MTL has better MAE".
  - If MTL's MAE beats every classical baseline -> report it, but note (as RESULTS.md section
    11 already does) that this single split's calibration verdict is itself seed-fragile, so
    "better MAE, same-or-worse coverage" is the honest framing, not "MTL wins across the board".
  - No cherry-picking between raw/calibrated variants for ARIMA-VaR or EWMA-VaR - both get
    reported for both methods.
------------------------------------------------------------------------------------------
"""
import json

import numpy as np
from scipy.stats import norm
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

from walk_forward import load_and_engineer_features, kupiec_from_counts

RESULTS_PATH = "results/classical_baselines_results.json"
N_TEST = 941
EWMA_LAMBDA = 0.94
Z_05 = norm.ppf(0.05)  # -1.6449


def kupiec_eval(returns_eval, var_eval):
    T = len(var_eval)
    N = int(np.sum(returns_eval < var_eval))
    return N, T, kupiec_from_counts(N, T)


def calibrate_half_half(returns_test, var_test):
    """Identical protocol to walk_forward.run_fold: first half of test fits a residual-quantile
    offset, second half (untouched) is evaluated. Returns both the raw (uncalibrated, full test
    window) and the calibrated (offset fit on first half, evaluated on second half) results."""
    N0, T0, p0 = kupiec_eval(returns_test, var_test)

    calib_n = len(var_test) // 2
    returns_calib, returns_eval = returns_test[:calib_n], returns_test[calib_n:]
    var_calib, var_eval = var_test[:calib_n], var_test[calib_n:]
    offset = float(np.quantile(returns_calib - var_calib, 0.05))
    var_eval_cal = var_eval + offset
    N1, T1, p1 = kupiec_eval(returns_eval, var_eval_cal)

    return {
        "raw": {"n": T0, "breaches": N0, "coverage": N0 / T0, "kupiec_p": p0},
        "calibrated": {"n": T1, "breaches": N1, "coverage": N1 / T1, "kupiec_p": p1, "offset": offset},
    }


def run_persistence(train_ret, test_ret):
    full_ret = np.concatenate([train_ret, test_ret])
    split = len(train_ret)
    forecast = full_ret[split - 1:split - 1 + len(test_ret)]  # yesterday's realized return
    mae = float(np.mean(np.abs(test_ret - forecast)))
    return {"ret_mae": mae}


def run_arima(train_ret, test_ret):
    model = ARIMA(train_ret, order=(1, 0, 1))
    fit = model.fit()
    train_resid = np.asarray(fit.resid)

    extended = fit.append(test_ret, refit=False)
    one_step = np.asarray(extended.fittedvalues)[-len(test_ret):]

    ret_mae = float(np.mean(np.abs(test_ret - one_step)))

    resid_q05 = float(np.quantile(train_resid, 0.05))
    var_pred_raw = one_step + resid_q05
    var_result = calibrate_half_half(test_ret, var_pred_raw)

    return {"ret_mae": ret_mae, "var": var_result,
            "params": {name: float(val) for name, val in zip(fit.model.param_names, fit.params)}}


def run_garch(train_ret, test_ret, test_yz_actual):
    # NOTE on units: yz_vol (this project's target) is a rolling MEAN of squared log-return-like
    # terms - a variance-scale quantity (test-period mean ~1.5e-4), not a standard deviation
    # (whose scale would be ~1.1e-2, ~70x larger). GARCH natively forecasts conditional
    # VARIANCE, so we compare variance-to-variance here, not GARCH's std against yz_vol's
    # variance - an earlier version of this script did that by mistake and got a MAE that was
    # really just measuring the unit mismatch, not forecast quality.
    am = arch_model(train_ret * 100, mean="Zero", vol="GARCH", p=1, q=1)
    res = am.fit(disp="off")

    full_ret_pct = np.concatenate([train_ret, test_ret]) * 100
    am_full = arch_model(full_ret_pct, mean="Zero", vol="GARCH", p=1, q=1)
    fixed_res = am_full.fix(res.params)
    fc = fixed_res.forecast(horizon=1, start=len(train_ret), reindex=False)
    cond_var_pct2 = fc.variance.values.flatten()
    cond_var = cond_var_pct2 / (100.0 ** 2)  # back to raw log-return variance units

    vol_mae = float(np.mean(np.abs(test_yz_actual - cond_var)))
    return {"vol_mae": vol_mae, "params": {k: float(v) for k, v in res.params.items()}}


def run_ewma_var(train_ret, test_ret):
    sigma2 = float(np.var(train_ret))
    full_ret = np.concatenate([train_ret, test_ret])
    split = len(train_ret)

    sigma2_series = np.zeros(len(full_ret))
    sigma2_series[0] = sigma2
    for t in range(1, len(full_ret)):
        sigma2_series[t] = EWMA_LAMBDA * sigma2_series[t - 1] + (1 - EWMA_LAMBDA) * full_ret[t - 1] ** 2

    sigma_test = np.sqrt(sigma2_series[split:])
    var_pred_raw = Z_05 * sigma_test
    var_result = calibrate_half_half(test_ret, var_pred_raw)
    return {"var": var_result}


def main():
    print(__doc__, flush=True)
    data = load_and_engineer_features("SPX500-18Y_scaled_y.csv")
    print(f"cleaned dataset rows: {len(data)}", flush=True)

    train_df = data.iloc[:-N_TEST].reset_index(drop=True)
    test_df = data.iloc[-N_TEST:].reset_index(drop=True)
    print(f"train rows: {len(train_df)}  test rows: {len(test_df)}  "
          f"test dates: {test_df['observation_date'].min()} -> {test_df['observation_date'].max()}",
          flush=True)

    train_ret = train_df["log_return"].values
    test_ret = test_df["log_return"].values
    test_yz = test_df["yz_vol"].values

    print("\n--- Persistence ---", flush=True)
    persistence = run_persistence(train_ret, test_ret)
    print(persistence, flush=True)

    print("\n--- ARIMA(1,0,1) ---", flush=True)
    arima = run_arima(train_ret, test_ret)
    print({"ret_mae": arima["ret_mae"], "var_raw": arima["var"]["raw"], "var_cal": arima["var"]["calibrated"]}, flush=True)

    print("\n--- GARCH(1,1) ---", flush=True)
    garch = run_garch(train_ret, test_ret, test_yz)
    print(garch, flush=True)

    print("\n--- EWMA VaR ---", flush=True)
    ewma = run_ewma_var(train_ret, test_ret)
    print({"var_raw": ewma["var"]["raw"], "var_cal": ewma["var"]["calibrated"]}, flush=True)

    results = {
        "test_period": {
            "n": N_TEST,
            "start": str(test_df["observation_date"].min()),
            "end": str(test_df["observation_date"].max()),
        },
        "persistence": persistence,
        "arima_1_0_1": arima,
        "garch_1_1": garch,
        "ewma_var": ewma,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
