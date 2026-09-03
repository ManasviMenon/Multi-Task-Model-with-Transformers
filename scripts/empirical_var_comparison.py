"""
Parallel comparison: does an empirical (historical-percentile) VaR fail the same
walk-forward test that the parametric VaR does? Addresses the reviewer objection that the
parametric-VaR walk-forward failure (RESULTS.md sections 3-8) could be an artifact of the
parametric target definition (EWMA of returns minus 1.2x Yang-Zhang volatility) rather than
a property of static calibration under regime shift.

Reuses, UNMODIFIED, from walk_forward.py:
  - FOLDS: the exact same 5 fold boundaries (train/test date ranges)
  - kupiec_from_counts: the exact same Kupiec POF test
  - predictions/*.npz: the exact same test-period dates, returns, and calib/eval split
    (calib_n = first half fits the offset, second half is held out and evaluated) used
    for the parametric model, so "same folds, same protocol" holds by construction.

Does NOT modify walk_forward.py or any existing results file. Writes a new JSON
(empirical_var_comparison_results.json) and a new RESULTS.md section only.

------------------------------------------------------------------------------------------
PRE-REGISTERED INTERPRETATION (written before any results were computed or looked at):

  - If empirical VaR ALSO fails (coverage far from 5%, Kupiec rejects) in the same folds
    that broke the parametric VaR -> the walk-forward failure is regime-driven and
    method-independent. This STRENGTHENS the paper's central claim: static calibration
    fails under regime shift regardless of how the VaR target is defined.

  - If empirical VaR PASSES where parametric VaR failed -> the target definition matters
    more than regime shift alone. This RESHAPES the conclusion and must be reported as
    such, not buried under the existing narrative.

  - The RAW (uncalibrated) empirical VaR result answers a distinct question: is a rolling
    historical percentile self-calibrating out of the box, with no post-hoc offset at all?
    A pass there is a materially different, stronger finding than "calibration helps it."

  - The sensitivity sweep (N = 125 / 250 / 500) exists to check the pass/fail pattern isn't
    an artifact of one arbitrary lookback choice. A pattern that flips between N=125 and
    N=500 is reported as unstable, not smoothed over by picking the most favorable N.
------------------------------------------------------------------------------------------
"""
import json
import os

import numpy as np
import pandas as pd

from walk_forward import FOLDS, kupiec_from_counts

LOOKBACKS = [125, 250, 500]  # trading days; 250 ~= 1 trading year, the standard hist-VaR default
DEFAULT_N = 250

RESULTS_PATH = "results/empirical_var_comparison_results.json"


def load_raw_returns(csv_path):
    """
    Log returns from the raw price series only - deliberately decoupled from the parametric
    pipeline's feature-engineering NaN cascade. walk_forward.load_and_engineer_features
    compounds several rolling(20) requirements (yz_vol, CVaR_5, ret_min20/max20, skew) via
    sequential dropna calls, which truncates the usable series by ~80 rows at the start.
    A historical-VaR lookback has no such requirement, so it shouldn't inherit that
    truncation - it should get the longest legitimate history available.
    """
    data = pd.read_csv(csv_path, parse_dates=['observation_date'], dayfirst=True)
    data = data.dropna(axis=1, how='all')
    data = data.drop(index=0)
    data['log_return'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1))
    data = data.dropna(subset=['log_return']).sort_values('observation_date').reset_index(drop=True)
    dates = data['observation_date'].values.astype('datetime64[D]')
    returns = data['log_return'].values
    return dates, returns


def empirical_var_series(raw_dates, raw_returns, target_dates, N):
    """
    5th percentile of the trailing N returns strictly before each target date - no
    look-ahead. The window for the date at series-index i is raw_returns[i-N:i], which
    excludes day i itself.
    """
    date_to_idx = {d: i for i, d in enumerate(raw_dates)}
    out = np.full(len(target_dates), np.nan)
    for k, d in enumerate(target_dates):
        i = date_to_idx.get(d)
        if i is None or i < N:
            continue
        out[k] = np.quantile(raw_returns[i - N:i], 0.05)
    return out


def fit_and_evaluate_offset(returns, var_pred, calib_n):
    """
    Mirrors, exactly, the inline calibration procedure in walk_forward.py's run_fold
    (lines ~416-428): fit a 5th-percentile-of-residuals offset on the first half of the
    test window, apply it unchanged to the untouched second half, Kupiec-test the second
    half only. Duplicated here rather than imported because that logic isn't factored into
    a function in walk_forward.py and this task is not to modify that file.
    """
    returns_calib, returns_eval = returns[:calib_n], returns[calib_n:]
    var_calib, var_eval = var_pred[:calib_n], var_pred[calib_n:]
    offset = float(np.quantile(returns_calib - var_calib, 0.05))
    var_eval_calibrated = var_eval + offset

    def kupiec_eval(returns_e, var_e):
        T = len(var_e)
        N = int(np.sum(returns_e < var_e))
        return N, T, kupiec_from_counts(N, T)

    N0, T0, p0 = kupiec_eval(returns_eval, var_eval)
    N1, T1, p1 = kupiec_eval(returns_eval, var_eval_calibrated)
    return {
        "offset": offset,
        "raw": {"breaches": N0, "n": T0, "coverage": N0 / T0 if T0 else None, "kupiec_p": p0},
        "calibrated": {"breaches": N1, "n": T1, "coverage": N1 / T1 if T1 else None, "kupiec_p": p1},
    }


def main():
    print(__doc__)

    raw_dates, raw_returns = load_raw_returns("SPX500-18Y_scaled_y.csv")
    print(f"Raw return series: {len(raw_returns)} days, {raw_dates[0]} to {raw_dates[-1]}\n")

    results = {}
    for fold_cfg in FOLDS:
        fold = fold_cfg["name"]
        path = os.path.join("results/predictions", f"{fold}.npz")
        if not os.path.exists(path):
            print(f"{fold}: no cached predictions at {path}, skipping (run walk_forward.py first)")
            continue
        d = np.load(path, allow_pickle=True)
        dates = d["dates"].astype("datetime64[D]")
        returns = d["returns"]
        var_pred_parametric = d["var_pred"]
        calib_n = int(d["calib_n"])

        parametric = fit_and_evaluate_offset(returns, var_pred_parametric, calib_n)

        empirical_by_N = {}
        for N in LOOKBACKS:
            emp_var = empirical_var_series(raw_dates, raw_returns, dates, N)
            missing = int(np.isnan(emp_var).sum())
            if missing:
                print(f"{fold} N={N}: {missing} day(s) lack {N} days of prior history, skipping N={N}")
                continue
            empirical_by_N[N] = fit_and_evaluate_offset(returns, emp_var, calib_n)

        results[fold] = {
            "test_start": fold_cfg["test_start"], "test_end": fold_cfg["test_end"],
            "calib_n": calib_n, "eval_n": len(returns) - calib_n,
            "parametric": parametric,
            "empirical": empirical_by_N,
        }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Raw results written to {RESULTS_PATH}\n")

    print("=" * 106)
    print(f"PRIMARY COMPARISON  (empirical lookback N={DEFAULT_N}, the standard 1-year historical-VaR default)")
    print(f"{'fold':22s} {'Param.cal%':>10s} {'Param.p':>10s} {'Emp.raw%':>9s} {'Emp.raw.p':>10s} "
          f"{'Emp.cal%':>9s} {'Emp.cal.p':>10s}")
    print("-" * 106)
    for fold, r in results.items():
        p = r["parametric"]["calibrated"]
        e = r["empirical"].get(DEFAULT_N)
        if e is None:
            print(f"{fold:22s}  -- N={DEFAULT_N} unavailable for this fold --")
            continue
        er, ec = e["raw"], e["calibrated"]
        p_p = f"{p['kupiec_p']:.4g}" if p['kupiec_p'] is not None else "nan"
        er_p = f"{er['kupiec_p']:.4g}" if er['kupiec_p'] is not None else "nan"
        ec_p = f"{ec['kupiec_p']:.4g}" if ec['kupiec_p'] is not None else "nan"
        print(f"{fold:22s} {p['coverage']*100:>9.2f}% {p_p:>10s} "
              f"{er['coverage']*100:>8.2f}% {er_p:>10s} "
              f"{ec['coverage']*100:>8.2f}% {ec_p:>10s}")

    print("\n" + "=" * 106)
    print("SENSITIVITY: empirical VaR (calibrated) pass/fail across lookback N")
    header = f"{'fold':22s}" + "".join(f"{'N=' + str(N):>20s}" for N in LOOKBACKS)
    print(header)
    print("-" * len(header))
    for fold, r in results.items():
        row = f"{fold:22s}"
        for N in LOOKBACKS:
            e = r["empirical"].get(N)
            if e is None:
                row += f"{'n/a':>20s}"
            else:
                pval = e["calibrated"]["kupiec_p"]
                cov = e["calibrated"]["coverage"]
                if pval is None:
                    row += f"{'fail p=nan':>20s}"
                else:
                    verdict = "PASS" if pval >= 0.05 else "fail"
                    row += f"{f'{verdict} {cov*100:.1f}% p={pval:.3g}':>20s}"
        print(row)

    def kupiec_fails(p):
        # None or NaN both mean "not a pass" - NaN happens at N=0 or N=T (e.g. zero
        # breaches at all), which is a real miscalibration, not a missing result. A plain
        # `p < 0.05` silently treats NaN as passing (NaN comparisons are always False in
        # Python), so it's checked explicitly rather than relied on to "just work."
        return p is None or (isinstance(p, float) and np.isnan(p)) or p < 0.05

    print("\n" + "=" * 106)
    print("INTERPRETATION CHECK (against the pre-registration printed above)")
    param_fail = sum(1 for r in results.values() if kupiec_fails(r["parametric"]["calibrated"]["kupiec_p"]))
    emp_fail_default = sum(1 for r in results.values() if DEFAULT_N in r["empirical"] and
                            kupiec_fails(r["empirical"][DEFAULT_N]["calibrated"]["kupiec_p"]))
    print(f"Parametric (calibrated) fails Kupiec in {param_fail}/5 folds.")
    print(f"Empirical, N={DEFAULT_N} (calibrated) fails Kupiec in {emp_fail_default}/5 folds.")

    print("\nSensitivity, fail counts by N (same kupiec_fails definition):")
    for N in LOOKBACKS:
        n_fail = sum(1 for r in results.values() if N in r["empirical"] and
                     kupiec_fails(r["empirical"][N]["calibrated"]["kupiec_p"]))
        n_avail = sum(1 for r in results.values() if N in r["empirical"])
        print(f"  N={N}: fails {n_fail}/{n_avail}")


if __name__ == "__main__":
    main()
