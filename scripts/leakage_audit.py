"""Audit contemporaneous-information leakage in the feature set, and test what it costs the
paper's conclusions.

Found while wiring up live monitoring: `log_close_ret` is bit-identical to the `log_return`
target. Close equals Adj Close on all 4705 rows, so log(Close_t / AdjClose_{t-1}) IS
log_return_t - max absolute difference 0.0. Feature 4 of 18 is target 1 of 5.

It is not confined to that pair. Sequences and targets are aligned on the same row
(make_sequences gives X[j] covering rows j..j+4, and y[j] is row j+4), so the input window
includes day t's own high, low and close while the target is day t's value. The task as built
is reconstruction of contemporaneous quantities, not forecasting.

This script does three things and writes nothing except a new JSON:

  1. AUDIT - classify every feature by the latest information it embeds, and measure how much
     of each target is linearly recoverable from the features.

  2. DECISIVE TEST - re-run the walk-forward calibration backtest using the RAW VaR_5 TARGET
     in place of model predictions, with the identical offset protocol and Kupiec code. If the
     pass/fail pattern matches the trained model's, the paper's central finding is a property
     of the target definition and the calibration procedure under regime shift, and the
     network is incidental to it.

  3. LEAKAGE-FREE FEATURE SET - define the lag-everything variant used by the retraining
     comparison, and confirm no member of it embeds day-t information.

PRE-REGISTERED (written before running):
  - If the raw-target backtest reproduces the model's fold-level pass/fail pattern, the
    calibration thesis survives the leakage, reframed as being about the target definition
    rather than a neural forecaster. Several existing observations already point this way:
    GFC came out bit-for-bit identical between the buggy and corrected models across four
    configurations, and the 10-seed distribution had a standard deviation as low as 0.08%.
  - If it does NOT reproduce, the model contributes materially after all, and the leakage
    then contaminates the central result rather than just the accuracy table.
"""
import json

import numpy as np
from sklearn.linear_model import LinearRegression

from mtl_var.backtest import kupiec_from_counts
from mtl_var.calibration import fixed_offset_calibration
from mtl_var.data import FEATURES, SEQUENCE_LENGTH, TARGETS, load_and_engineer_features
from mtl_var.pipeline import FOLDS

RESULTS_PATH = "results/leakage_audit_results.json"
WALK_FORWARD_PATH = "results/walk_forward_results.json"

# What is the latest bar each feature reads from? "t" means it embeds the current day, which
# is the same day the target describes.
FEATURE_INFORMATION_TIME = {
    "log_open_ret":    ("t", "day t's open"),
    "log_high_ret":    ("t", "day t's high - not known until the close"),
    "log_low_ret":     ("t", "day t's low - not known until the close"),
    "log_close_ret":   ("t", "day t's close - IDENTICAL to the log_return target"),
    "log_volume_lag1": ("t-1", "lagged"),
    "log_volume_lag5": ("t-5", "lagged"),
    "ret_lag1":        ("t-1", "lagged"),
    "ret_lag2":        ("t-2", "lagged"),
    "ret_lag3":        ("t-3", "lagged"),
    "ret_lag4":        ("t-4", "lagged"),
    "ret_lag5":        ("t-5", "lagged"),
    "yz_vol_lag1":     ("t-1", "lagged"),
    "VaR_5_lag1":      ("t-1", "lagged"),
    "drawdown":        ("t", "uses day t's close via cummax"),
    "ret_min20":       ("t", "rolling(20) window includes day t's return"),
    "ret_max20":       ("t", "rolling(20) window includes day t's return"),
    "CVaR_5_lag1":     ("t-1", "lagged"),
    "ret_skew20":      ("t", "rolling(20) window includes day t's return"),
}

CLEAN_FEATURES = [f for f, (when, _) in FEATURE_INFORMATION_TIME.items() if when != "t"]
CONTAMINATED_FEATURES = [f for f, (when, _) in FEATURE_INFORMATION_TIME.items() if when == "t"]


def audit_features(data):
    print("=" * 100)
    print("1. FEATURE AUDIT - what is the latest bar each feature reads?")
    print("=" * 100)
    for f in FEATURES:
        when, why = FEATURE_INFORMATION_TIME[f]
        flag = "LEAKS day t" if when == "t" else "ok"
        print(f"  {f:18s} {when:4s}  {flag:12s} {why}")
    print(f"\n  {len(CONTAMINATED_FEATURES)} of {len(FEATURES)} features embed day-t information: "
          f"{', '.join(CONTAMINATED_FEATURES)}")

    identical = float(np.abs(data["log_close_ret"] - data["log_return"]).max())
    print(f"\n  max |log_close_ret - log_return| = {identical}  "
          f"({'IDENTICAL' if identical == 0.0 else 'not identical'})")
    return identical


def recoverability(data):
    print("\n" + "=" * 100)
    print("2. HOW MUCH OF EACH TARGET IS A DETERMINISTIC FUNCTION OF THE FEATURES?")
    print("   in-sample linear R^2; 1.0 means the target is computable from the inputs")
    print("=" * 100)
    X_all = data[FEATURES].to_numpy(float)
    X_clean = data[CLEAN_FEATURES].to_numpy(float)

    rows = {}
    print(f"  {'target':12s} {'R^2 all 18 features':>22s} {'R^2 lag-only features':>24s}")
    for t in TARGETS:
        y = data[t].to_numpy(float)
        r2_all = LinearRegression().fit(X_all, y).score(X_all, y)
        r2_clean = LinearRegression().fit(X_clean, y).score(X_clean, y)
        rows[t] = {"r2_all_features": r2_all, "r2_lagged_only": r2_clean}
        print(f"  {t:12s} {r2_all:22.10f} {r2_clean:24.10f}")
    return rows


def raw_target_backtest(data):
    """The decisive test: the paper's calibration protocol applied to the raw VaR_5 TARGET,
    with no model in the loop at all."""
    print("\n" + "=" * 100)
    print("3. DECISIVE TEST - calibration backtest on the RAW TARGET, no model")
    print("   same folds, same half/half offset protocol, same Kupiec code")
    print("=" * 100)

    try:
        with open(WALK_FORWARD_PATH) as f:
            recorded = {r["fold"]: r for r in json.load(f) if "error" not in r}
    except FileNotFoundError:
        recorded = {}

    print(f"  {'fold':22s} {'raw-target cov':>15s} {'p':>12s} {'verdict':>9s} | "
          f"{'model cov':>10s} {'model p':>12s} {'verdict':>9s}  agree?")

    out = {}
    for fold_cfg in FOLDS:
        name = fold_cfg["name"]
        dates = data["observation_date"]
        mask = (dates >= fold_cfg["test_start"]) & (dates <= fold_cfg["test_end"])
        test = data.loc[mask].reset_index(drop=True)

        # Align exactly as build_fold_data does, so the evaluation window matches the model's.
        returns = test["log_return"].to_numpy(float)[SEQUENCE_LENGTH - 1:]
        var_target = test["VaR_5"].to_numpy(float)[SEQUENCE_LENGTH - 1:]

        res = fixed_offset_calibration(returns, var_target)
        cov = res["calibrated"]["coverage"]
        p = res["calibrated"]["kupiec_p"]
        verdict = "PASS" if (p is not None and not np.isnan(p) and p >= 0.05) else "FAIL"

        rec = recorded.get(name, {}).get("var_calibration", {}).get("calibrated", {})
        mcov, mp = rec.get("coverage"), rec.get("kupiec_p")
        mverdict = ("PASS" if (mp is not None and not np.isnan(mp) and mp >= 0.05) else "FAIL") if mp is not None else "?"
        agree = "yes" if verdict == mverdict else "NO"

        print(f"  {name:22s} {cov*100:14.2f}% {p:12.6g} {verdict:>9s} | "
              f"{(mcov or 0)*100:9.2f}% {mp if mp is not None else float('nan'):12.6g} {mverdict:>9s}  {agree}")

        out[name] = {"raw_target": {"coverage": cov, "kupiec_p": p, "verdict": verdict},
                     "trained_model": {"coverage": mcov, "kupiec_p": mp, "verdict": mverdict},
                     "agree": verdict == mverdict}
    return out


def main():
    print(__doc__)
    data = load_and_engineer_features("SPX500-18Y_scaled_y.csv")

    identical = audit_features(data)
    r2 = recoverability(data)
    backtest = raw_target_backtest(data)

    agreements = sum(1 for v in backtest.values() if v["agree"])
    print("\n" + "=" * 100)
    print(f"RESULT: raw target and trained model agree on {agreements}/{len(backtest)} folds.")
    if agreements == len(backtest):
        print("  Pre-registered reading: the calibration finding is a property of the TARGET")
        print("  DEFINITION and the offset procedure under regime shift. The network is")
        print("  incidental to it, which is consistent with GFC being bit-identical across")
        print("  models and with the near-zero seed variance. The thesis survives, reframed.")
    else:
        print("  Pre-registered reading: the model contributes materially, so the leakage")
        print("  reaches the central result and not only the accuracy table.")
    print("=" * 100)

    results = {
        "log_close_ret_equals_log_return": identical == 0.0,
        "max_abs_difference": identical,
        "feature_information_time": {f: FEATURE_INFORMATION_TIME[f][0] for f in FEATURES},
        "contaminated_features": CONTAMINATED_FEATURES,
        "clean_features": CLEAN_FEATURES,
        "target_recoverability_r2": r2,
        "raw_target_backtest": backtest,
        "folds_in_agreement": agreements,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
