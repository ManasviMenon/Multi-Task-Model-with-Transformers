"""Do the recalibration experiments survive with no model at all?

RESULTS.md section 13 showed the fixed-offset calibration verdicts are model-independent: the raw
VaR_5 target, run through the identical offset protocol and Kupiec code with no network in the
loop, reproduced all five fold verdicts, as did the leakage-free retrain. The experiments built on
top of that baseline were never re-checked. Rolling recalibration (section 4), the volatility-ratio
monitor (section 5) and the calibration-window sweep (section 8) were all computed on the original
model's cached predictions, and that model was trained on leaky features.

This re-runs each of them on the raw target instead, through the same `rolling_calibration_analysis`,
and side by side with the original model's numbers recomputed from its prediction cache - so the
tables headed for the paper can state whether their conclusions depend on the model.

PRE-REGISTERED, written before running. A conclusion survives if its verdict pattern reproduces on
the raw target. Magnitudes are expected to differ from the model's; only the patterns are tested,
and any disagreement is reported as one:

  1. Rolling recalibration (L=125, R=21): GFC still fails Kupiec; EU crisis, 2015 selloff, COVID and
     2022 bear all pass.
  2. Window sweep: L=40 rescues GFC, AND some fold that passes at L=125 fails at L=40 - i.e. no single
     static L passes every fold.
  3. Volatility-ratio monitor: in both shock folds (GFC, COVID) the breach rate on elevated readings
     exceeds the rate on normal readings; in both grind folds (EU crisis, 2015 selloff) it is lower.

No training. Writes results/model_free_recalibration_results.json.

NOTE, added after running. The raw target is not an ex-ante VaR forecast: VaR_5 is EWMA_10 of returns
through day t minus 1.2 x a volatility term that includes day t, so the value tested against r_t carries
roughly 18% of r_t itself. This is therefore a check on whether a model is needed to reach these
conclusions, not a leakage-free backtest; the ex-ante check is recalibration_leakage_free.py, on the
leakage-free model's predictions. Outcome here: checks 1 and 2 passed; check 3 FAILED, because GFC's
breach rate on elevated readings (5.7%) was not above its rate on normal readings (6.1%); and 23 of 25
window-sweep cells agreed with the original model, the two exceptions being 2015 selloff at L=40 and
L=175, both marginal failures in the model (p=0.042, p=0.046) that pass on the raw target.
"""
import json
import os

import numpy as np

from mtl_var.calibration import rolling_calibration_analysis
from mtl_var.data import SEQUENCE_LENGTH, load_and_engineer_features
from mtl_var.pipeline import FOLDS

CSV = "SPX500-18Y_scaled_y.csv"
PREDICTIONS_DIR = "results/predictions"
RESULTS_PATH = "results/model_free_recalibration_results.json"
L_VALUES = [40, 60, 90, 125, 175]
SHOCK = ["fold1_gfc", "fold4_covid"]
GRIND = ["fold2_eu_crisis", "fold3_2015_selloff"]


def verdict(p):
    """A NaN p-value is the zero-or-all-breaches boundary, which the research records as a FAIL."""
    return "FAIL" if p is None or np.isnan(p) or p < 0.05 else "PASS"


def raw_target_arrays(data, fold):
    d = data["observation_date"]
    test = data.loc[(d >= fold["test_start"]) & (d <= fold["test_end"])].reset_index(drop=True)
    dates = test["observation_date"].to_numpy(dtype="datetime64[D]")[SEQUENCE_LENGTH - 1:]
    returns = test["log_return"].to_numpy(float)[SEQUENCE_LENGTH - 1:]
    var = test["VaR_5"].to_numpy(float)[SEQUENCE_LENGTH - 1:]
    return dates, returns, var, len(var) // 2


def model_arrays(fold):
    z = np.load(os.path.join(PREDICTIONS_DIR, f"{fold['name']}.npz"), allow_pickle=True)
    return z["dates"].astype("datetime64[D]"), z["returns"], z["var_pred"], int(z["calib_n"])


def analyse(dates, returns, var, calib_n, L):
    r = rolling_calibration_analysis(dates, returns, var, eval_start_idx=calib_n, L=L, trigger_on_elevated=False)
    if r is None:
        return None
    es, mon = r["eval_slice_only"], r["monitor"]
    p = es["kupiec_p"]
    return {
        "coverage": es["coverage"], "kupiec_p": None if p is None or np.isnan(p) else float(p),
        "breaches": es["breaches"], "n": es["n"], "verdict": verdict(p),
        "monitor": {k: mon[k] for k in ("elevated_breach_rate", "normal_breach_rate",
                                        "breach_preceded_by_elevated_rate", "baseline_preceded_by_elevated_rate")},
    }


def fmt(row):
    if row is None:
        return "     -- too short --"
    p = "n/a" if row["kupiec_p"] is None else f"{row['kupiec_p']:.3g}"
    return f"{row['coverage'] * 100:5.2f}% p={p:>9s} {row['verdict']}"


def main():
    print(__doc__)
    data = load_and_engineer_features(CSV)

    results = {}
    for fold in FOLDS:
        name = fold["name"]
        sources = {"raw_target": raw_target_arrays(data, fold), "original_model": model_arrays(fold)}
        results[name] = {src: {str(L): analyse(*arrays, L) for L in L_VALUES} for src, arrays in sources.items()}

    print("=" * 100)
    print("WINDOW SWEEP (L=125 is the rolling-recalibration baseline) - original model vs raw target")
    print("=" * 100)
    for fold in FOLDS:
        name = fold["name"]
        print(f"\n{name}")
        for L in L_VALUES:
            m, r = results[name]["original_model"][str(L)], results[name]["raw_target"][str(L)]
            agree = "" if m is None or r is None else ("agree" if m["verdict"] == r["verdict"] else "DISAGREE")
            print(f"  L={L:<4d} model {fmt(m)}   |   raw target {fmt(r)}   {agree}")

    print("\n" + "=" * 100)
    print("VOLATILITY-RATIO MONITOR at L=125 - breach rate on elevated vs normal readings")
    print("=" * 100)
    for name in SHOCK + GRIND + ["fold5_2022_bear"]:
        for src in ("original_model", "raw_target"):
            mon = results[name][src]["125"]["monitor"]
            e, n = mon["elevated_breach_rate"], mon["normal_breach_rate"]
            lead, base = mon["breach_preceded_by_elevated_rate"], mon["baseline_preceded_by_elevated_rate"]
            show = lambda x: "  n/a" if x is None else f"{x * 100:5.1f}%"  # noqa: E731
            print(f"  {name:20s} {src:15s} elevated {show(e)}  normal {show(n)}  lead {show(lead)} / base {show(base)}")

    print("\n" + "=" * 100)
    print("PRE-REGISTERED CHECKS - on the raw target")
    print("=" * 100)
    raw = {name: results[name]["raw_target"] for name in results}
    checks = {}

    expected = {"fold1_gfc": "FAIL", "fold2_eu_crisis": "PASS", "fold3_2015_selloff": "PASS",
                "fold4_covid": "PASS", "fold5_2022_bear": "PASS"}
    got = {name: raw[name]["125"]["verdict"] for name in expected}
    checks["1_rolling_recalibration_pattern"] = got == expected
    print(f"  1. rolling recalibration: GFC fails, the other four pass: "
          f"{'PASS' if checks['1_rolling_recalibration_pattern'] else 'FAIL'}   {got}")

    gfc_rescued = raw["fold1_gfc"]["40"]["verdict"] == "PASS"
    regressed = [n for n in raw if raw[n]["125"]["verdict"] == "PASS" and raw[n]["40"]["verdict"] == "FAIL"]
    checks["2_no_single_static_L"] = gfc_rescued and bool(regressed)
    print(f"  2. L=40 rescues GFC ({gfc_rescued}) and regresses a fold that passes at L=125 ({regressed}): "
          f"{'PASS' if checks['2_no_single_static_L'] else 'FAIL'}")
    all_pass = [L for L in L_VALUES if all(raw[n][str(L)] is not None and raw[n][str(L)]["verdict"] == "PASS" for n in raw)]
    print(f"     window lengths at which every fold passes on the raw target: {all_pass or 'none'}")
    checks["2_window_lengths_passing_every_fold"] = all_pass

    def direction(name):
        mon = raw[name]["125"]["monitor"]
        e, n = mon["elevated_breach_rate"], mon["normal_breach_rate"]
        return None if e is None or n is None else e > n

    shock_ok = all(direction(n) is True for n in SHOCK)
    grind_ok = all(direction(n) is False for n in GRIND)
    checks["3_monitor_shock_vs_grind"] = shock_ok and grind_ok
    print(f"  3. monitor: elevated > normal in both shock folds ({shock_ok}), < normal in both grind folds "
          f"({grind_ok}): {'PASS' if checks['3_monitor_shock_vs_grind'] else 'FAIL'}")

    agreements = sum(
        1 for n in results for L in L_VALUES
        if results[n]["original_model"][str(L)] and results[n]["raw_target"][str(L)]
        and results[n]["original_model"][str(L)]["verdict"] == results[n]["raw_target"][str(L)]["verdict"]
    )
    total = sum(1 for n in results for L in L_VALUES
                if results[n]["original_model"][str(L)] and results[n]["raw_target"][str(L)])
    checks["verdict_agreement_cells"] = f"{agreements}/{total}"
    print(f"\n  cell-by-cell verdict agreement, original model vs raw target, all folds x all L: {agreements}/{total}")

    with open(RESULTS_PATH, "w") as f:
        json.dump({"preregistered_checks": checks, "results": results}, f, indent=2)
    print(f"\nwrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
