"""
Post-hoc only: does tightening the elevated-ratio threshold make event-triggered
recalibration actually rescue folds, or does it just fire less often without changing
anything? Uses the cached predictions in predictions/*.npz - no retraining.

Pre-registered success criterion (decided before running this):
  - a fold that was failing Kupiec under scheduled-only rolling now passes (p > 0.05)
  - AND no fold that already passed under scheduled-only rolling regresses

Tests elevated_ratio in {1.5 (baseline), 2.0, 2.5} and reports, per fold per threshold,
the firing rate (how often "elevated" actually triggers) alongside coverage/Kupiec -
so a threshold that still fires constantly in a fold is visible as exactly that, not
silently credited with a result it didn't earn.
"""
import glob
import os

import numpy as np

from walk_forward import rolling_calibration_analysis

THRESHOLDS = [1.5, 2.0, 2.5]

SCHEDULED_ONLY_BASELINE = {
    # fold: (coverage, kupiec_p) from the scheduled-only rolling run, for the regression check
    "fold1_gfc": (0.0080, 0.000163),
    "fold2_eu_crisis": (0.0480, 0.858),
    "fold3_2015_selloff": (0.0399, 0.352),
    "fold4_covid": (0.0637, 0.337),
    "fold5_2022_bear": (0.0164, 0.049),
}

FOLD_ORDER = ["fold1_gfc", "fold2_eu_crisis", "fold3_2015_selloff", "fold4_covid", "fold5_2022_bear"]


def main():
    print(f"{'fold':22s} {'thresh':>7s} {'fire%':>7s} {'cov%':>7s} {'kupiec_p':>12s} "
          f"{'sched':>6s} {'trig':>6s}  vs scheduled-only baseline")
    print("-" * 100)

    results = {}
    for fold in FOLD_ORDER:
        path = os.path.join("results/predictions", f"{fold}.npz")
        if not os.path.exists(path):
            print(f"{fold}: no cached predictions, skipping")
            continue
        d = np.load(path, allow_pickle=True)
        dates = d["dates"].astype("datetime64[D]")
        returns = d["returns"]
        var_pred = d["var_pred"]
        calib_n = int(d["calib_n"])

        base_cov, base_p = SCHEDULED_ONLY_BASELINE[fold]
        results[fold] = []

        for thr in THRESHOLDS:
            r = rolling_calibration_analysis(
                dates, returns, var_pred, eval_start_idx=calib_n,
                elevated_ratio=thr, trigger_on_elevated=True,
            )
            mon = r["monitor"]
            fire_rate = mon["n_elevated_days"] / mon["n_valid_days"] if mon["n_valid_days"] else float("nan")
            es = r["eval_slice_only"]
            rf = r["refreshes"]

            was_failing = base_p < 0.05
            now_passing = es["kupiec_p"] is not None and es["kupiec_p"] >= 0.05
            regressed = (not was_failing) and (es["kupiec_p"] is None or es["kupiec_p"] < 0.05)

            flag = ""
            if was_failing and now_passing:
                flag = "  <-- RESCUED"
            elif regressed:
                flag = "  <-- REGRESSED (was passing)"

            print(f"{fold:22s} {thr:>6.1f}x {fire_rate*100:>6.1f}% {es['coverage']*100:>6.2f}% "
                  f"{es['kupiec_p']:>12.4g} {rf['scheduled']:>6d} {rf['triggered']:>6d}"
                  f"  (baseline {base_cov*100:.2f}%, p={base_p:.4g}){flag}")

            results[fold].append({
                "threshold": thr, "fire_rate": fire_rate, "coverage": es["coverage"],
                "kupiec_p": es["kupiec_p"], "scheduled": rf["scheduled"], "triggered": rf["triggered"],
            })
        print()

    print("=" * 100)
    print("Pre-registered check: any fold RESCUED (failing -> passing) at any threshold, with no REGRESSED fold?")
    rescued = []
    regressed_any = []
    for fold in FOLD_ORDER:
        base_cov, base_p = SCHEDULED_ONLY_BASELINE[fold]
        was_failing = base_p < 0.05
        for row in results.get(fold, []):
            now_passing = row["kupiec_p"] is not None and row["kupiec_p"] >= 0.05
            if was_failing and now_passing:
                rescued.append((fold, row["threshold"], row["fire_rate"]))
            if (not was_failing) and (row["kupiec_p"] is None or row["kupiec_p"] < 0.05):
                regressed_any.append((fold, row["threshold"]))
    print("Rescued:", rescued if rescued else "none")
    print("Regressed:", regressed_any if regressed_any else "none")


if __name__ == "__main__":
    main()
