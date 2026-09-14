"""Recalibration experiments on the leakage-free model: the ex-ante version of the paper's tables.

Rolling recalibration (RESULTS.md section 4), the volatility-ratio monitor (section 5) and the
calibration-window sweep (section 8) were computed on the original model's predictions, and that
model was trained on leaky features (section 13). model_free_recalibration.py re-ran them on the raw
VaR_5 target, which removes the model but not every contemporaneous term: the target is EWMA_10 of
returns through day t minus 1.2 x a volatility term that includes day t, so VaR_t carries roughly 18%
of r_t - the very return it is tested against. That makes the raw target a check on whether a model
is needed to reach the conclusions, not an ex-ante VaR backtest.

The leakage-free model is the ex-ante version: every forecast of day t uses only information through
day t-1. This re-runs the same three analyses on its cached predictions, produced by
`mtl-var walk-forward --feature-set leakage_free`, side by side with the original model and the raw
target, through the same `rolling_calibration_analysis`.

PRE-REGISTERED. Written while the leakage-free walk-forward was still running, before any of its
recalibration numbers existed. It was written after the raw-target results were seen, which is stated
rather than hidden; the criteria below are the raw-target script's three, carried over unchanged:

  1. Rolling recalibration (L=125, R=21): GFC fails Kupiec; EU crisis, 2015 selloff, COVID and 2022
     bear pass.
  2. Window sweep: L=40 rescues GFC, AND some fold that passes at L=125 fails at L=40 - so no single
     static L passes every fold.
  3. Volatility-ratio monitor: the breach rate on elevated readings exceeds the rate on normal readings
     in both shock folds (GFC, COVID), and is lower in both grind folds (EU crisis, 2015 selloff).

Gate before any of that: the leakage-free walk-forward's fixed-offset results must be bit-identical to
results/leakage_free_retrain_results.json. If they are not, these predictions are not the model
section 13 evaluated, and the script stops without analysing them.

No training. Writes results/recalibration_leakage_free_results.json.
"""
import json
import math
import os
import sys

import numpy as np

from model_free_recalibration import GRIND, L_VALUES, SHOCK, analyse, fmt, raw_target_arrays
from mtl_var.data import TARGETS, load_and_engineer_features
from mtl_var.pipeline import FOLDS

CSV = "SPX500-18Y_scaled_y.csv"
PREDICTION_CACHES = {
    "leakage_free_model": "results/predictions_leakage_free",
    "original_model": "results/predictions",
}
LEAKAGE_FREE_WALK_FORWARD = "results/walk_forward_results_leakage_free.json"
RECORDED_LEAKAGE_FREE = "results/leakage_free_retrain_results.json"
RESULTS_PATH = "results/recalibration_leakage_free_results.json"
SOURCE_ORDER = ("leakage_free_model", "original_model", "raw_target")


def cached_arrays(directory, fold):
    z = np.load(os.path.join(directory, f"{fold['name']}.npz"), allow_pickle=True)
    return z["dates"].astype("datetime64[D]"), z["returns"], z["var_pred"], int(z["calib_n"])


def same(x, y):
    both_nan = isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y)
    return both_nan or x == y


def parity_check():
    with open(LEAKAGE_FREE_WALK_FORWARD) as f:
        walk = {r["fold"]: r for r in json.load(f) if "error" not in r}
    with open(RECORDED_LEAKAGE_FREE) as f:
        recorded = {r["fold"]: r for r in json.load(f)}

    failed = []
    for fold in FOLDS:
        name = fold["name"]
        a, b = walk.get(name), recorded.get(name)
        if a is None or b is None:
            print(f"  {name:20s} MISSING from {'walk-forward' if a is None else 'recorded run'}")
            failed.append(name)
            continue
        pairs = [("epochs_run", a["epochs_run"], b["epochs_run"]),
                 ("best_val_loss", a["best_val_loss"], b["best_val_loss"]),
                 ("offset", a["var_calibration"]["offset"], b["var_calibration"]["offset"])]
        pairs += [(f"mae[{t}]", a["accuracy"][t]["mae"], b["accuracy"][t]["mae"]) for t in TARGETS]
        pairs += [(f"{blk}.{k}", a["var_calibration"][blk][k], b["var_calibration"][blk][k])
                  for blk in ("uncalibrated", "calibrated") for k in ("breaches", "coverage", "kupiec_p")]
        bad = [label for label, x, y in pairs if not same(x, y)]
        print(f"  {name:20s} {'bit-identical (' + str(len(pairs)) + ' values)' if not bad else 'MISMATCH: ' + ', '.join(bad)}")
        if bad:
            failed.append(name)
    return failed


def evaluate(src):
    """The three pre-registered checks, on one source's results."""
    expected = {"fold1_gfc": "FAIL", "fold2_eu_crisis": "PASS", "fold3_2015_selloff": "PASS",
                "fold4_covid": "PASS", "fold5_2022_bear": "PASS"}
    rolling = {name: src[name]["125"]["verdict"] for name in expected}

    gfc_rescued = src["fold1_gfc"]["40"]["verdict"] == "PASS"
    regressed = [n for n in src if src[n]["125"]["verdict"] == "PASS" and src[n]["40"]["verdict"] == "FAIL"]
    every_fold = [L for L in L_VALUES if all(src[n][str(L)] is not None and src[n][str(L)]["verdict"] == "PASS"
                                             for n in src)]

    def elevated_above_normal(name):
        mon = src[name]["125"]["monitor"]
        e, n = mon["elevated_breach_rate"], mon["normal_breach_rate"]
        return None if e is None or n is None else e > n

    shock = {n: elevated_above_normal(n) for n in SHOCK}
    grind = {n: elevated_above_normal(n) for n in GRIND}
    return {
        "1_rolling_recalibration_pattern": rolling == expected,
        "1_rolling_verdicts": rolling,
        "2_no_single_static_L": gfc_rescued and bool(regressed),
        "2_gfc_rescued_at_L40": gfc_rescued,
        "2_regressed_at_L40": regressed,
        "2_window_lengths_passing_every_fold": every_fold,
        "3_monitor_shock_vs_grind": all(v is True for v in shock.values()) and all(v is False for v in grind.values()),
        "3_shock_elevated_above_normal": shock,
        "3_grind_elevated_above_normal": grind,
    }


def main():
    print(__doc__)
    print("=" * 110)
    print("GATE - leakage-free walk-forward vs the recorded section 13 run")
    print("=" * 110)
    failed = parity_check()
    if failed:
        print(f"\nPARITY FAILED for {failed}: these predictions are not the model section 13 evaluated. Stopping.")
        sys.exit(1)

    data = load_and_engineer_features(CSV)
    results = {}
    for fold in FOLDS:
        arrays = {src: cached_arrays(path, fold) for src, path in PREDICTION_CACHES.items()}
        arrays["raw_target"] = raw_target_arrays(data, fold)
        results[fold["name"]] = {src: {str(L): analyse(*arr, L) for L in L_VALUES} for src, arr in arrays.items()}

    print("\n" + "=" * 110)
    print("WINDOW SWEEP (L=125 is the rolling-recalibration baseline)")
    print("=" * 110)
    for fold in FOLDS:
        name = fold["name"]
        print(f"\n{name}")
        for L in L_VALUES:
            cells = [results[name][src][str(L)] for src in SOURCE_ORDER]
            print(f"  L={L:<4d} leakage-free {fmt(cells[0])} | original {fmt(cells[1])} | raw target {fmt(cells[2])}")

    print("\n" + "=" * 110)
    print("VOLATILITY-RATIO MONITOR at L=125 - breach rate on elevated vs normal readings, lead vs baseline")
    print("=" * 110)
    show = lambda x: "  n/a" if x is None else f"{x * 100:5.1f}%"  # noqa: E731
    for name in SHOCK + GRIND + ["fold5_2022_bear"]:
        for src in SOURCE_ORDER:
            mon = results[name][src]["125"]["monitor"]
            print(f"  {name:20s} {src:20s} elevated {show(mon['elevated_breach_rate'])}  normal "
                  f"{show(mon['normal_breach_rate'])}  lead {show(mon['breach_preceded_by_elevated_rate'])} / base "
                  f"{show(mon['baseline_preceded_by_elevated_rate'])}")

    per_source = {src: evaluate({n: results[n][src] for n in results}) for src in SOURCE_ORDER}
    lf = per_source["leakage_free_model"]

    print("\n" + "=" * 110)
    print("PRE-REGISTERED CHECKS - on the leakage-free model")
    print("=" * 110)
    print(f"  1. rolling recalibration: GFC fails, the other four pass: {'PASS' if lf['1_rolling_recalibration_pattern'] else 'FAIL'}"
          f"   {lf['1_rolling_verdicts']}")
    print(f"  2. L=40 rescues GFC ({lf['2_gfc_rescued_at_L40']}) and regresses a fold passing at L=125 "
          f"({lf['2_regressed_at_L40']}): {'PASS' if lf['2_no_single_static_L'] else 'FAIL'}")
    print(f"     window lengths at which every fold passes: {lf['2_window_lengths_passing_every_fold'] or 'none'}")
    print(f"  3. monitor: elevated > normal in shock folds {lf['3_shock_elevated_above_normal']}, "
          f"< normal in grind folds {lf['3_grind_elevated_above_normal']}: "
          f"{'PASS' if lf['3_monitor_shock_vs_grind'] else 'FAIL'}")

    print("\n  same checks on the other sources, for comparison:")
    for src in ("original_model", "raw_target"):
        c = per_source[src]
        print(f"    {src:16s} 1={'PASS' if c['1_rolling_recalibration_pattern'] else 'FAIL'}  "
              f"2={'PASS' if c['2_no_single_static_L'] else 'FAIL'} (regressed {c['2_regressed_at_L40']})  "
              f"3={'PASS' if c['3_monitor_shock_vs_grind'] else 'FAIL'}")

    def agreement(a, b):
        cells = [(n, L) for n in results for L in L_VALUES if results[n][a][str(L)] and results[n][b][str(L)]]
        same_verdict = [c for c in cells if results[c[0]][a][str(c[1])]["verdict"] == results[c[0]][b][str(c[1])]["verdict"]]
        differing = [f"{n}@L={L}" for n, L in cells if (n, L) not in same_verdict]
        return len(same_verdict), len(cells), differing

    print("\n  window-sweep verdict agreement, cell by cell (5 folds x 5 window lengths):")
    agree = {}
    for a, b in (("leakage_free_model", "original_model"), ("leakage_free_model", "raw_target"),
                 ("original_model", "raw_target")):
        k, n, diff = agreement(a, b)
        agree[f"{a}_vs_{b}"] = {"agree": k, "cells": n, "differing": diff}
        print(f"    {a} vs {b}: {k}/{n}" + (f"   differ at {diff}" if diff else ""))

    with open(RESULTS_PATH, "w") as f:
        json.dump({"preregistered_checks_leakage_free": lf, "checks_by_source": per_source,
                   "verdict_agreement": agree, "results": results}, f, indent=2)
    print(f"\nwrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
