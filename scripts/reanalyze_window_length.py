"""
Window length as its own properly-scoped experiment (RESULTS.md section 7 / "what to do
next" item 4). Cadence (event-triggered refresh) and threshold (elevated_ratio) were both
tried on 2008 GFC and eliminated by evidence. Window length (L) is the one parameter that
was held fixed at 125 days through every prior experiment. This tests it properly:

  - several L values, not one guess
  - the reactivity-vs-noise tradeoff measured directly, not just a pass/fail number
  - a regression check on the folds rolling recalibration already fixed at L=125
  - refresh cadence (R=21) and monitor window (S=10) held fixed so L is isolated

Runs entirely from predictions/*.npz - no retraining.

PRE-REGISTERED SUCCESS CRITERION (decided before running, do not revise after seeing results):
  RESCUE  = 2008 GFC flips from failing Kupiec (p < 0.05) to passing (p >= 0.05) at some L.
  REGRESSION = any of {2011 EU crisis, 2015 selloff, 2020 COVID} - all passing at L=125 -
               drops below p = 0.05 at that same L.
  A result only counts as a genuine fix if it rescues 2008 AND causes zero regressions.
"""
import os

import numpy as np

from walk_forward import rolling_calibration_analysis

L_VALUES = [40, 60, 90, 125, 175]
R = 21  # refresh cadence held fixed - isolate window length as the only changed variable
S = 10  # live-vol window for the monitor signal, unused here (trigger_on_elevated=False)

BASELINE_PASS = {
    # fold: was it passing at L=125 (scheduled-only rolling)? used for the regression check
    "fold1_gfc": False,
    "fold2_eu_crisis": True,
    "fold3_2015_selloff": True,
    "fold4_covid": True,
    "fold5_2022_bear": False,  # borderline-fail at L=125, not part of the regression set
}

FOLD_ORDER = ["fold1_gfc", "fold2_eu_crisis", "fold3_2015_selloff", "fold4_covid", "fold5_2022_bear"]


def reactivity_and_noise(returns, idx, offset_series, refresh_days):
    """
    reactivity = |correlation| between the offset and a trailing 10-day realized-vol series
                 over the whole walk. Higher = the offset tracks current volatility more
                 closely (a shorter window should score higher here).
    noise      = mean absolute change in the offset between consecutive refreshes. Higher =
                 the offset itself swings around more from refresh to refresh - conflates
                 genuine adaptation with pure small-sample estimation noise; that conflation
                 is exactly the tradeoff being measured, not a flaw to fix.
    """
    trailing_vol = np.array([returns[t - 10:t].std() if t >= 10 else np.nan for t in idx])
    offset_arr = np.array(offset_series, dtype=float)
    valid = ~np.isnan(trailing_vol)
    reactivity = float(np.corrcoef(offset_arr[valid], trailing_vol[valid])[0, 1]) if valid.sum() > 2 else float("nan")

    if len(refresh_days) < 2:
        noise = float("nan")
    else:
        idx_pos = {t: i for i, t in enumerate(idx)}
        refresh_offsets = [offset_arr[idx_pos[t]] for t in refresh_days if t in idx_pos]
        diffs = np.diff(refresh_offsets)
        noise = float(np.mean(np.abs(diffs))) if len(diffs) else float("nan")

    return reactivity, noise


def main():
    print("PRE-REGISTERED: rescue = fold1_gfc flips to p>=0.05; regression = any of "
          "{fold2_eu_crisis, fold3_2015_selloff, fold4_covid} drops below p=0.05.\n")

    print(f"{'fold':22s} {'L':>5s} {'cov%':>7s} {'kupiec_p':>10s} {'reactivity':>11s} "
          f"{'noise(offset)':>14s} {'n_refresh':>10s}")
    print("-" * 90)

    all_rows = {}
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

        rows = []
        for L in L_VALUES:
            r = rolling_calibration_analysis(
                dates, returns, var_pred, eval_start_idx=calib_n,
                L=L, R=R, S=S, trigger_on_elevated=False,
            )
            if r is None:
                print(f"{fold:22s} {L:>4d}d   -- fold too short for this L, skipped --")
                continue
            es = r["eval_slice_only"]
            idx = np.arange(L, L + len(r["timeline"]["offset"]))
            reactivity, noise = reactivity_and_noise(
                returns, idx, r["timeline"]["offset"], r["refresh_days"]
            )
            p_str = f"{es['kupiec_p']:.4g}" if es["kupiec_p"] is not None else "nan"
            print(f"{fold:22s} {L:>4d}d {es['coverage']*100:>6.2f}% {p_str:>10s} "
                  f"{reactivity:>11.3f} {noise:>14.5f} {len(r['refresh_days']):>10d}")
            rows.append({"L": L, "coverage": es["coverage"], "kupiec_p": es["kupiec_p"],
                         "reactivity": reactivity, "noise": noise})
        all_rows[fold] = rows
        print()

    print("=" * 90)
    print("PRE-REGISTERED CHECK")
    rescues = []
    for row in all_rows.get("fold1_gfc", []):
        if row["kupiec_p"] is not None and row["kupiec_p"] >= 0.05:
            rescues.append(row["L"])

    if not rescues:
        print("No L value rescues fold1_gfc. Result: window length ELIMINATED as a fix for 2008,")
        print("same as cadence and threshold. Reporting the reactivity/noise tradeoff as the")
        print("finding in its own right, not as a search for a value that didn't work.")
    else:
        print(f"fold1_gfc rescued at L in {rescues}. Checking regressions on the passing folds...")
        for L in rescues:
            regressed = []
            for fold in ["fold2_eu_crisis", "fold3_2015_selloff", "fold4_covid"]:
                row = next((rr for rr in all_rows[fold] if rr["L"] == L), None)
                if row is None or row["kupiec_p"] is None or row["kupiec_p"] < 0.05:
                    regressed.append(fold)
            if regressed:
                print(f"  L={L}: rescues fold1_gfc but REGRESSES {regressed} - not a clean fix.")
            else:
                print(f"  L={L}: rescues fold1_gfc with NO regressions - genuine fix.")


if __name__ == "__main__":
    main()
