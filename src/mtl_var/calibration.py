"""Post-hoc VaR calibration: the frozen offset, and rolling/event-triggered recalibration.

`fixed_offset_calibration` is the split-test-in-half procedure that was inline in `run_fold`;
`rolling_calibration_analysis` is moved from scripts/walk_forward.py with one change - the
per-day volatility-ratio arithmetic now calls `monitor.py` instead of repeating itself
inline, so there is a single implementation of that signal. Output is unchanged
(tests/test_refactor_parity.py checks this against the cached fold predictions).
"""
import numpy as np

from .backtest import kupiec_from_counts, kupiec_test
from .monitor import elevated_mask, is_elevated, live_volatility, vol_ratio


def fixed_offset_calibration(returns, var_pred):
    """Split the test window in half chronologically, fit a 5th-percentile-of-residuals offset
    on the first half, evaluate the untouched second half before and after applying it.

    This is the procedure behind every "fixed offset" number in RESULTS.md.
    """
    calib_n = len(var_pred) // 2

    returns_calib, returns_eval = returns[:calib_n], returns[calib_n:]
    var_calib, var_eval = var_pred[:calib_n], var_pred[calib_n:]

    residuals_calib = returns_calib - var_calib
    offset = float(np.quantile(residuals_calib, 0.05))
    var_eval_calibrated = var_eval + offset

    N0, T0, p0 = kupiec_test(returns_eval, var_eval)
    N1, T1, p1 = kupiec_test(returns_eval, var_eval_calibrated)

    return {
        "calib_n": calib_n,
        "eval_n": T0,
        "offset": offset,
        "uncalibrated": {"breaches": N0, "coverage": N0 / T0 if T0 else None, "kupiec_p": p0},
        "calibrated": {"breaches": N1, "coverage": N1 / T1 if T1 else None, "kupiec_p": p1},
    }


def rolling_calibration_analysis(dates, returns, var_pred, eval_start_idx,
                                  L=125, R=21, S=10, elevated_ratio=1.5, lead_days=5,
                                  trigger_on_elevated=False, min_trigger_gap=5):
    """
    Walk day-by-day through a fold's test period with a causal, periodically-refreshed
    VaR offset, instead of one offset frozen at the start.

    L = trailing window (days) the offset is fit on
    R = refresh cadence (days) - how often the offset is recomputed on schedule
    S = short "live" window (days) used for the volatility-ratio monitor signal
    eval_start_idx = index marking where the fixed-offset approach's eval half began,
                      so rolling coverage can also be reported on that identical slice
    trigger_on_elevated = if True, an "elevated" vol-ratio reading forces an immediate
                      recalibration instead of waiting for the next scheduled refresh
                      (still gated by min_trigger_gap so it can't refresh every single
                      day during a sustained elevated stretch)

    Everything here is causal: the offset active on day t was fit only on
    returns[t-L:t], and the monitor's vol_ratio on day t only looks backward. A
    trigger check uses the ratio computed from the *currently active* offset window,
    so it can never see into the future either.
    """
    n = len(returns)
    if n < L + R:
        return None

    offset = None
    offset_window_vol = None
    last_refresh = None

    idx, breach, ratio_series, refresh_kind, offset_series, refresh_days = [], [], [], [], [], []
    n_scheduled_refreshes, n_triggered_refreshes = 0, 0

    for t in range(L, n):
        live_vol = live_volatility(returns, t, S)
        ratio = vol_ratio(live_vol, offset_window_vol)
        is_elevated_now = is_elevated(ratio, elevated_ratio)

        scheduled_due = last_refresh is None or t - last_refresh >= R
        gap_ok = last_refresh is None or t - last_refresh >= min_trigger_gap
        triggered_due = trigger_on_elevated and is_elevated_now and gap_ok and not scheduled_due

        kind = None
        if scheduled_due:
            kind = "scheduled"
            n_scheduled_refreshes += 1
        elif triggered_due:
            kind = "triggered"
            n_triggered_refreshes += 1

        if kind is not None:
            window_returns = returns[t - L:t]
            window_var = var_pred[t - L:t]
            offset = float(np.quantile(window_returns - window_var, 0.05))
            offset_window_vol = float(window_returns.std())
            last_refresh = t
            refresh_days.append(t)

        calibrated_var = var_pred[t] + offset
        idx.append(t)
        breach.append(bool(returns[t] < calibrated_var))
        ratio_series.append(ratio)
        refresh_kind.append(kind)
        offset_series.append(offset)

    idx = np.array(idx)
    breach = np.array(breach)
    ratio_series = np.array(ratio_series, dtype=float)
    timeline_dates = [str(np.datetime_as_string(dates[t], unit='D')) for t in idx]

    T = len(breach)
    N = int(breach.sum())
    coverage = N / T
    kupiec_p = kupiec_from_counts(N, T)

    # restrict to the same trailing slice the fixed-offset eval half covered, for a
    # like-for-like comparison against the single-offset numbers
    mask_eval_slice = idx >= eval_start_idx
    T_eval = int(mask_eval_slice.sum())
    N_eval = int(breach[mask_eval_slice].sum())
    coverage_eval = N_eval / T_eval if T_eval else None
    kupiec_p_eval = kupiec_from_counts(N_eval, T_eval) if T_eval else None

    valid, elevated = elevated_mask(ratio_series, elevated_ratio)

    elevated_breach_rate = float(breach[elevated].mean()) if elevated.any() else None
    normal_breach_rate = float(breach[valid & ~elevated].mean()) if (valid & ~elevated).any() else None

    # lead-time check: for each breach day, was an "elevated" reading already showing
    # in the preceding `lead_days`? compare that rate to a random day's rate, to test
    # whether the signal actually leads breaches rather than just coinciding with them.
    def preceded_by_elevated(day_pos):
        lo = max(0, day_pos - lead_days)
        return bool(elevated[lo:day_pos].any())

    breach_positions = np.where(breach)[0]
    non_breach_positions = np.where(~breach)[0]
    breach_lead_rate = float(np.mean([preceded_by_elevated(p) for p in breach_positions])) if len(breach_positions) else None
    baseline_lead_rate = float(np.mean([preceded_by_elevated(p) for p in non_breach_positions])) if len(non_breach_positions) else None

    return {
        "params": {"L": L, "R": R, "S": S, "elevated_ratio": elevated_ratio, "lead_days": lead_days,
                   "trigger_on_elevated": trigger_on_elevated, "min_trigger_gap": min_trigger_gap},
        "refreshes": {"scheduled": n_scheduled_refreshes, "triggered": n_triggered_refreshes},
        "full_walk": {"n": T, "breaches": N, "coverage": coverage, "kupiec_p": kupiec_p},
        "eval_slice_only": {"n": T_eval, "breaches": N_eval, "coverage": coverage_eval, "kupiec_p": kupiec_p_eval},
        "monitor": {
            "elevated_breach_rate": elevated_breach_rate,
            "normal_breach_rate": normal_breach_rate,
            "breach_preceded_by_elevated_rate": breach_lead_rate,
            "baseline_preceded_by_elevated_rate": baseline_lead_rate,
            "n_elevated_days": int(elevated.sum()),
            "n_valid_days": int(valid.sum()),
        },
        "timeline": {
            "date": timeline_dates,
            "vol_ratio": [None if np.isnan(v) else round(float(v), 4) for v in ratio_series],
            "breach": [bool(b) for b in breach],
            "elevated": [bool(e) for e in elevated],
            "offset": [round(float(o), 6) for o in offset_series],
        },
        "refresh_days": refresh_days,
    }
