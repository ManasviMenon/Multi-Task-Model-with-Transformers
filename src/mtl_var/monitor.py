"""The volatility-ratio monitor: live 10-day realised vol vs the vol of the calibration window.

These are the per-day primitives that were inline inside `rolling_calibration_analysis`,
extracted so the same signal can run standalone against a live stream later without
duplicating the arithmetic. `calibration.rolling_calibration_analysis` calls them, so there
is exactly one implementation and the published monitor numbers (RESULTS.md section 5) are
unchanged.

"Elevated" is symmetric in log space: |log2(live / reference)| > log2(threshold) fires on a
volatility collapse as well as a spike, which is deliberate - the offset is equally stale
either way.
"""
import numpy as np

LIVE_WINDOW = 10       # S: trailing days for the "live" volatility estimate
CALIB_WINDOW = 125     # L: trailing days the active offset was fit on
ELEVATED_RATIO = 1.5   # default firing threshold


def live_volatility(returns, t, window=LIVE_WINDOW):
    """Trailing realised vol over `window` days ending at (not including) t. NaN before warm-up."""
    return float(returns[t - window:t].std()) if t >= window else float("nan")


def vol_ratio(live_vol, reference_vol):
    """live / reference, NaN when either side is unavailable (reference None or 0.0)."""
    if reference_vol and not np.isnan(live_vol):
        return live_vol / reference_vol
    return float("nan")


def is_elevated(ratio, elevated_ratio=ELEVATED_RATIO):
    """Single-day flag: is the ratio far enough from 1.0, in either direction, to fire?"""
    return (not np.isnan(ratio)) and (abs(np.log2(ratio)) > np.log2(elevated_ratio))


def elevated_mask(vol_ratios, elevated_ratio=ELEVATED_RATIO):
    """Vectorised form over a whole series. Returns (valid_mask, elevated_mask)."""
    log_ratio = np.log2(np.asarray(vol_ratios, dtype=float))
    valid = ~np.isnan(log_ratio)
    return valid, valid & (np.abs(log_ratio) > np.log2(elevated_ratio))
