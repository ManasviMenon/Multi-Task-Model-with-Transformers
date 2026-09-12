"""Guards that the Phase 1 packaging refactor changed no published number.

The refactor moved code between modules and pulled the volatility-ratio arithmetic out of
`rolling_calibration_analysis` into `monitor.py`. Neither is supposed to alter a single
result, so this replays the calibration and monitor stages against the cached per-fold
predictions and asserts they still reproduce results/walk_forward_results.json exactly.

Runs in seconds - it reads the prediction cache rather than retraining anything. The
training path is verified separately by retraining a fold and diffing the recorded metrics.
"""
import json
import os

import numpy as np
import pytest

from mtl_var.calibration import fixed_offset_calibration, rolling_calibration_analysis

RESULTS_PATH = "results/walk_forward_results.json"
PREDICTIONS_DIR = "results/predictions"

pytestmark = pytest.mark.skipif(
    not os.path.exists(RESULTS_PATH),
    reason="recorded walk-forward results not present (run from the repo root)",
)


def load_recorded():
    with open(RESULTS_PATH) as f:
        return {r["fold"]: r for r in json.load(f) if "error" not in r}


def load_cached_predictions(fold):
    path = os.path.join(PREDICTIONS_DIR, f"{fold}.npz")
    d = np.load(path, allow_pickle=True)
    return (d["dates"].astype("datetime64[D]"), d["returns"], d["var_pred"], int(d["calib_n"]))


RECORDED = load_recorded() if os.path.exists(RESULTS_PATH) else {}
FOLDS = sorted(RECORDED)


@pytest.mark.parametrize("fold", FOLDS)
def test_fixed_offset_calibration_reproduces_recorded(fold):
    _, returns, var_pred, _ = load_cached_predictions(fold)
    recorded = RECORDED[fold]["var_calibration"]

    got = fixed_offset_calibration(returns, var_pred)

    assert got["calib_n"] == recorded["calib_n"]
    assert got["eval_n"] == recorded["eval_n"]
    assert got["offset"] == pytest.approx(recorded["offset"], rel=1e-12)
    for key in ("uncalibrated", "calibrated"):
        assert got[key]["breaches"] == recorded[key]["breaches"]
        assert got[key]["coverage"] == pytest.approx(recorded[key]["coverage"], rel=1e-12)
        assert got[key]["kupiec_p"] == pytest.approx(recorded[key]["kupiec_p"], rel=1e-9, abs=1e-15)


@pytest.mark.parametrize("fold", FOLDS)
@pytest.mark.parametrize("triggered,result_key", [
    (False, "rolling_calibration"),
    (True, "rolling_calibration_triggered"),
])
def test_rolling_calibration_reproduces_recorded(fold, triggered, result_key):
    dates, returns, var_pred, calib_n = load_cached_predictions(fold)
    recorded = RECORDED[fold][result_key]
    if recorded is None:
        pytest.skip(f"{fold} has no recorded {result_key}")

    got = rolling_calibration_analysis(dates, returns, var_pred, eval_start_idx=calib_n,
                                       trigger_on_elevated=triggered)

    assert got["refreshes"] == recorded["refreshes"]
    for block in ("full_walk", "eval_slice_only"):
        assert got[block]["n"] == recorded[block]["n"]
        assert got[block]["breaches"] == recorded[block]["breaches"]
        assert got[block]["coverage"] == pytest.approx(recorded[block]["coverage"], rel=1e-12)
        assert got[block]["kupiec_p"] == pytest.approx(recorded[block]["kupiec_p"], rel=1e-9, abs=1e-15)


@pytest.mark.parametrize("fold", FOLDS)
def test_monitor_signal_reproduces_recorded(fold):
    """The vol-ratio arithmetic moved into monitor.py - these are the numbers in section 5."""
    dates, returns, var_pred, calib_n = load_cached_predictions(fold)
    recorded = RECORDED[fold]["rolling_calibration"]
    if recorded is None:
        pytest.skip(f"{fold} has no recorded rolling_calibration")

    got = rolling_calibration_analysis(dates, returns, var_pred, eval_start_idx=calib_n,
                                       trigger_on_elevated=False)["monitor"]
    exp = recorded["monitor"]

    assert got["n_elevated_days"] == exp["n_elevated_days"]
    assert got["n_valid_days"] == exp["n_valid_days"]
    for key in ("elevated_breach_rate", "normal_breach_rate",
                "breach_preceded_by_elevated_rate", "baseline_preceded_by_elevated_rate"):
        if exp[key] is None:
            assert got[key] is None
        else:
            assert got[key] == pytest.approx(exp[key], rel=1e-12)


@pytest.mark.parametrize("fold", FOLDS)
def test_timeline_series_reproduce_recorded(fold):
    """Per-day offset and vol-ratio series, not just the aggregates."""
    dates, returns, var_pred, calib_n = load_cached_predictions(fold)
    recorded = RECORDED[fold]["rolling_calibration"]
    if recorded is None:
        pytest.skip(f"{fold} has no recorded rolling_calibration")

    got = rolling_calibration_analysis(dates, returns, var_pred, eval_start_idx=calib_n,
                                       trigger_on_elevated=False)["timeline"]
    exp = recorded["timeline"]

    assert got["date"] == exp["date"]
    assert got["breach"] == exp["breach"]
    assert got["elevated"] == exp["elevated"]
    assert got["offset"] == exp["offset"]
    assert got["vol_ratio"] == exp["vol_ratio"]
