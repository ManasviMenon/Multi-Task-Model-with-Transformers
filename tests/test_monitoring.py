"""Tests for the live monitoring layer.

Each signal is tested on the property that would make it wrong in production, not just on
"returns a dict". Two of those are lessons from the research carried into code:

  - An underpowered window must not produce a verdict. RESULTS.md section 11 found the 2022
    fold's Kupiec pass was largely a sample-size artifact; a monitor judging on 20 observations
    would make the same mistake live, in either direction.
  - The Kupiec boundary must reject. Zero breaches in 250 days is the most extreme under-breach
    possible, and the research code reports that p-value as NaN. A monitor that reads NaN as
    "no rejection" would stay silent exactly when the VaR is most wrong.
"""
import json
import math

import numpy as np
import pytest

from mtl_var.monitoring import (
    Alerter,
    BreachMonitor,
    FeatureDriftMonitor,
    LatencyTracker,
    MonitoringState,
    VolRatioMonitor,
)


def feed(monitor, n, n_breaches, var=-0.02):
    """Score n predictions, the first n_breaches of which are breached."""
    for i in range(n):
        date = f"d{i:04d}"
        monitor.record_prediction(date, var)
        monitor.record_outcome(date, var - 0.01 if i < n_breaches else var + 0.01)


# ------------------------------------------------------------------------- breach monitor

def test_breach_monitor_refuses_a_verdict_on_an_underpowered_window():
    m = BreachMonitor(window=250, min_observations=100)
    feed(m, n=40, n_breaches=40)  # a 100% breach rate - and still not enough data to judge

    snap = m.snapshot()
    assert snap["status"] == "insufficient_data"
    assert snap["rejects_calibration"] is False
    assert snap["kupiec_p"] is None
    assert snap["breach_rate"] == 1.0


def test_breach_monitor_accepts_a_calibrated_rate():
    m = BreachMonitor(window=250, min_observations=100)
    feed(m, n=250, n_breaches=12)  # 4.8% against a 5% target

    snap = m.snapshot()
    assert snap["status"] == "ok"
    assert snap["rejects_calibration"] is False
    assert snap["kupiec_p"] > 0.05


def test_breach_monitor_rejects_over_breaching():
    m = BreachMonitor(window=250, min_observations=100)
    feed(m, n=250, n_breaches=40)  # 16%

    snap = m.snapshot()
    assert snap["rejects_calibration"] is True
    assert snap["direction"] == "over-breaching"
    assert snap["kupiec_p"] < 0.05


def test_zero_breaches_is_a_rejection_not_a_nan():
    m = BreachMonitor(window=250, min_observations=100)
    feed(m, n=250, n_breaches=0)

    snap = m.snapshot()
    assert snap["boundary_case"] is True
    assert snap["direction"] == "under-breaching"
    assert snap["rejects_calibration"] is True
    # LR = -2 * 250 * ln(0.95) = 25.65 on chi2(1)
    assert snap["kupiec_p"] == pytest.approx(4.14e-7, rel=0.05)


def test_breach_window_rolls():
    m = BreachMonitor(window=100, min_observations=50)
    feed(m, n=100, n_breaches=30)
    assert m.snapshot()["n_breaches"] == 30

    for i in range(100):  # a clean stretch pushes every earlier breach out of the window
        date = f"late{i:04d}"
        m.record_prediction(date, -0.02)
        m.record_outcome(date, 0.0)
    snap = m.snapshot()
    assert snap["n_scored"] == 100
    assert snap["n_breaches"] == 0


def test_outcome_without_a_prediction_is_ignored():
    m = BreachMonitor()
    assert m.record_outcome("2099-01-01", -0.5) is None
    assert m.snapshot()["n_scored"] == 0


# ------------------------------------------------------------------------- volatility ratio

def test_vol_ratio_needs_a_full_live_window():
    v = VolRatioMonitor(calibration_window_vol=0.01, live_window=10)
    assert v.update(np.full(5, 0.01)) is None
    assert v.snapshot()["status"] == "insufficient_data"


def test_vol_ratio_flags_a_volatility_shock():
    rng = np.random.default_rng(0)
    v = VolRatioMonitor(calibration_window_vol=0.01, live_window=10, elevated_ratio=1.5)

    calm = v.update(rng.normal(0, 0.01, 60))
    assert calm["elevated"] is False

    shock = v.update(rng.normal(0, 0.05, 60))  # five times the calibration window's vol
    assert shock["vol_ratio"] > 3
    assert shock["elevated"] is True


# ------------------------------------------------------------------------- feature drift

def test_feature_drift_quiet_on_the_training_distribution():
    rng = np.random.default_rng(1)
    d = FeatureDriftMonitor(["a", "b"], train_means=[0.0, 10.0], train_stds=[1.0, 2.0], window=60)
    for _ in range(60):
        d.record([rng.normal(0, 1), rng.normal(10, 2)])

    snap = d.snapshot()
    assert snap["status"] == "ok"
    assert snap["drifted"] is False


def test_feature_drift_names_the_feature_that_moved():
    rng = np.random.default_rng(2)
    d = FeatureDriftMonitor(["stable", "shifted"], train_means=[0.0, 0.0], train_stds=[1.0, 1.0], window=60)
    for _ in range(60):
        d.record([rng.normal(0, 1), rng.normal(4, 1)])  # four training sd away

    snap = d.snapshot()
    assert snap["drifted"] is True
    assert [f["feature"] for f in snap["drifted_features"]] == ["shifted"]


# ------------------------------------------------------------------------- alerting, latency

def test_alerter_fires_on_transition_not_on_every_request():
    a = Alerter()
    for _ in range(5):
        a.evaluate("performance_drift", True, {"breach_rate": 0.2})
    a.evaluate("performance_drift", False, {"breach_rate": 0.05})

    states = [e["state"] for e in a.history]
    assert states == ["FIRING", "RESOLVED"]  # five firing evaluations, one alert
    assert a.snapshot()["firing"] == []


def test_latency_percentiles():
    t = LatencyTracker()
    for ms in range(1, 101):
        t.record(ms)
    snap = t.snapshot()
    assert snap["n"] == 100
    assert snap["p50_ms"] == pytest.approx(50.5)
    assert snap["max_ms"] == 100


# ------------------------------------------------------------------------- end to end

def test_monitoring_state_logs_scores_and_alerts(tmp_path):
    log = tmp_path / "predictions.jsonl"
    state = MonitoringState(feature_names=["x"], train_means=[0.0], train_stds=[1.0],
                            calibration_window_vol=0.01, prediction_log=str(log))

    prediction = {"log_return": 0.0, "log_volume": 20.0, "yz_vol": 1e-4,
                  "var_5_raw": -0.001, "var_5": -0.02, "cvar_5": -0.03}
    for i in range(120):
        date = f"2020-{i:04d}"
        state.record_prediction(date, prediction, latency_ms=50.0, feature_row=[0.0],
                                recent_returns=np.full(20, 0.01))
        state.record_outcome(date, -0.05)  # every forecast breached

    snap = state.snapshot()
    assert snap["performance_drift"]["rejects_calibration"] is True
    assert "performance_drift" in snap["alerts"]["firing"]
    assert snap["latency"]["n"] == 120

    lines = [json.loads(x) for x in log.read_text().splitlines()]
    assert [x["event"] for x in lines[:2]] == ["prediction", "outcome"]
    assert len(lines) == 240
    assert all(x["breached"] for x in lines if x["event"] == "outcome")
    assert not math.isnan(snap["performance_drift"]["kupiec_p"])
