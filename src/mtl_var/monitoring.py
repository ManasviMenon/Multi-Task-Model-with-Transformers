"""Monitoring that runs alongside serving: performance drift, concept drift, latency, alerts.

This is the layer the research actually argues for. The central finding is that a VaR
calibration fitted on one window fails on the next - 4 of 5 regime-aligned folds fail Kupiec, a
verdict unchanged across model variants - and no static configuration fixes it: refresh cadence,
trigger threshold and calibration-window length were each tried and eliminated. If a static
calibration cannot be trusted to hold, the operational answer is to watch it and know when it
has broken. That is what this module does.

Four signals, each mapping onto something established in RESULTS.md:

  1. **Performance drift** - the rolling breach rate against the 5% target, with a rolling
     Kupiec test. This is section 3's backtest, run continuously instead of once.

  2. **Concept drift** - the volatility-ratio monitor (live 10-day vol over the vol of the
     window the active offset was fitted on). Section 5 found this leads acute shocks (2008,
     2020) and inverts on slow grinds (2011, 2015), so it is exposed as a signal with that
     asymmetry documented, not as a general-purpose alarm.

  3. **Data drift** - per-feature mean and variance shift against the training distribution.
     Cheap, and it catches the case the VaR backtest cannot: inputs moving somewhere the
     model never saw, before any outcome has arrived to score.

  4. **Latency** - serving percentiles.

One deliberate refusal, carried over from section 11: an underpowered window does not raise a
performance alert. The 2022 fold "passed" Kupiec largely because its evaluation window was a
third the size of the others, not because it was well calibrated. The same trap runs in
reverse live - a handful of observations can produce an extreme p-value that means nothing -
so the breach monitor reports `insufficient_data` until it has a usable sample.
"""
import json
import logging
import os
from collections import deque
from datetime import datetime, timezone

import numpy as np

from .backtest import kupiec_from_counts
from .monitor import ELEVATED_RATIO, LIVE_WINDOW, is_elevated, vol_ratio

logger = logging.getLogger("mtl_var.monitoring")

VAR_TARGET = 0.05

# One trading year, the standard backtest window. Kupiec needs a real sample before its
# p-value means anything; below MIN_OBSERVATIONS the monitor declines to judge.
BREACH_WINDOW = 250
MIN_OBSERVATIONS = 100

# Feature drift: how far a feature's recent mean may move, in training standard deviations,
# and how far its variance may stretch, before it is called drifted.
FEATURE_MEAN_SHIFT_LIMIT = 2.0
FEATURE_VAR_RATIO_LIMIT = 3.0
FEATURE_WINDOW = 60

LATENCY_WINDOW = 500

PREDICTION_LOG = os.environ.get("MTL_VAR_PREDICTION_LOG", "monitoring/predictions.jsonl")
WEBHOOK_URL = os.environ.get("MTL_VAR_ALERT_WEBHOOK")  # stub: logged, never posted


def _utcnow():
    return datetime.now(timezone.utc).isoformat()


class Alerter:
    """Structured alerts to the log, with a webhook stub.

    Deliberately not real infrastructure: it emits one JSON line per alert and, if a webhook
    URL is configured, records that it *would* post there. Deduplicates by signal so a
    sustained condition alerts on transition rather than on every single request - a monitor
    that fires 200 times during one crisis is a monitor people turn off.
    """

    def __init__(self):
        self.active = {}
        self.history = deque(maxlen=100)

    def evaluate(self, signal, firing, detail):
        was_firing = self.active.get(signal, False)
        self.active[signal] = firing

        if firing and not was_firing:
            self._emit(signal, "FIRING", detail)
        elif was_firing and not firing:
            self._emit(signal, "RESOLVED", detail)

    def _emit(self, signal, state, detail):
        alert = {"timestamp": _utcnow(), "signal": signal, "state": state, **detail}
        self.history.append(alert)
        logger.warning("ALERT %s", json.dumps(alert))
        if WEBHOOK_URL:
            logger.warning("ALERT would POST to webhook %s (stub - not sent)", WEBHOOK_URL)

    def snapshot(self):
        return {
            "firing": sorted(k for k, v in self.active.items() if v),
            "recent": list(self.history)[-10:],
        }


class LatencyTracker:
    def __init__(self, window=LATENCY_WINDOW):
        self.samples = deque(maxlen=window)

    def record(self, latency_ms):
        self.samples.append(float(latency_ms))

    def snapshot(self):
        if not self.samples:
            return {"n": 0, "p50_ms": None, "p95_ms": None, "p99_ms": None, "max_ms": None}
        arr = np.array(self.samples)
        return {
            "n": int(arr.size),
            "p50_ms": round(float(np.percentile(arr, 50)), 2),
            "p95_ms": round(float(np.percentile(arr, 95)), 2),
            "p99_ms": round(float(np.percentile(arr, 99)), 2),
            "max_ms": round(float(arr.max()), 2),
        }


class BreachMonitor:
    """Rolling VaR breach rate and Kupiec test - section 3's backtest, run continuously.

    Predictions are held until their realised return arrives; only scored pairs count.
    """

    def __init__(self, window=BREACH_WINDOW, min_observations=MIN_OBSERVATIONS, target=VAR_TARGET):
        self.window = window
        self.min_observations = min_observations
        self.target = target
        self.pending = {}                      # as_of_date -> predicted VaR
        self.scored = deque(maxlen=window)     # (date, var, realised, breached)

    def record_prediction(self, as_of_date, var_5):
        self.pending[as_of_date] = float(var_5)

    def record_outcome(self, as_of_date, realised_return):
        """Score a prediction once its realised return is known. Returns the scored row."""
        if as_of_date not in self.pending:
            return None
        var = self.pending.pop(as_of_date)
        breached = bool(realised_return < var)
        row = {"date": as_of_date, "var_5": var, "realised_return": float(realised_return),
               "breached": breached}
        self.scored.append(row)
        return row

    def snapshot(self):
        n = len(self.scored)
        breaches = sum(1 for r in self.scored if r["breached"])

        if n < self.min_observations:
            return {
                "status": "insufficient_data",
                "n_scored": n,
                "n_breaches": breaches,
                "n_pending": len(self.pending),
                "min_observations": self.min_observations,
                "breach_rate": (breaches / n) if n else None,
                "kupiec_p": None,
                "rejects_calibration": False,
                "note": (f"Kupiec has little power below {self.min_observations} observations. "
                         f"RESULTS.md section 11 found the 2022 fold's apparent pass was largely a "
                         f"sample-size artifact; this monitor will not repeat that mistake live."),
            }

        rate = breaches / n
        boundary = breaches in (0, n)
        # kupiec_from_counts handles the boundary itself (0*log(0) = 0): zero breaches in 250
        # days against 12.5 expected is a rejection, never a silent NaN.
        p = kupiec_from_counts(breaches, n, self.target)
        rejects = bool(p is not None and not np.isnan(p) and p < 0.05)
        return {
            "status": "ok",
            "n_scored": n,
            "n_breaches": breaches,
            "n_pending": len(self.pending),
            "breach_rate": rate,
            "target_rate": self.target,
            "kupiec_p": None if (p is None or np.isnan(p)) else float(p),
            "rejects_calibration": rejects,
            "boundary_case": boundary,
            "direction": ("over-breaching" if rate > self.target else
                          "under-breaching" if rate < self.target else "on-target"),
        }


class VolRatioMonitor:
    """The section 5 signal: live 10-day realised vol over the calibration window's vol."""

    def __init__(self, calibration_window_vol=None, live_window=LIVE_WINDOW,
                 elevated_ratio=ELEVATED_RATIO):
        self.calibration_window_vol = calibration_window_vol
        self.live_window = live_window
        self.elevated_ratio = elevated_ratio
        self.latest = None

    def update(self, recent_returns):
        """recent_returns: array-like of the most recent realised log returns."""
        arr = np.asarray(recent_returns, dtype=float)
        if arr.size < self.live_window or not self.calibration_window_vol:
            self.latest = None
            return None
        live = float(arr[-self.live_window:].std())
        ratio = vol_ratio(live, self.calibration_window_vol)
        self.latest = {
            "live_vol": live,
            "calibration_window_vol": self.calibration_window_vol,
            "vol_ratio": None if np.isnan(ratio) else float(ratio),
            "elevated": bool(is_elevated(ratio, self.elevated_ratio)),
            "threshold": self.elevated_ratio,
        }
        return self.latest

    def snapshot(self):
        if self.latest is None:
            return {"status": "insufficient_data", "live_window": self.live_window,
                    "elevated": False}
        return {
            "status": "ok",
            **self.latest,
            "caveat": ("Section 5: this signal leads acute shocks (2008, 2020) but inverts on "
                       "slow grinds (2011, 2015), where breaches cluster in NORMAL readings. "
                       "Treat an elevated reading as informative and a quiet one as weak evidence."),
        }


class FeatureDriftMonitor:
    """Per-feature mean and variance shift against the training distribution.

    Catches what the breach monitor cannot: inputs moving somewhere the model never saw,
    visible immediately rather than after enough outcomes have arrived to score.
    """

    def __init__(self, feature_names, train_means, train_stds,
                 window=FEATURE_WINDOW, mean_limit=FEATURE_MEAN_SHIFT_LIMIT,
                 var_limit=FEATURE_VAR_RATIO_LIMIT):
        self.feature_names = list(feature_names)
        self.train_means = np.asarray(train_means, dtype=float)
        self.train_stds = np.asarray(train_stds, dtype=float)
        self.window = window
        self.mean_limit = mean_limit
        self.var_limit = var_limit
        self.recent = deque(maxlen=window)

    def record(self, feature_row):
        self.recent.append(np.asarray(feature_row, dtype=float))

    def snapshot(self):
        n = len(self.recent)
        if n < max(10, self.window // 4):
            return {"status": "insufficient_data", "n_observations": n, "window": self.window,
                    "drifted": False, "drifted_features": []}

        arr = np.vstack(self.recent)
        means = arr.mean(axis=0)
        stds = arr.std(axis=0)

        safe_std = np.where(self.train_stds > 0, self.train_stds, np.nan)
        mean_shift = np.abs(means - self.train_means) / safe_std
        var_ratio = np.where(safe_std > 0, (stds / safe_std) ** 2, np.nan)

        drifted = []
        for i, name in enumerate(self.feature_names):
            ms = mean_shift[i]
            vr = var_ratio[i]
            reasons = []
            if np.isfinite(ms) and ms > self.mean_limit:
                reasons.append(f"mean shifted {ms:.2f} train-sd")
            if np.isfinite(vr) and (vr > self.var_limit or vr < 1.0 / self.var_limit):
                reasons.append(f"variance ratio {vr:.2f}")
            if reasons:
                drifted.append({"feature": name, "mean_shift_sd": round(float(ms), 3),
                                "variance_ratio": round(float(vr), 3), "reasons": reasons})

        drifted.sort(key=lambda d: -d["mean_shift_sd"])
        return {
            "status": "ok",
            "n_observations": n,
            "window": self.window,
            "mean_shift_limit_sd": self.mean_limit,
            "variance_ratio_limit": self.var_limit,
            "drifted": bool(drifted),
            "n_drifted_features": len(drifted),
            "drifted_features": drifted[:10],
            "max_mean_shift_sd": round(float(np.nanmax(mean_shift)), 3),
        }


class MonitoringState:
    """Ties the four signals together and drives alerting."""

    def __init__(self, feature_names=None, train_means=None, train_stds=None,
                 calibration_window_vol=None, prediction_log=PREDICTION_LOG):
        self.latency = LatencyTracker()
        self.breaches = BreachMonitor()
        self.vol_ratio = VolRatioMonitor(calibration_window_vol=calibration_window_vol)
        self.features = (FeatureDriftMonitor(feature_names, train_means, train_stds)
                         if feature_names is not None else None)
        self.alerter = Alerter()
        self.prediction_log = prediction_log
        self.started_at = _utcnow()
        self.n_predictions = 0

    # ------------------------------------------------------------------ ingestion

    def _append_log(self, record):
        """One JSON line per event. Not a database - just an append-only file so the record
        survives a restart, which in-memory deques do not."""
        if not self.prediction_log:
            return
        try:
            os.makedirs(os.path.dirname(self.prediction_log) or ".", exist_ok=True)
            with open(self.prediction_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except OSError as e:
            logger.warning("could not append to prediction log: %s", e)

    def record_prediction(self, as_of_date, prediction, latency_ms, feature_row=None,
                          recent_returns=None):
        self.n_predictions += 1
        self.latency.record(latency_ms)
        self.breaches.record_prediction(as_of_date, prediction["var_5"])

        if feature_row is not None and self.features is not None:
            self.features.record(feature_row)
        if recent_returns is not None:
            self.vol_ratio.update(recent_returns)

        self._append_log({"event": "prediction", "timestamp": _utcnow(), "as_of_date": as_of_date,
                          "latency_ms": round(float(latency_ms), 2), **prediction})
        self._evaluate_alerts()

    def record_outcome(self, as_of_date, realised_return):
        row = self.breaches.record_outcome(as_of_date, realised_return)
        self._append_log({"event": "outcome", "timestamp": _utcnow(), "as_of_date": as_of_date,
                          "realised_return": float(realised_return),
                          "matched_prediction": row is not None,
                          "breached": row["breached"] if row else None})
        self._evaluate_alerts()
        return row

    # ------------------------------------------------------------------ alerting

    def _evaluate_alerts(self):
        b = self.breaches.snapshot()
        self.alerter.evaluate(
            "performance_drift",
            b.get("rejects_calibration", False),
            {"breach_rate": b.get("breach_rate"), "kupiec_p": b.get("kupiec_p"),
             "n_scored": b.get("n_scored"), "direction": b.get("direction"),
             "action": "calibration rejected on live data - see the retrain trigger in the README"},
        )

        v = self.vol_ratio.snapshot()
        self.alerter.evaluate(
            "volatility_regime_shift",
            bool(v.get("elevated")),
            {"vol_ratio": v.get("vol_ratio"), "threshold": v.get("threshold"),
             "action": "live volatility has diverged from the calibration window"},
        )

        if self.features is not None:
            f = self.features.snapshot()
            self.alerter.evaluate(
                "feature_drift",
                bool(f.get("drifted")),
                {"n_drifted_features": f.get("n_drifted_features"),
                 "max_mean_shift_sd": f.get("max_mean_shift_sd"),
                 "features": [d["feature"] for d in f.get("drifted_features", [])],
                 "action": "inputs have moved away from the training distribution"},
            )

    # ------------------------------------------------------------------ reporting

    def snapshot(self):
        return {
            "status": "ok",
            "started_at": self.started_at,
            "n_predictions": self.n_predictions,
            "performance_drift": self.breaches.snapshot(),
            "volatility_regime": self.vol_ratio.snapshot(),
            "feature_drift": (self.features.snapshot() if self.features is not None
                              else {"status": "not_configured", "drifted": False}),
            "latency": self.latency.snapshot(),
            "alerts": self.alerter.snapshot(),
        }
