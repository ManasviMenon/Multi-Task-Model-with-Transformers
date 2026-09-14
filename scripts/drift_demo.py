"""Drift demo: replay history through the serving app one trading day at a time.

Each simulated day does what a production client would. POST /predict with the trailing 150
bars, which returns a forecast for the next trading day. POST /outcome once that day has
happened, with the return actually realised. Then GET /metrics and record what every monitor
says. The service is the real FastAPI app loading the registered @production model. By default
it runs in-process through FastAPI's TestClient, which gives each replay fresh monitoring
state; --url drives a running server instead (whose state then carries across replays).

Two replays, chosen so the monitors have to be right in both directions:

  2022 - the served model's own out-of-sample year. The network was trained through 2021. The
         calibration offset was fitted on fold5's calibration half (early 2022), so that stretch
         is in-sample for the offset and the rest of the year is not. A monitor that alarms
         constantly here is noise.

  2019-06 to 2020-12 - a calm year and then the COVID crash, against an offset fitted on the
         volatile first half of 2022. CAVEAT: the network saw 2019-2020 in training, since fold5
         trains through 2021. This replay tests the calibration offset meeting a regime it was
         not fitted on - which RESULTS.md section 13 showed is model-independent - and the
         monitoring mechanics. It is not a test of the network generalising.

PRE-REGISTERED, written before either replay was run:

  1. No performance verdict before 100 scored forecasts. Guaranteed by construction; checked
     anyway so the timeline shows it.
  2. The volatility_regime_shift alert fires during the Feb-Mar 2020 crash. Section 5 found this
     signal leads acute shocks. If it stays quiet through the fastest crash in the data, that
     contradicts section 5 and is reported as a failure of the monitor.
  3. Alerts are deduplicated: far fewer alert events than days on which a signal is firing.
  4. The direction of any performance verdict in the COVID replay is genuinely uncertain. An
     offset fitted on volatile 2022 H1 should make VaR too conservative for calm 2019
     (under-breaching), while the crash should over-breach. Which dominates the rolling 250-day
     window at a given moment is what this measures, not something assumed.
  5. For 2022, no sustained performance alert is expected: the recorded leakage-free fold5
     evaluation half passed Kupiec (3.23%, p=0.334) and the calibration half is in-sample by
     construction. Feature drift may still fire on inputs whose level has moved since the
     training years - data drift worth knowing about, not a calibration failure.

CORRECTION to the first run's reporting, made after it ran: check 2 originally counted any
volatility_regime_shift firing inside the crash window. That alert is two-sided - it also fires
when live volatility is far BELOW the calibration window's - and the "firing from 2020-02-20"
it printed was a low-volatility alert still on from calm February. Check 2 now counts only
high-side readings and reports how many crash breaches came before the first one. The
criterion itself is unchanged. Everything the replay showed that was not pre-registered is
printed separately under OBSERVATIONS and is reported, not scored.
"""
import argparse
import json
import logging
import os
import time

import numpy as np
import pandas as pd

from mtl_var.data import SEQUENCE_LENGTH, load_and_engineer_features
from mtl_var.monitoring import BreachMonitor
from mtl_var.pipeline import FOLDS

CSV = "SPX500-18Y_scaled_y.csv"
RESULTS_PATH = "results/drift_demo_results.json"
N_BARS = 150
SIGNALS = ("performance_drift", "volatility_regime_shift", "feature_drift")
CRASH_WINDOW = ("2020-02-20", "2020-04-01")
SHOCK_PHASE = ("2020-02-20", "2020-05-01")


class HttpClient:
    """Same .get/.post/.json() surface as TestClient, for driving a running server."""

    def __init__(self, url):
        import requests

        self.session = requests.Session()
        self.url = url.rstrip("/")

    def get(self, path):
        return self.session.get(self.url + path, timeout=120)

    def post(self, path, json):
        return self.session.post(self.url + path, json=json, timeout=120)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.session.close()


def load_bars():
    raw = pd.read_csv(CSV, parse_dates=["observation_date"], dayfirst=True)
    raw = raw.dropna(axis=1, how="all").drop(index=0).dropna(subset=["observation_date"]).reset_index(drop=True)
    dates = [str(d.date()) for d in raw["observation_date"]]
    bars = [
        {"date": d, "open": float(o), "high": float(h), "low": float(lo), "close": float(c),
         "adj_close": float(ac), "volume": float(v)}
        for d, o, h, lo, c, ac, v in zip(dates, raw["Open"], raw["High"], raw["Low"], raw["Close"],
                                         raw["Adj Close"], raw["Volume"])
    ]
    return dates, raw["Adj Close"].to_numpy(float), bars


def fold5_evaluation_start():
    """First forecast date in fold5's evaluation half. The served offset is in-sample before it."""
    data = load_and_engineer_features(CSV)
    fold = next(f for f in FOLDS if f["name"] == "fold5_2022_bear")
    dates = data["observation_date"]
    test = dates[(dates >= fold["test_start"]) & (dates <= fold["test_end"])].reset_index(drop=True)
    targets = test.iloc[SEQUENCE_LENGTH - 1:].reset_index(drop=True)
    return str(targets.iloc[len(targets) // 2].date())


def replay_plans(eval_start):
    return {
        "2022_out_of_sample": {
            "window": ("2022-01-03", "2022-12-29"),
            "phases": [
                ("offset in-sample (fold5 calibration half)", "2022-01-01", eval_start),
                ("offset out-of-sample (fold5 evaluation half)", eval_start, "2023-01-01"),
            ],
        },
        "2019_2020_covid_shock": {
            "window": ("2019-06-03", "2020-12-30"),
            "phases": [
                ("calm, pre-shock", "2019-06-01", SHOCK_PHASE[0]),
                ("COVID shock", SHOCK_PHASE[0], SHOCK_PHASE[1]),
                ("recovery", SHOCK_PHASE[1], "2021-01-01"),
            ],
        },
    }


def kupiec_summary(rows):
    """Breach rate and Kupiec over a set of rows, through the monitor's own boundary-aware test."""
    m = BreachMonitor(window=10**9, min_observations=1)
    for r in rows:
        m.record_prediction(r["forecast_date"], r["var_5"])
        m.record_outcome(r["forecast_date"], r["realised_return"])
    snap = m.snapshot()
    p = snap.get("kupiec_p")
    return {
        "days": snap["n_scored"],
        "breaches": snap["n_breaches"],
        "breach_rate": snap.get("breach_rate"),
        "kupiec_p": p,
        "verdict": None if p is None else ("PASS" if p >= 0.05 else "FAIL"),
        "direction": snap.get("direction"),
    }


def episodes(timeline):
    """Runs of consecutive days each signal was firing. Short runs are threshold flapping."""
    out = {}
    for s in SIGNALS:
        runs, run = [], 0
        for t in timeline:
            if s in t["firing"]:
                run += 1
            elif run:
                runs.append(run)
                run = 0
        if run:
            runs.append(run)
        out[s] = {"count": len(runs), "lasting_at_most_2_days": sum(r <= 2 for r in runs),
                  "longest": max(runs, default=0)}
    return out


def alert_event(date, signal, state, perf, vol, feat):
    event = {"date": date, "signal": signal, "state": state}
    if signal == "performance_drift":
        event.update(rolling_n=perf.get("n_scored"), breach_rate=perf.get("breach_rate"),
                     kupiec_p=perf.get("kupiec_p"), direction=perf.get("direction"))
    elif signal == "volatility_regime_shift":
        event.update(vol_ratio=vol.get("vol_ratio"))
    else:
        event.update(features=[d["feature"] for d in feat.get("drifted_features", [])],
                     max_mean_shift_sd=feat.get("max_mean_shift_sd"))
    return event


def run_replay(client, plan, dates, closes, bars):
    start, end = plan["window"]
    indices = [i for i, d in enumerate(dates) if start <= d <= end and i >= N_BARS - 1 and i + 1 < len(dates)]

    health = client.get("/health").json()
    if health.get("status") != "ok":
        raise SystemExit(f"service is not healthy: {health}")

    timeline, events = [], []
    firing_before = set()
    days_firing = {s: 0 for s in SIGNALS}
    started = time.perf_counter()

    for k, i in enumerate(indices):
        r = client.post("/predict", json={"bars": bars[i - N_BARS + 1: i + 1]})
        if r.status_code != 200:
            raise RuntimeError(f"{dates[i]}: /predict returned {r.status_code}: {r.text[:300]}")
        forecast = r.json()
        assert forecast["as_of_date"] == dates[i], (forecast["as_of_date"], dates[i])

        realised = float(np.log(closes[i + 1] / closes[i]))
        scored = client.post("/outcome", json={"as_of_date": dates[i], "realised_return": realised}).json()
        m = client.get("/metrics").json()
        perf, vol, feat = m["performance_drift"], m["volatility_regime"], m["feature_drift"]

        firing = set(m["alerts"]["firing"])
        for s in firing:
            days_firing[s] += 1
        for s in sorted(firing - firing_before):
            events.append(alert_event(dates[i + 1], s, "FIRING", perf, vol, feat))
        for s in sorted(firing_before - firing):
            events.append(alert_event(dates[i + 1], s, "RESOLVED", perf, vol, feat))
        firing_before = firing

        timeline.append({
            "as_of_date": dates[i],
            "forecast_date": dates[i + 1],
            "var_5": forecast["prediction"]["var_5"],
            "realised_return": realised,
            "breached": bool(scored["breached"]),
            "prior_high_used": forecast["prior_high_used"],
            "rolling_status": perf["status"],
            "rolling_n": perf["n_scored"],
            "rolling_breach_rate": perf.get("breach_rate"),
            "rolling_kupiec_p": perf.get("kupiec_p"),
            "rolling_rejects": bool(perf.get("rejects_calibration")),
            "rolling_direction": perf.get("direction"),
            "vol_ratio": vol.get("vol_ratio"),
            "vol_elevated": bool(vol.get("elevated")),
            "feature_drifted": bool(feat.get("drifted")),
            "n_drifted_features": feat.get("n_drifted_features"),
            "firing": sorted(firing),
            "latency_ms": forecast["latency_ms"],
        })
        if (k + 1) % 100 == 0 or k + 1 == len(indices):
            print(f"    {k + 1}/{len(indices)} forecasts  (through {dates[i + 1]})", flush=True)

    phases = []
    for label, p_start, p_end in plan["phases"]:
        rows = [t for t in timeline if p_start <= t["forecast_date"] < p_end]
        if not rows:
            continue
        summary = kupiec_summary(rows)
        summary.update(
            phase=label,
            first=rows[0]["forecast_date"],
            last=rows[-1]["forecast_date"],
            vol_elevated_days=sum(t["vol_elevated"] for t in rows),
            days_by_alert={s: sum(s in t["firing"] for t in rows) for s in SIGNALS},
        )
        phases.append(summary)

    final = client.get("/metrics").json()
    return {
        "window": {"first_forecast": timeline[0]["forecast_date"], "last_forecast": timeline[-1]["forecast_date"]},
        "model": {k: health.get(k) for k in ("model_version", "model_alias", "feature_set", "leakage_status",
                                             "trained_on_fold", "calibration_offset", "git_commit")},
        "whole_window": kupiec_summary(timeline),
        "phases": phases,
        "days_firing": days_firing,
        "episodes": episodes(timeline),
        "alert_events": events,
        "final_metrics": final,
        "elapsed_s": round(time.perf_counter() - started, 1),
        "timeline": timeline,
    }


def fmt_p(p):
    return "   n/a" if p is None else f"{p:.3g}"


def report(name, res):
    w, model = res["window"], res["model"]
    print(f"\n{'=' * 104}")
    print(f"{name}: forecasts for {w['first_forecast']} .. {w['last_forecast']}   "
          f"model v{model['model_version']} @{model['model_alias']} ({model['feature_set']}, {model['leakage_status']})")
    print("=" * 104)
    print(f"  {'phase':46s} {'days':>5s} {'breach':>7s} {'rate':>7s} {'kupiec p':>9s} {'verdict':>8s} "
          f"{'vol-elev':>9s} {'perf-alert':>11s}")
    for ph in res["phases"] + [dict(res["whole_window"], phase="WHOLE WINDOW", vol_elevated_days=None,
                                    days_by_alert={"performance_drift": res["days_firing"]["performance_drift"]})]:
        ve = "" if ph["vol_elevated_days"] is None else ph["vol_elevated_days"]
        print(f"  {ph['phase']:46s} {ph['days']:5d} {ph['breaches']:7d} {ph['breach_rate'] * 100:6.2f}% "
              f"{fmt_p(ph['kupiec_p']):>9s} {str(ph['verdict']):>8s} {ve!s:>9s} "
              f"{ph['days_by_alert']['performance_drift']:>11d}")

    print("\n  alert events (transitions only - the alerter deduplicates):")
    if not res["alert_events"]:
        print("    none")
    for e in res["alert_events"]:
        detail = {k: v for k, v in e.items() if k not in ("date", "signal", "state")}
        if "breach_rate" in detail and detail["breach_rate"] is not None:
            detail["breach_rate"] = f"{detail['breach_rate'] * 100:.2f}%"
        if detail.get("kupiec_p") is not None:
            detail["kupiec_p"] = f"{detail['kupiec_p']:.3g}"
        if detail.get("vol_ratio") is not None:
            detail["vol_ratio"] = round(detail["vol_ratio"], 3)
        if "features" in detail:
            detail["features"] = f"{len(detail['features'])} features"
        print(f"    {e['date']}  {e['signal']:24s} {e['state']:8s} {detail}")

    lat = res["final_metrics"]["latency"]
    print(f"\n  days each signal was firing: {res['days_firing']}   alert events: {len(res['alert_events'])}")
    print("  episodes: " + "; ".join(f"{s} {e['count']} ({e['lasting_at_most_2_days']} lasting <=2 days, "
                                    f"longest {e['longest']})" for s, e in res["episodes"].items()))
    print(f"  serving latency: p50={lat['p50_ms']} ms  p95={lat['p95_ms']} ms  (n={lat['n']})   "
          f"replay took {res['elapsed_s']} s")


def preregistered_checks(results):
    print(f"\n{'=' * 104}\nPRE-REGISTERED CHECKS\n{'=' * 104}")
    checks = {}

    early = [t for res in results.values() for t in res["timeline"] if t["rolling_n"] < 100]
    checks["1_no_verdict_before_100"] = all(not t["rolling_rejects"] for t in early)
    print(f"  1. no performance verdict before 100 scored forecasts: "
          f"{'PASS' if checks['1_no_verdict_before_100'] else 'FAIL'}")

    covid = results.get("2019_2020_covid_shock")
    if covid:
        crash = [t for t in covid["timeline"] if CRASH_WINDOW[0] <= t["forecast_date"] < CRASH_WINDOW[1]]
        firing = [t for t in crash if "volatility_regime_shift" in t["firing"]]
        high = [t for t in firing if (t["vol_ratio"] or 0) > 1]
        low = [t for t in firing if (t["vol_ratio"] or 0) < 1]
        breaches = [t["forecast_date"] for t in crash if t["breached"]]
        checks["2_vol_alert_during_crash"] = bool(high)
        print(f"  2. volatility_regime_shift fires on HIGH volatility during the Feb-Mar 2020 crash: "
              f"{'PASS' if high else 'FAIL'}")
        if high:
            first = high[0]
            before = [d for d in breaches if d < first["forecast_date"]]
            print(f"     first high-side firing {first['forecast_date']} (ratio {first['vol_ratio']:.3f}), "
                  f"on {len(high)} of {len(crash)} crash days")
            print(f"     {len(before)} of the {len(breaches)} crash-window breaches came before it"
                  + (" - it lagged the onset rather than leading it" if before else " - it led every one"))
            checks["2_crash_breaches_before_first_high_alert"] = len(before)
        if low:
            print(f"     {len(low)} crash-window days were a LOW-volatility alert still on from calm February "
                  f"- counted as detection by the first run's reporting, which was wrong")

    total_events = sum(len(res["alert_events"]) for res in results.values())
    total_days = sum(sum(res["days_firing"].values()) for res in results.values())
    checks["3_deduplicated"] = total_days == 0 or total_events < total_days / 5
    print(f"  3. alerts deduplicated: {total_events} events across {total_days} signal-firing days: "
          f"{'PASS' if checks['3_deduplicated'] else 'FAIL'}")

    if covid:
        directions = sorted({e["direction"] for e in covid["alert_events"]
                             if e["signal"] == "performance_drift" and e["state"] == "FIRING"} - {None})
        checks["4_covid_verdict_directions"] = directions
        print(f"  4. COVID replay performance verdict direction(s), measured not assumed: {directions or 'no rejection'}")

    y2022 = results.get("2022_out_of_sample")
    if y2022:
        perf_days = y2022["days_firing"]["performance_drift"]
        checks["5_2022_no_sustained_perf_alert"] = perf_days < 20
        print(f"  5. 2022: no sustained performance alert ({perf_days} firing days): "
              f"{'PASS' if checks['5_2022_no_sustained_perf_alert'] else 'FAIL'}")
    return checks


def observations(results):
    """What the replay showed that was not pre-registered. Reported, not scored."""
    print(f"\n{'=' * 104}\nOBSERVATIONS - not pre-registered, reported rather than scored\n{'=' * 104}")
    obs = {}

    covid = results.get("2019_2020_covid_shock")
    if covid:
        shock = [t for t in covid["timeline"] if SHOCK_PHASE[0] <= t["forecast_date"] < SHOCK_PHASE[1]]
        s = kupiec_summary(shock)
        peak = max(t["rolling_breach_rate"] for t in shock)
        over = any("performance_drift" in t["firing"] and t["rolling_direction"] == "over-breaching" for t in shock)
        obs["rolling_window_masked_the_crash"] = {
            "shock_breaches": s["breaches"], "shock_days": s["days"], "shock_rate": s["breach_rate"],
            "shock_kupiec_p": s["kupiec_p"], "peak_rolling_rate_during_shock": peak,
            "performance_alert_for_over_breaching": over,
        }
        print(f"  a. The rolling breach monitor averaged the crash away. The shock phase breached "
              f"{s['breaches']}/{s['days']} ({s['breach_rate'] * 100:.1f}%, Kupiec p={s['kupiec_p']:.3g}), but the "
              f"rolling 250-day rate peaked at {peak * 100:.2f}% during it and performance_drift "
              f"{'did' if over else 'never'} fired for over-breaching: the calm stretch's under-breaches were "
              f"still in the window.")

    for name, res in results.items():
        n = len(res["timeline"])
        fd_days = res["days_firing"]["feature_drift"]
        fd = res["final_metrics"]["feature_drift"]
        listed = fd.get("drifted_features", [])
        variance_only = sum(1 for d in listed if not any(r.startswith("mean") for r in d["reasons"]))
        obs[f"feature_drift_{name}"] = {
            "firing_days": fd_days, "days": n, "flagged_at_end": fd.get("n_drifted_features"),
            "listed_flagged_on_variance_only": variance_only, "listed": len(listed),
            "max_mean_shift_sd": fd.get("max_mean_shift_sd"),
        }
        print(f"  b. {name}: feature_drift fired on {fd_days}/{n} days. At the end {fd.get('n_drifted_features')} "
              f"features were flagged; {variance_only} of the {len(listed)} listed were flagged on variance ratio "
              f"alone, and the largest mean shift was {fd.get('max_mean_shift_sd')} training SDs.")

    flaps = {name: {s: e["lasting_at_most_2_days"] for s, e in res["episodes"].items()} for name, res in results.items()}
    obs["short_alert_episodes"] = flaps
    print(f"  c. Alert episodes lasting at most 2 days (threshold flapping): {flaps}")
    return obs


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default=None, help="drive a running server instead of the in-process app")
    parser.add_argument("--replay", action="append", default=None, help="run only this replay (repeatable)")
    parser.add_argument("--out", default=RESULTS_PATH)
    args = parser.parse_args()

    # Keep per-request INFO lines quiet; ALERT lines are WARNING and still print.
    logging.basicConfig(level=logging.WARNING, format="%(name)s %(message)s")

    dates, closes, bars = load_bars()
    plans = replay_plans(fold5_evaluation_start())
    names = args.replay or list(plans)

    results = {}
    for name in names:
        print(f"\n--- replay {name} ---", flush=True)
        log_path = f"monitoring/drift_demo_{name}.jsonl"
        os.makedirs("monitoring", exist_ok=True)
        if os.path.exists(log_path):
            os.remove(log_path)
        os.environ["MTL_VAR_PREDICTION_LOG"] = log_path

        if args.url:
            client_ctx = HttpClient(args.url)
        else:
            from fastapi.testclient import TestClient

            from mtl_var.serve import app
            client_ctx = TestClient(app)

        with client_ctx as client:
            results[name] = run_replay(client, plans[name], dates, closes, bars)
        report(name, results[name])

    checks = preregistered_checks(results)
    obs = observations(results)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"preregistered_checks": checks, "observations": obs, "replays": results}, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
