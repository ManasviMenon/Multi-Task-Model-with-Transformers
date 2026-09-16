# Monitoring, drift, and the retrain trigger

The detailed companion to the README's monitoring summary. Research write-up: [`RESULTS.md`](../RESULTS.md).

## Why this layer exists

A VaR calibration fitted on one window fails the Kupiec test on the next in 4 of 5 regime-aligned folds, and
every static fix tried — refresh cadence, trigger threshold, calibration-window length — was eliminated
(RESULTS.md §3–§8). If no static configuration holds, the operational answer is to watch the calibration and
know when it has broken.

`src/mtl_var/monitoring.py` runs inside the serving process and is exposed through `GET /metrics`, with
realised outcomes arriving through `POST /outcome`.

## Signals

| Signal | What it measures | Fires when | What it means |
|---|---|---|---|
| `performance_drift` | Rolling 250-day VaR breach rate and Kupiec test, scored as realised returns arrive | Kupiec rejects at 5%, with at least 100 scored forecasts | The calibration is wrong on live data |
| `volatility_regime_shift` | Live 10-day realised vol ÷ vol of the window the offset was fitted on | Ratio above 1.5× or below 1/1.5× | Live conditions have left the calibration window's regime |
| `feature_drift` | Per-feature mean and variance of recent inputs against the training distribution | Mean shifts more than 2 training SDs, or variance ratio leaves [1/3, 3] | Inputs have moved somewhere the model never saw |
| latency | p50 / p95 / p99 serving latency | — (reported, not alerted) | — |

Three design choices carry lessons from the research:

- **No verdict on an underpowered window.** Below 100 scored forecasts the monitor reports `insufficient_data`.
  A monitor judging on 20 observations produces extreme p-values that mean nothing, in either direction.
- **The Kupiec boundary rejects.** The research code reports the p-value at zero or all breaches as undefined.
  Read naively, *zero breaches in 250 days* — the most extreme under-breach there is — would look like "no
  rejection". The monitor uses the likelihood ratio's well-defined limit instead.
- **The volatility signal is asymmetric, and says so.** It leads acute shocks and inverts on slow grinds
  (RESULTS.md §5), and `/metrics` carries that caveat next to the reading.

Alerts are structured JSON log lines emitted on state *transitions*, not on every request. Setting
`MTL_VAR_ALERT_WEBHOOK` records where a webhook would be posted; nothing is sent.

## From alert to retrain, in a production system

Nothing below is automated in this repository, by design. It describes how the signals would connect to
retraining if the system were run for real.

**1. Route each signal to the cheapest action that addresses it.**

| Alert | First response | Escalates to a retrain when |
|---|---|---|
| `volatility_regime_shift` | Recalibrate the offset on a recent trailing window — no new model. Rolling recalibration rescued 4 of 5 folds that a frozen offset failed (RESULTS.md §4). | It persists long enough that `performance_drift` also fires |
| `performance_drift` | Open a retrain job | Immediately — the served calibration is rejected on live data |
| `feature_drift` | Investigate which features moved; a level shift in volume is not a broken input feed | It is sustained (for example 20 consecutive trading days) and not explained by a data error |

**2. The retrain job is the existing pipeline, pointed at newer data.** The webhook would hit an orchestrator —
an Airflow DAG, or a GitHub Actions `workflow_dispatch` — which runs the same tracked training used for every
model here, with an expanding window through the latest scored date. The MLflow run records the triggering
alert, its payload and the data hash alongside the usual params, metrics and model artifact.

**3. A candidate is registered, never promoted automatically.** `mtl-var register-model` creates a new version
with full lineage at `@staging`, not `@production`.

**4. Promotion is gated on walk-forward evidence, not on one window.** Single-split calibration verdicts proved
unstable in the research; only the walk-forward folds held up. So the gate is:

- evaluate the candidate walk-forward on the most recent held-out windows, with the same calibration and Kupiec
  code;
- evaluate the production model on the same windows;
- promote only if the candidate passes where production fails, and regresses nowhere production passes;
- a human moves the alias (`mtl-var promote-model <version> --alias production`), and the superseded version is
  retired with its run and artifact kept (`mtl-var retire-model <version> --reason ...`).

**5. Why the loop stops at a human.** No static calibration configuration survived every regime. A pipeline that
retrained and promoted automatically on each alert would optimise against whichever window most recently broke,
which is the fragile single-window verdict the research spent most of its effort discrediting. Sometimes the
right response to `performance_drift` is recalibration plus a tighter caveat, not a new model.

## What the drift replay showed

`scripts/drift_demo.py` replays history through the serving app one trading day at a time — forecast, realised
outcome, `/metrics` — and records every alert transition. Its expectations are pre-registered in the script's
docstring; full output is in `results/drift_demo_results.json`.

**2022 — the served model's out-of-sample year. Performance monitoring stays quiet, as it should.**

| Phase | Forecasts | Breaches | Rate | Kupiec p |
|---|---|---|---|---|
| Offset in-sample (calibration half) | 126 | 6 | 4.76% | 0.902 |
| Offset out-of-sample (evaluation half) | 124 | 4 | 3.23% | 0.334 |
| Whole year | 250 | 10 | 4.00% | 0.453 |

`performance_drift` never fired, and the out-of-sample half reproduced the walk-forward backtest's figure exactly
through the live request path.

**2019–2020 — calm, then the COVID crash, against an offset fitted on volatile 2022 H1.** The network saw these
years in training, so this tests the calibration offset and the monitors, not the network.

| Phase | Forecasts | Breaches | Rate | Kupiec p |
|---|---|---|---|---|
| Calm, pre-shock | 180 | 3 | 1.67% | 0.018 (under-breaching) |
| COVID shock, 20 Feb – 30 Apr 2020 | 50 | 10 | **20.00%** | **0.00019** (over-breaching) |
| Recovery | 170 | 5 | 2.94% | 0.184 |
| Whole window | 400 | 18 | 4.50% | 0.641 |

All pre-registered checks passed. The replay also exposed four weaknesses in the monitor itself:

1. **The rolling breach monitor averaged the crash away.** It caught calm 2019's under-breaching (fired 10 Jan
   2020), then the crash's breaches pulled the rolling rate back toward 5% and the alert resolved mid-sell-off.
   On 18 Mar 2020 the window read 5.00%, p=1.00 — "on target" — although the shock phase alone breached at 20%.
   A 250-day Kupiec test is itself a static window, and it blends regimes.
2. **The volatility monitor caught the crash, but after it started.** Its first high-side reading came on
   4 Mar 2020, after the first two crash breaches (24 Feb, 27 Feb). The alert is also two-sided: every 2022 firing
   was on *low* volatility.
3. **`feature_drift`, as built, carries no information.** It fired on 236 of 250 days in 2022 and 386 of 400 in
   2019–20, all on variance ratio: a 60-day window of a slow feature naturally has a small fraction of its
   18-year variance, so ordinary behaviour reads as drift.
4. **Alerts flap around their thresholds** — 4 of the volatility alert's 9 episodes in 2022 lasted two days or
   less.

**Proposed fixes, not implemented.** Changing the monitor after seeing the replay would be tuning it to the
replay, so each fix is paired with the criterion it must meet first:

| Fix | Criterion |
|---|---|
| A short-window breach test (e.g. a 20-day binomial tail test) alongside the 250-day Kupiec | Fires during the Feb–Mar 2020 shock; fires on no day in 2022 |
| A feature-drift baseline built from training windows of the same length, not 18-year totals | Fires on under 10% of 2022 days; every flag names a window statistic outside the training windows' 1st–99th percentile |
| Hysteresis: fire at the threshold, resolve only after a margin (Kupiec back above p=0.10, vol ratio inside 1.3×) | No alert episode of two days or less in either replay, with the first crash-detection date unchanged |

Serving latency during the replays, in-process: p50 ≈ 152 ms, p95 ≈ 183–194 ms per forecast, most of it rebuilding
features from 150 raw bars.
