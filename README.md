# mtl-var

[![CI](https://github.com/ManasviMenon/Multi-Task-Model-with-Transformers/actions/workflows/ci.yml/badge.svg)](https://github.com/ManasviMenon/Multi-Task-Model-with-Transformers/actions/workflows/ci.yml)

Post-hoc calibration of a parametric Value-at-Risk estimator under regime shift, operationalised as a
tracked, registered, served and monitored model. Research write-up: [`RESULTS.md`](RESULTS.md).

> This README is being built phase by phase. The CI and monitoring sections are complete; the portfolio
> front page, architecture diagram and quickstart are still to come.

## Continuous integration

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on every push and pull request, as two parallel
jobs:

- **Lint and test** — installs the package with its pinned dependencies, runs `ruff`, then the full `pytest`
  suite.
- **Build image** — builds the Docker image, starts it, and checks that `/health` responds.

This is integration, not deployment: nothing is pushed to an image registry or a host.

A clean checkout has no `mlruns/` — the local model registry is gitignored — so two things behave
differently in CI than on a machine with a registered model, both by design:

- The serving tests that need a trained model **skip**, with the reason printed. Everything else still runs,
  including request validation, the leakage-free feature invariants, the forecast-row invariance, the
  registry's lineage and retirement logic, and the monitoring suite.
- The image builds **without a model** and starts in a `degraded` state, which the smoke test accepts. A
  local build bakes the registry in and serves `@production` immediately.

## Monitoring, and how a drift alert would trigger retraining

The research is the reason this layer exists. A VaR calibration fitted on one window fails Kupiec on the
next in 4 of 5 regime-aligned folds, robustly across 10 seeds, and every static fix tried — refresh cadence,
trigger threshold, calibration-window length — was eliminated (RESULTS.md §3–§8). If no static
configuration holds, the operational answer is to watch the calibration and know when it has broken.

`src/mtl_var/monitoring.py` runs inside the serving process and exposes `GET /metrics`.

| Signal | What it measures | Fires when | What it means |
|---|---|---|---|
| `performance_drift` | Rolling 250-day VaR breach rate and Kupiec test, scored as realised returns arrive via `POST /outcome` | Kupiec rejects at 5%, with at least 100 scored forecasts | The calibration is wrong on live data |
| `volatility_regime_shift` | Live 10-day realised vol ÷ vol of the window the offset was fitted on (§5) | Ratio above 1.5× or below 1/1.5× | Live conditions have left the calibration window's regime |
| `feature_drift` | Per-feature mean and variance of recent inputs against the training distribution | Mean shifts more than 2 training SDs, or variance ratio leaves [1/3, 3] | Inputs have moved somewhere the model never saw |
| latency | p50 / p95 / p99 serving latency | — (reported, not alerted) | — |

Three design choices carry lessons from the research rather than defaults:

- **No verdict on an underpowered window.** Below 100 scored forecasts the monitor reports
  `insufficient_data`. §11 found the 2022 fold's Kupiec pass was largely a sample-size artifact; a monitor
  judging on 20 observations would repeat that mistake live, in either direction.
- **The Kupiec boundary rejects.** The research code reports the p-value at zero or all breaches as NaN.
  Read naively, that makes *zero breaches in 250 days* — the most extreme under-breach there is — look like
  "no rejection". The monitor uses the likelihood ratio's well-defined limit instead.
- **The volatility signal is asymmetric, and says so.** §5 found it leads acute shocks (2008, 2020) and
  inverts on slow grinds (2011, 2015). `/metrics` carries that caveat next to the reading.

Alerts are structured JSON log lines, emitted on state *transitions* rather than on every request — a
monitor that fires 200 times during one crisis is a monitor people switch off. Setting
`MTL_VAR_ALERT_WEBHOOK` records where a webhook *would* be posted; nothing is sent.

### From alert to retrain, in a production system

Nothing below is automated in this repository, by design (see the last point). This is how the signals
would connect to retraining if it were run for real.

**1. Route each signal to the cheapest action that addresses it.**

| Alert | First response | Escalates to a retrain when |
|---|---|---|
| `volatility_regime_shift` | Recalibrate the offset on a recent trailing window — no new model. §4 found rolling recalibration rescued 4 of 5 folds that a frozen offset failed. | It persists long enough that `performance_drift` also fires |
| `performance_drift` | Open a retrain job | Immediately — the served calibration is rejected on live data |
| `feature_drift` | Investigate which features moved; a level shift in volume is not the same as a broken input feed | It is sustained (for example, 20 consecutive trading days) and not explained by a data error |

**2. The retrain job is the existing pipeline, pointed at newer data.** The webhook would hit an
orchestrator (an Airflow DAG, or a GitHub Actions `workflow_dispatch`), which runs the same tracked
training used for every model here: `mtl_var.pipeline.run_fold` on the leakage-free feature set, with an
expanding training window through the latest scored date. The MLflow run records the triggering alert,
its payload and the data hash alongside the usual params, metrics and model artifact.

**3. A candidate is registered, never promoted automatically.** `mtl-var register-model` creates a new
version with full lineage; it goes to `@staging`, not `@production`.

**4. Promotion is gated on walk-forward evidence, not on one window.** This is the part the research
dictates most directly. The single-split Kupiec verdict flipped under a seed change (§11) and again under a
feature-timing change (§13); only the walk-forward folds held up. So the gate is:

- evaluate the candidate walk-forward on the most recent held-out windows, with the same half/half
  calibration and Kupiec code;
- evaluate the production model on the same windows;
- promote only if the candidate passes where production fails, and regresses nowhere production passes;
- a human moves the alias (`mtl-var promote-model <version> --alias production`), and the superseded
  version is retired with its evidence kept (`mtl-var retire-model <version> --reason ...`).

**5. Why the loop stops at a human.** §7–§8 found that no static calibration configuration survives every
regime. A pipeline that retrains and promotes automatically on each alert would be optimising against
whichever window most recently broke — which is the fragile single-window verdict the research spent most
of its effort discrediting. Sometimes the correct response to `performance_drift` is recalibration plus a
tighter caveat rather than a new model, and that is a judgement call.

### What the drift demo showed

`scripts/drift_demo.py` replays history through the serving app one trading day at a time — forecast,
realised outcome, `/metrics` — and records every alert transition. Its expectations are pre-registered in
the script's docstring; full output is in `results/drift_demo_results.json`. The model is registry
version 2 at `@production` (leakage-free).

**2022 — the served model's out-of-sample year. Performance monitoring stays quiet, as it should.**

| Phase | Forecasts | Breaches | Rate | Kupiec p |
|---|---|---|---|---|
| Offset in-sample (fold5 calibration half) | 126 | 6 | 4.76% | 0.902 |
| Offset out-of-sample (fold5 evaluation half) | 124 | 4 | 3.23% | 0.334 |
| Whole year | 250 | 10 | 4.00% | 0.453 |

`performance_drift` never fired. The out-of-sample half — 4 breaches in 124 days, p=0.334 — is exactly the
figure the §13 backtest recorded, now reproduced end to end through the live request path: features rebuilt
from raw bars, the registry-pinned model, the offset read from its tag, and the monitor's own Kupiec test.

**2019–2020 — calm, then the COVID crash, against an offset fitted on volatile 2022 H1.** The network saw
these years in training, so this tests the calibration offset and the monitors, not the network.

| Phase | Forecasts | Breaches | Rate | Kupiec p |
|---|---|---|---|---|
| Calm, pre-shock | 180 | 3 | 1.67% | 0.018 (under-breaching) |
| COVID shock, 20 Feb – 30 Apr 2020 | 50 | 10 | **20.00%** | **0.00019** (over-breaching) |
| Recovery | 170 | 5 | 2.94% | 0.184 |
| Whole window | 400 | 18 | 4.50% | 0.641 |

All pre-registered checks passed: no verdict before 100 scored forecasts; a high-volatility alert during the
crash; alerts deduplicated (43 events across 983 signal-firing days, a ratio flattered by `feature_drift`
being on almost continuously — see 3 below); no sustained performance alert in 2022. Check 4 asked which
failure direction would dominate the rolling window during the crash. The answer is the first finding.

**What the replay found, including four weaknesses in the monitor itself:**

1. **The rolling breach monitor averaged the crash away.** Calm 2019 under-breached, and the monitor caught
   it: `performance_drift` fired on 10 Jan 2020 (1.95%, p=0.049). Then the crash's breaches pushed the
   rolling rate *up* toward 5%, and the alert resolved on 27 Feb, in the middle of the sell-off. On
   **18 Mar 2020**, with the market in free fall, the rolling window read **5.00%, p=1.00: "on target"**. It
   never fired for over-breaching, although the shock phase alone breached at 20%. The whole-window test
   *passes* (p=0.641) while containing two opposite, individually significant failures. This is the research
   thesis showing up inside the monitor: a static window blends regimes, and a 250-day Kupiec test is itself
   a static window.

2. **The volatility monitor caught the crash, but after it started.** Its first high-side reading came on
   4 Mar 2020 (ratio 1.54). Of the 10 breaches in the shock phase, the first two — 24 Feb (−3.4%) and 27 Feb
   (−4.5%) — came before it; the other eight came while it was on, through 22 Apr. A 10-day live window lags
   a shock by construction. The alert is also two-sided: every one of its 2022 firings was on *low*
   volatility (ratios 0.47–0.66, markets calmer than the window the offset was fitted on). The demo's own
   first report counted a low-volatility alert still on from calm February as crash detection; that
   reporting was corrected in the script.

3. **`feature_drift`, as built, carries no information.** It fired on 236 of 250 days in 2022 and 386 of 400
   in 2019–20. At the end of each replay every flagged feature was flagged on variance ratio alone, and no
   feature's mean had moved more than 1.4 training SDs. The cause is the design, not the markets: a 60-day
   window of a slow feature — drawdown, volume level, lagged volatility — naturally has a small fraction of
   that feature's variance across 18 years of regimes (0.04 for drawdown), so the rule flags ordinary
   behaviour as drift. The one flag pointing the other way was informative — 2022's overnight gaps were five
   times as variable as in training — but a signal that is almost always on buries it.

4. **Alerts flap around their thresholds.** In 2022 the volatility alert had 9 episodes, 4 lasting two days
   or less. In late February 2020 the performance alert resolved at p=0.0503, re-fired the next day at
   p=0.0486, and resolved again two days later.

**Proposed fixes — not made yet.** Changing the monitor after seeing these results would be tuning it to the
replay, so each fix is paired with the criterion it has to meet before it counts:

| Fix | Criterion |
|---|---|
| A short-window breach test (for example a 20-day binomial tail test) alongside the 250-day Kupiec, so an acute over-breach cannot be averaged away | Fires during the Feb–Mar 2020 shock; fires on no day in 2022 |
| A feature-drift baseline built from training windows of the *same length* — compare a live 60-day window with the distribution of 60-day windows during training, not with 18-year totals | Fires on under 10% of 2022 days; every flag names a window statistic outside the training windows' 1st–99th percentile |
| Hysteresis: fire at the threshold, resolve only after a margin (Kupiec back above p=0.10, vol ratio back inside 1.3×) | No alert episode of two days or less in either replay, with the first crash-detection date unchanged |

Serving latency during the replays, in-process: p50 ≈ 152 ms, p95 ≈ 183–194 ms per forecast, most of it
rebuilding features from 150 raw bars.
