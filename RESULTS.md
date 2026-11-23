# Results

> **✅ Loss-shape bug: found, verified, fixed, and re-run. This document reflects the
> corrected numbers.** A prior version of this file (and the buggy `weighted_loss` it was
> generated from) is preserved — see "Files" at the bottom for the full before/after record.
>
> **⚠️ Read §13 before citing any accuracy number.** `log_close_ret` is bit-identical to the
> `log_return` target, and 8 of 18 features embed day-*t* information. The per-task MAE/RMSE
> results measure reconstruction, not forecasting. The VaR calibration conclusions (§3–§9,
> §11) are unaffected — §13 verifies they reproduce with no model in the loop at all.

Multi-task transformer on SPX500 daily data (2004–2022) predicting log-return, log-volume,
Yang-Zhang volatility, VaR₅, and CVaR₅. Summary of everything found and fixed, in order.

## 0. The bug, and how the fix was verified before trusting it

`walk_forward.py`'s manual `GradientTape` training loop (and `MTL_Improved.ipynb`'s
transformer cell) called its loss functions with `y_true` shape `(batch,)` against `y_pred`
shape `(batch,1)`. This doesn't error — it broadcasts to `(batch,batch)`, averaging the loss
over every cross-pair of true/predicted values in a batch instead of each sample's own
residual. Fix: squeeze `y_pred` to `(batch,)` before the loss call.

**Verification checklist, each piece proven rather than assumed:**

1. **Decisive retrain** (`verify_loss_bug_impact.py`): same seed/architecture/data, only the
   squeeze differs. Every task's MAE dropped 16–74%, and the single-split Kupiec result
   flipped from p=0.44 (pass) to p=0.015 (fail). Material, not noise.
2. **Tiny hand-made example** (`tests/test_loss_functions.py`): 3 known data points, expected huber
   / quantile-05 / log-cosh values computed independently in plain NumPy (not by calling
   anything in the codebase), then checked against `huber_loss`/`quantile_loss_05`/`log_cosh`
   and against `weighted_loss` fed an unsquashed `(N,1)` prediction (the real training-loop
   shape). All match to float precision. **All checks passed.**
3. **Shape assertions added permanently** to every loss function in both `walk_forward.py`
   and the notebook (`_assert_same_shape`) — `assert y_true.shape == y_pred.shape` before the
   subtraction. Verified it fires on the exact bug pattern (`(N,)` vs `(N,1)`, unsquashed) and
   stays silent on correctly-shaped input. This bug class cannot be silently reintroduced.
4. **Fix confirmed identical in both places**: the patched `walk_forward.weighted_loss`,
   trained fresh (`sanity_check_patched_loss.py`), reproduced the standalone verification's
   numbers **exactly** — 48.20% uncal / 2.76% cal / p=0.015156, to five decimal places, same
   epoch count (32), same best val_loss (0.107070). The notebook's `cell-29 weighted_loss` and
   `cell-17` loss functions were patched with the logically identical squeeze-before-call fix
   and the notebook was re-run end-to-end to confirm it executes cleanly (the shape assertion
   sits inside functions the concept model also calls via Keras `.compile()`/`.fit()`, which
   reconciles shapes internally — confirmed not to break by actually running it, not assumed).
5. **Hand spot-check** (fold 5, smallest fold): manually counted breaches with a plain Python
   loop over `predictions/fold5_2022_bear.npz` — 3 breaches / 124 days = 0.024194 coverage,
   offset −0.021252. Matched `walk_forward_results.json`'s reported breach count, coverage,
   and Kupiec p exactly. **The evaluation/calibration pipeline (separate from the training
   loss) was already correct** — MAE, RMSE, coverage, and Kupiec were always computed on
   flattened arrays; only the training objective was wrong.

Old (buggy-loss) artifacts are preserved, not deleted: `walk_forward_results_buggy_loss.json`,
`predictions_buggy_loss/`, and the pre-fix backups listed under "Files" below.

## 1. Task-weighting scheme was unstable

The concept-model inverse-validation-loss weighting was tried two ways and broke differently
each time:

| Weighting scheme | Starved task | Its weight |
|---|---|---|
| Raw inverse loss | `log_volume` | 0.0011 |
| Baseline-normalized inverse loss | `yz_vol` | 0.0000477 |

Whichever task's loss (or relative loss) could get closest to zero dominated the `1/loss`
normalization and crowded out every other task's weight. Root cause: five targets on wildly
different native scales (`log_volume` ~O(20), `yz_vol` ~O(0.0005)). *(This finding predates
the loss-bug fix and wasn't re-run under it — the math of why `1/loss` weighting collapses is
a property of the weighting scheme, not of whether the underlying loss call was shaped
correctly, so it should still hold, but it hasn't been independently re-verified.)*

**Fix:** standardize every target (z-score, fit on train only) and use fixed equal weights
(0.2 each). Result, on the corrected loss — every task's MAE improved *simultaneously*
relative to the buggy-loss version of the same model:

| Task | MAE (buggy loss) | MAE (corrected loss) |
|---|---|---|
| log_return | 0.00957 | 0.00423 |
| log_volume | 0.15810 | 0.13237 |
| yz_vol | 0.00006 | 0.00004 |
| VaR₅ | 0.00576 | **0.00150** |
| CVaR₅ | 0.00666 | 0.00384 |

## 2. VaR was miscalibrated, but trainable to fix — and the fix's effect on calibration flips once the loss is correct

VaR₅ is trained with proper pinball/quantile loss (q=0.05) — always was — but pinball loss
only guarantees the *population* quantile, not finite-sample coverage under regime shift.

Post-hoc calibration procedure (unchanged from the original design): split the test set in
half chronologically, fit an offset (5th percentile of return-minus-prediction residuals) on
the first half, apply it to the untouched second half.

| | Coverage | Kupiec p | (buggy-loss result, for reference) |
|---|---|---|---|
| Uncalibrated | 48.20% | ≈0 (fail) | 30.6% / ≈0 (fail) |
| Calibrated | **2.76%** | **0.0152 (fail)** | 4.25% / 0.44 (pass) |

> ⚠️ **Superseded — see §13.** Retrained leakage-free (and with the feature scaler fit
> train-only rather than on the full dataset) this split gives 4.26%, p=0.448, a PASS.
> Combined with §11's seed-fragility, the single-split verdict should not be relied on
> at all; the walk-forward folds are the evidence.

**The single split's one clean pass does not survive the loss fix.** The corrected model is
substantially more accurate (§1) but its VaR quantile sits in a different place relative to
the 5% target — worse, not better, on this split. This is the finding that made re-running
the rest of the campaign necessary rather than optional.

## 3. Walk-forward: does this hold across regimes?

*(This section reports one seed per fold, as originally run. §11 later retrains every fold at
10 seeds each and finds this table is robust, not a lucky draw — 4/5 fail at all 10 seeds
tested, 2022 passes at all 10. Read §11 for the full distribution; the single-point numbers
below turned out to be representative, not fragile.)*

Re-ran the full pipeline (fresh model, fresh scalers, fresh calibration) across 5
expanding-window folds spanning named regimes instead of one 80/20 split:

| Fold | Train | Test | Uncal. coverage | Fixed-offset coverage | Kupiec p |
|---|---|---|---|---|---|
| 2008 GFC | 2004–07 | 2008–09 | 29.5% | 1.59% | 0.0041 (fail) |
| 2011 EU crisis | 2004–10 | 2011–13 | 35.2% | 0.80% | 0.0000042 (fail) |
| 2015 selloff | 2004–14 | 2015–17 | 42.3% | 0.80% | 0.0000040 (fail) |
| 2020 COVID | 2004–18 | 2019–20 | 42.2% | 17.53% | 6.0×10⁻¹³ (fail) |
| 2022 bear | 2004–21 | 2022 | 55.7% | **2.42%** | **0.1444 (PASS)** |

**4 of 5 regime-aligned folds still fail — and 3 of those got measurably worse than the
buggy-loss version, not better.** The one that changed direction, 2022, flips from
borderline-fail (buggy: p=0.045) to a clean pass (p=0.144). Net read: the central thesis
survives correction and arguably strengthens (worse failures dominate), but "0/5 pass" is no
longer accurate — it's "4/5 fail, 1/5 passes."

The volatility-ratio mechanism from the original analysis is **unchanged and re-confirmed**,
because it's a property of realized market returns, not of any model's predictions: in every
fold, the sign of the miscalibration still matches which half of the test window was more
volatile. Calibration half wilder than the eval half that follows → overcorrects →
undercoverage (GFC, EU crisis, 2015, and now 2022, mildly). Calibration half calmer than what
follows → undercorrects → overcoverage (COVID — now even more pronounced, 17.53% vs the
buggy version's 12.75%).

## 4. Does rolling recalibration fix it? Now 4 of 5, up from 3 of 5

> **Re-verified on the leakage-free model — see §13.** Same verdicts: GFC fails, the other four pass
> (EU crisis 4.27%, p=0.504; 2015 selloff 4.26%, p=0.497; COVID 7.17%, p=0.137; 2022 3.28%, p=0.353).
> The numbers below are the original model's.

Same 5 folds, offset refit on a trailing 125-day window, refreshed every 21 days instead of
frozen once.

| Fold | Fixed offset | Rolling | Verdict | (buggy-loss rolling, for reference) |
|---|---|---|---|---|
| 2008 GFC | 1.59% cov, p=0.0041 | **0.80% cov, p=0.000163** | still fails | 0.80% cov, p=0.000163 — **identical to the decimal** |
| 2011 EU crisis | 0.80% cov, p=0.0000042 | 4.00% cov, p=0.358 | fixed | 4.80% cov, p=0.858 |
| 2015 selloff | 0.80% cov, p=0.0000040 | 3.72% cov, p=0.235 | fixed | 3.99% cov, p=0.352 |
| 2020 COVID | 17.53% cov, p=6.0×10⁻¹³ | 7.57% cov, p=0.0815 | fixed, barely | 6.37% cov, p=0.337 |
| 2022 bear | 2.42% cov, p=0.144 | 2.46% cov, p=0.155 | still passes | 1.64% cov, p=0.049 |

**GFC's rolling-recalibration result is bit-for-bit identical between the buggy and corrected
models.** This independently confirms the §8 mechanism (no calm data exists in that window to
anchor on) has nothing to do with model quality — it's structural. Rolling recalibration now
rescues 4 of 5 folds (EU crisis, 2015, COVID, 2022 all pass; only GFC remains fixed-by-nothing).

## 5. Volatility-ratio monitor: a shock detector, not a drift detector — pattern holds under correction

> **Re-verified on the leakage-free model — see §13.** The pattern holds. Breach rates on elevated vs
> normal readings are 7.0% vs 5.4% (GFC) and 6.8% vs 5.4% (COVID), and invert to 3.2% vs 7.2% (EU
> crisis) and 3.3% vs 5.9% (2015 selloff). The numbers below are the original model's.

Signal: ratio of live 10-day realized volatility to the volatility of the 125-day window
backing the currently active offset.

| Fold | Type | Breach rate, elevated | Breach rate, normal | Breach preceded by elevated (5d) | Baseline |
|---|---|---|---|---|---|
| 2008 GFC | acute shock | 7.0% | 4.7% | 87.0% | 70.8% |
| 2020 COVID | acute shock | 8.0% | 4.5% | 96.2% | 80.0% |
| 2011 EU crisis | slow grind | 3.6% | 7.2% | 33.3% | 54.3% |
| 2015 selloff | slow grind | 3.7% | 5.4% | 55.2% | 60.5% |
| 2022 bear | — | 0.0% | 3.0% | 33.3% | 35.3% (still too few breaches — not a reliable read) |

**Same shape as before the fix, if anything slightly cleaner** — the COVID shock signal got
stronger (elevated/normal ratio 1.79× vs the buggy version's 1.25×), and both grind folds
still invert. The claim stands unchanged: this signal leads sudden volatility shocks, not
gradual regime drift.

## 6. Event-triggered recalibration: still doesn't fix 2008; the "identical" pattern is now partial, not universal

Let an elevated volatility-ratio reading force an immediate recalibration instead of waiting
for the next scheduled (21-day) refresh.

| Fold | Scheduled-only | Event-triggered | Refreshes (sched / trig) |
|---|---|---|---|
| 2008 GFC | 0.80% cov, p=0.000163 | **0.80% cov, p=0.000163 — identical** | 3 / 51 |
| 2011 EU crisis | 4.00% cov, p=0.358 | 4.27% cov, p=0.504 | 10 / 54 |
| 2015 selloff | 3.72% cov, p=0.235 | 3.72% cov, p=0.235 — identical | 9 / 68 |
| 2020 COVID | 7.57% cov, p=0.0815 | 7.17% cov, p=0.137 | 3 / 54 |
| 2022 bear | 2.46% cov, p=0.155 | 3.28% cov, p=0.353 | 3 / 7 |

**GFC still completely unmoved.** Under the buggy loss, triggering was a no-op in 4 of 5
folds; under the corrected loss it's a no-op in only 2 of 5 (GFC, 2015) — EU crisis, COVID,
and 2022 now shift slightly under triggering, though all were already passing either way, so
no verdict changes. The core finding is unchanged: cadence was never the lever for GFC.

### 6a. Threshold sweep: cleaner result under the corrected loss

Pre-registered: **rescue = a failing fold now passes; regression = a passing fold now fails.**
Swept `elevated_ratio` ∈ {1.5×, 2.0×, 2.5×}, purely post-hoc from the corrected predictions
cache, no retraining.

| Fold | Threshold | Fires on | Coverage | Kupiec p |
|---|---|---|---|---|
| 2008 GFC | 1.5× / 2.0× / 2.5× | 57.3% / 20.0% / 5.6% | 0.80% (all three) | 0.000163 (all three) — **completely unmoved** |
| 2022 bear | 1.5× | 18.2% | 3.28% | **0.353 — rescued** |
| | 2.0× | 1.7% | 2.46% | **0.155 — rescued** |
| | 2.5× | 0.0% | 2.46% | **0.155 — rescued** |
| EU crisis, 2015, COVID | all three | tightens with threshold | unchanged | still passing, no regressions |

**Improvement over the buggy-loss version of this test: 2022 is now rescued at every threshold
tested, not just 1.5×** (previously 2.0×/2.5× undid the rescue). GFC remains completely
unmoved at every threshold, including 5.6% — rarer than the rate that rescues 2022. Same
conclusion, more robust evidence for it: threshold tightening helps a borderline fold, it
categorically does not touch the fastest-moving crisis.

## 7. Where this stands: two eliminated, window length reworked, GFC's resistance now triangulated four ways

Two independent levers on 2008, both eliminated by evidence, and confirmed twice now (buggy
and corrected loss both give the same verdict):

- **Cadence** (§6): 17× more recalibrations, zero change.
- **Threshold** (§6a): even at a firing rate rarer than what rescues 2022, zero change.

**GFC's fixed-offset (§3), rolling (§4), and event-triggered (§6) results are bit-for-bit
identical between the buggy and corrected models.** That's a third and fourth independent
confirmation, on top of cadence and threshold, that nothing about training quality, refresh
timing, or alarm sensitivity touches this fold. Whatever is wrong there is structural to the
regime itself (no calm data in the window, established in the original §4 mechanism), not an
artifact of any of the four things that have now been ruled out.

## 8. Window length: still the strongest lever, but the regression got worse under the corrected model

> **Partly withdrawn — see §13.** On the leakage-free model L=40 still rescues GFC (4.78%, p=0.873), still
> regresses COVID (9.56%, p=0.003), and no window length passes every fold. But 2015 selloff passes at
> every L (6.91%, p=0.106 at L=40; 3.19%, p=0.086 at L=175). Its failures at both ends below were
> marginal (p=0.042, p=0.046) and an artifact of the leaky features. The claim that L=40 regresses two
> folds is withdrawn: it regresses one.

Pre-registered: **rescue = fold1_gfc flips to p ≥ 0.05; regression = any of {2011 EU crisis,
2015 selloff, 2020 COVID} — passing at L=125 — drops below p = 0.05.** Swept `L` ∈
{40, 60, 90, 125, 175} days, `R=21` held fixed, post-hoc from the corrected cache.

| Fold | L=40 | L=60 | L=90 | L=125 (baseline) | L=175 |
|---|---|---|---|---|---|
| 2008 GFC | **4.78% cov, p=0.873** | 2.39%, p=0.035 | 1.99%, p=0.013 | 0.80%, p=0.00016 | 0.00%, p=3.9×10⁻⁷ (FAIL) |
| 2011 EU crisis | 5.33%, p=0.77 | 4.80%, p=0.86 | 4.53%, p=0.67 | 4.00%, p=0.36 | 4.27%, p=0.50 |
| 2015 selloff | 7.45%, p=**0.042 (fail)** | 6.65%, p=0.16 | 4.52%, p=0.67 | 3.72%, p=0.24 | 2.93%, p=**0.046 (fail)** |
| 2020 COVID | 9.56%, p=0.003 (fail) | 9.96%, p=0.001 (fail) | 8.37%, p=0.025 (fail) | 7.57%, p=0.081 | 7.17%, p=0.137 |
| 2022 bear | 5.65%, p=0.75 | 4.03%, p=0.61 | 3.23%, p=0.33 | 2.46%, p=0.15 | 1.39%, p=0.098 |

**GFC's L=40 rescue is bit-for-bit identical to the buggy-loss result** (4.78%/p=0.873 both
times) — a fifth independent confirmation that this fold's behavior at a given configuration
doesn't depend on model training quality at all.

**But the regression at L=40 is now worse, not the same.** Under the buggy loss, L=40 only
regressed COVID. Under the corrected loss, L=40 regresses **both** 2015 selloff (newly fails,
p=0.042) **and** COVID (still fails, p=0.003) — and 2015 selloff now *also* fails at the long
end, L=175 (p=0.046), a "sweet spot in the middle" (L=60–125) that wasn't visible before.
2022 bear improved — it now passes at every `L` tested (previously failed at L=125/175).

**Pre-registered check, run honestly: L=40 rescues fold1_gfc but regresses two folds, not
one — fails the zero-regression bar more decisively than before.** No single `L` in the range
tested passes GFC and 2015-selloff-and-COVID simultaneously.

**This strengthens, not weakens, the "no static configuration works everywhere" conclusion.**
It's no longer one fold pulling against GFC — it's two, for related but not identical reasons
(both need calibration-window stability that a 40-day lookback discards, just at different
points in their respective test windows).

## 9. Empirical vs Parametric VaR under Walk-Forward — reviewer objection answered, more decisively now

**Question**: is the parametric VaR's walk-forward failure a property of static calibration
under regime shift, or an artifact of the parametric target itself? Tested a pure historical
VaR — 5th percentile of trailing realized returns, no model — through the identical protocol.

**Note on what changed vs. what didn't**: the empirical VaR calculation uses only historical
*returns*, never the trained model's predictions — so its numbers below are **unchanged** by
the loss fix. Only the "Parametric" column moved.

### Primary comparison (N=250, the standard 1-year historical-VaR lookback)

| Fold | Parametric, calibrated | Empirical, raw | Empirical, calibrated |
|---|---|---|---|
| 2008 GFC | 1.59% cov, p=0.0041 | 0.80% cov, p=0.0002 | 0.00% cov, p=3.9×10⁻⁷ |
| 2011 EU crisis | 0.80% cov, p=0.0000042 | 3.20% cov, p=0.088 | 1.60% cov, p=0.0005 |
| 2015 selloff | 0.80% cov, p=0.0000040 | 2.39% cov, p=0.010 | 2.13% cov, p=0.0041 |
| 2020 COVID | 17.53% cov, p=6.0×10⁻¹³ | 8.37% cov, p=0.025 | 11.55% cov, p=0.00004 |
| 2022 bear | **2.42% cov, p=0.144 (PASS)** | 2.42% cov, p=0.144 | 0.81% cov, p=0.0083 |

**Parametric now fails only 4/5 (was 5/5); empirical still fails 5/5 at N=250.** The reviewer
objection is answered even more clearly than before: the properly-trained parametric model is
now *at least as robust as*, not merely equal to, a naive historical VaR — empirical VaR fails
strictly more often. 2022 is the clearest case: the corrected parametric model passes there;
empirical VaR (which can't be told anything by a model) still fails, at 0.81% coverage.

The raw-vs-calibrated wrinkle from the original analysis holds: calibrating empirical VaR
still makes it *worse*, not better (2/5 pass raw vs. 0/5 pass calibrated at N=250) — the
calibration procedure itself is what's fragile to regime shift, independent of the target.

### Sensitivity: is this stable across lookback length?

| Fold | N=125 | N=250 | N=500 |
|---|---|---|---|
| 2008 GFC | fail (0.0%, p=3.9×10⁻⁷) | fail (0.0%, p=3.9×10⁻⁷) | fail (0.4%, p=0.00002) |
| 2011 EU crisis | fail (1.6%, p=0.0005) | fail (1.6%, p=0.0005) | fail (0.8%, p=0.000004) |
| 2015 selloff | **PASS (5.1%, p=0.962)** | fail (2.1%, p=0.0041) | fail (0.8%, p=0.000004) |
| 2020 COVID | **PASS (7.6%, p=0.082)** | fail (11.6%, p=0.00004) | fail (12.0%, p=0.00002) |
| 2022 bear | fail (0.8%, p=0.0083) | fail (0.8%, p=0.0083) | fail (1.6%, p=0.045) |

Identical to the pre-fix version (as expected — this table never depended on the parametric
model). N=125 fails 3/5; N=250 and N=500 fail 5/5.

**Conclusion, updated.** The reviewer objection is answered, and more decisively than the
original analysis suggested: at the standard 1-year lookback, empirical VaR is now *worse*
than the (corrected, properly-trained) parametric model, not merely equal to it. The paper's
claim should read "static calibration fails under regime shift, and a properly-trained
parametric target is at least as robust as the naive empirical alternative" — the earlier
"equally bad" framing undersold the corrected parametric model.

## 10. Ablation: does the multi-task structure help, hurt, or leave VaR calibration unchanged?

Built a single-task VaR-only model, identical to the 5-head MTL model in every respect except
head count — same transformer body (Dense(64) → 4 transformer blocks, 4 heads, key_dim=128 →
GlobalAveragePooling1D), same 18 features, same 5-day sequences, same parametric VaR target,
same pinball loss, same target standardization, same optimizer/schedule/early-stopping, same
calibration procedure. Both trained on the corrected loss (`single_task_var_ablation.py`).

**Pre-registered**: single-task better → the multi-task burden costs VaR calibration.
Same → MTL is neutral. Single-task worse → the shared representation benefits VaR. Only a
direction consistent across the single split AND a majority of folds counts as a finding.

**Table 1 — single split:**

| | uncal cov | cal cov | cal Kupiec p |
|---|---|---|---|
| MTL (5-head) | 48.20% | 2.76% | 0.0152 (fail) |
| Single-task | 40.76% | **6.79%** | **0.0895 (pass)** |

**Table 2 — walk-forward, 5 folds (calibrated):**

| Fold | MTL | Single-task | Direction |
|---|---|---|---|
| 2008 GFC | 1.59%, p=0.0041 | 1.20%, p=0.00097 | single WORSE |
| 2011 EU crisis | 0.80%, p=4.2×10⁻⁶ | 1.07%, p=2.5×10⁻⁵ | single BETTER |
| 2015 selloff | 0.80%, p=4.0×10⁻⁶ | 1.06%, p=2.4×10⁻⁵ | single BETTER |
| 2020 COVID | 17.53%, p=6.0×10⁻¹³ | 17.93%, p=1.4×10⁻¹³ | single WORSE |
| 2022 bear | 2.42%, p=0.144 | **4.84%, p=0.934** | single BETTER |

> ⚠️ **This section's conclusion is WITHDRAWN — see §13.** Retrained on leakage-free
> features the direction reverses to 1 better / 2 identical / 2 worse, so the finding was
> an artifact of the contaminated feature set. Head count makes no consistent difference
> to VaR calibration.

**Pre-registered read: single-task is directionally better in the single split AND the
majority of folds (3/5) — consistent, so this is a real finding: the multi-task burden costs
VaR calibration.** Precision matters here, though — the practically meaningful improvements
(large enough to flip a verdict) are concentrated exactly where MTL was already closest to
passing: the single split flips fail→pass, and 2022 bear goes from a weak pass to a near-
perfect one. In EU crisis and 2015 selloff, single-task is directionally better but both
models are still catastrophically miscalibrated (p~10⁻⁵ either way) — no verdict changes. In
GFC and COVID, single-task is measurably *worse*. "Single-task helps" is accurate; "single-task
fixes it" is not.

## 11. Single-split Kupiec verdicts are seed-fragile — the fold-level ones are not

While confirming the notebook's own re-run reproduced the corrected-loss numbers, it didn't:
the notebook (concept model trained first, no re-seed before the transformer) gave a **PASS**
(p=0.740) on the single split, where the "clean" pipeline used throughout this document
(seed=42, no concept model) gives a **FAIL** (p=0.015) — identical data, identical corrected
loss. Investigated properly rather than asserted, per the plan agreed before running it:

**Step 1 — is this general seed-fragility, or specific to the concept-model mechanism?**
(`verify_seed_sensitivity.py` — an initial version had its own bug, silently re-seeding to a
hardcoded 42 inside a reused helper and nullifying every alternate seed tested; caught via a
suspiciously-exact-duplicate result, fixed, re-run.) Result: **3 of 4 alternate configurations
pass** (seed=43, seed=44, and seed=42-with-a-dummy-model-trained-first all pass; only the
seed=42 reference fails). Not narrowly about the concept model — general fragility.

**Step 2 — does this generalize to the walk-forward folds**, where every headline number in
§3–§9 comes from exactly one trained model per fold? Rather than assume it does, ran the
correct scientific fix: retrain **every fold at 10 seeds each** (seeds 42–51, 50 trainings
total, `seed_distribution_walk_forward.py`), converting each fold's single point estimate into
a distribution. Scoped to the fixed-offset baseline only — the rolling/triggered/threshold/
window-length sweeps are each their own multi-seed undertaking and aren't included here.

| Fold | Pass rate | Coverage (mean ± std, min–max) |
|---|---|---|
| 2008 GFC | **0/10** | 1.67% ± 0.24% (1.20–1.99%) |
| 2011 EU crisis | **0/10** | 0.83% ± 0.08% (0.80–1.07%) |
| 2015 selloff | **0/10** | 1.06% ± 0.17% (0.80–1.33%) |
| 2020 COVID | **0/10** | 17.65% ± 0.62% (16.73–18.73%) |
| 2022 bear | **10/10** | 3.15% ± 0.56% (2.42–4.03%) |

**The fold-level failures are extremely robust — the opposite of what the single-split
fragility might have predicted.** Four folds fail at every one of 10 seeds tested, with tiny
standard deviations (EU crisis: std=0.08%, essentially deterministic regardless of random
init). 2022 bear passes at all 10 seeds too. The seed-fragility that flipped the single
split's verdict does **not** generalize to the walk-forward methodology — it was a property of
that one setup (short calibration window relative to a large single-model capacity, on a much
larger single split), not of the underlying pipeline. This is the reassuring direction: the
central walk-forward finding just got *stronger* evidence behind it, not weaker.

**2022's pass needs the same precision as §10's, not a cleaner story than it earns.** Its
coverage (3.15% mean) is not meaningfully closer to the 5% target in absolute terms than some
failing folds — EU crisis, for comparison, runs ~1–3 breaches out of ~375 eval days (0.8–1.1%
coverage) versus 2022's ~3–5 breaches out of ~124 days (2.4–4.8%). Similar breach *counts*,
very different eval sample sizes. With roughly a third of the data, the Kupiec test has much
less power to flag the same magnitude of miscalibration as significant. 2022's robust pass is
better described as "not detected as significant at this sample size" than "well-calibrated."

**Practical implication for §3's original framing**: "4/5 fail" (the single-seed, seed=42
point estimate) undersold how solid that finding actually is. It should read "4/5 fail
robustly across 10 seeds each; the 5th passes robustly too, but likely due to reduced
statistical power from a shorter test window rather than genuinely better calibration."

## 12. Classical baselines for the single split (Persistence, ARIMA(1,0,1), GARCH(1,1), EWMA VaR)

The paper's "final model vs. classical baselines" table had shipped with four baseline rows
that didn't exist anywhere in this codebase — confirmed via grep across the notebook, every
script, and every results JSON for "arima"/"garch"/"persistence": zero matches. `classical_
baselines.py` actually builds and evaluates them, on the identical single 80/20 split the
notebook's numbers come from (same 941-row test window, 2019-04-09 to 2022-12-30 — confirmed
against the notebook's own printed `Train samples: 3760, Test samples: 941`). Reuses
`load_and_engineer_features` and `kupiec_from_counts` from `walk_forward.py` unmodified.

All four methods are deterministic (no training-seed dependence), and the run reproduced
identically on a second execution — unlike the neural model, there's no seed-fragility
question here.

| Model | Ret. MAE | Vol. MAE | VaR Cov. |
|---|---|---|---|
| Persistence (naive) | 0.0143 | — | — |
| ARIMA(1,0,1) | 0.0096 | — | 3.18% cov, p=0.053 (calibrated; barely passes) |
| GARCH(1,1) | — | 0.0000587 | — |
| EWMA VaR | — | — | 5.94% cov, p=0.360 (calibrated; comfortable pass) |
| MTL Transformer (corrected) | **0.0042** | **0.00004** | 2.76% cov, p=0.0152 (**fail**) |

Notes:
- **Vol. MAE units**: `yz_vol` is a variance-scale quantity (test-period mean ≈1.5×10⁻⁴), not a
  standard deviation (≈1.1×10⁻²). GARCH natively forecasts conditional variance, so it's
  compared variance-to-variance — an earlier draft of this script accidentally compared
  GARCH's *std* against `yz_vol`'s *variance* and got a meaningless ~200× inflated MAE; caught
  by checking the actual scale of both quantities before trusting the number.
- **VaR Cov. is the calibrated variant** for ARIMA and EWMA, matching how the MTL number is
  reported (footnoted as calibrated) — both raw and calibrated are in
  `results/classical_baselines_results.json` for full transparency. Raw: ARIMA 7.55% cov,
  p=0.0008 (fail); EWMA 6.16% cov, p=0.113 (pass).
- ARIMA's coefficients (const≈0.00024, AR(1)≈0.248, MA(1)≈−0.347) are both fit once on train
  and frozen; one-step-ahead forecasts across test use `statsmodels`' `.append(refit=False)`
  so parameters never see test data, matching how the MTL model is trained once and evaluated
  frozen.

> ⚠️ **The MAE comparison in this table is not like-for-like — see §13.** `log_close_ret` is
> bit-identical to the `log_return` target, so the transformer reads day *t*'s close off its
> own input while ARIMA genuinely forecasts day *t* from *t−1*. The VaR-coverage column is
> unaffected (§13 shows the calibration result is model-independent); the accuracy columns
> should not be read as forecasting skill.

**Pre-registered read, honored**: MTL clearly wins both MAE metrics — nearly 2.3× lower return
error than ARIMA (and 3.4× lower than naive persistence), and a smaller volatility-proxy error
than GARCH. **But it does not win on VaR coverage.** On this identical single split, both
classical VaR methods pass Kupiec where the corrected MTL model fails: EWMA VaR passes
comfortably (p=0.36), and even a simple ARIMA-residual VaR passes, if only barely (p=0.053, a
hair over the 0.05 line). This is the honest picture the pre-registration asked for — better
point-forecast accuracy did not buy better single-split VaR calibration, which is fully
consistent with this document's central thesis (§2, §11): the multi-task model's single-split
VaR result is fragile and this table adds one more data point showing it isn't automatically
better than a naive parametric alternative on that axis. The walk-forward result (§9) remains
the stronger, regime-robust version of this comparison, where empirical VaR fails 5/5 and the
corrected parametric model fails only 4/5 — that comparison, not this single-split table,
carries the paper's actual generalization claim.

## 13. Target leakage: the accuracy numbers measure reconstruction, not forecasting

Found while wiring up live monitoring, by checking what "the realized outcome" even means for
a prediction. It is the most serious finding in this document and it invalidates part of it.

**`log_close_ret` is bit-identical to the `log_return` target.** `Close` equals `Adj Close` on
all 4705 rows, so `log(Close_t / AdjClose_{t-1})` *is* `log_return_t`: max absolute difference
**0.0**, correlation exactly 1.0. Feature 4 of 18 is target 1 of 5.

It is not confined to that pair. Sequences and targets are aligned on the same row —
`make_sequences` produces `X[j]` covering rows `j..j+4` while `y[j]` is row `j+4` — so the
input window contains day *t*'s own high, low and close, and the target is day *t*'s value.
**8 of 18 features embed day-t information**: `log_open_ret`, `log_high_ret`, `log_low_ret`,
`log_close_ret`, `drawdown`, `ret_min20`, `ret_max20`, `ret_skew20`. The task as built is
reconstruction of contemporaneous quantities, not forecasting.

### How much of each target is simply computable from the inputs

In-sample linear R² of each target on the features (`scripts/leakage_audit.py`). A value at
1.0 means no learning is required — the answer is a deterministic function of what is fed in.

| Target | R², all 18 features | R², lagged-only features |
|---|---|---|
| log_return | **1.0000000000** | **0.0190** |
| log_volume | 0.8154 | 0.7931 |
| yz_vol | 0.9960 | 0.9937 |
| VaR₅ | **0.9999782** | 0.5937 |
| CVaR₅ | **1.0000000000** | 0.9623 |

The right-hand column is what matters for damage assessment, and it splits the five heads
into three groups:

- **Return: entirely leakage.** R² collapses 1.0000 → 0.0190. Essentially all apparent skill
  was the target being handed to the model as an input. Genuine daily-return predictability of
  ~2% is what market efficiency would lead you to expect.
- **Volatility and CVaR: largely legitimate.** `yz_vol` barely moves (0.9960 → 0.9937) and
  CVaR₅ holds up (1.0000 → 0.9623). Volatility is persistent, so `yz_vol_lag1` already carries
  almost all of it. These heads were never leakage-dependent.
- **VaR: in between.** 0.99998 → 0.5937. Roughly half the target is genuinely available from
  lagged information.

### What this invalidates

**The accuracy table (§1) and the Table 1 baseline comparison.** "MTL 0.0042 vs ARIMA 0.0096"
is not like-for-like: ARIMA forecasts day *t* from *t−1*, while the transformer reads day *t*'s
close off its own input. The model is in fact *worse than the trivial copy baseline* — copying
`log_close_ret` gives MAE exactly 0.0 against the model's 0.0042 — which is the clearest
possible statement that the return head is not doing forecasting work. Any framing of this
project as "a transformer that forecasts next-day returns" is unsupported. The Methodology's
"(i) next-day log-return" is also inconsistent with the implementation, which is same-day.

### What survives — tested, not assumed

The central calibration thesis does **not** depend on the model. Re-running the walk-forward
calibration backtest on the **raw VaR₅ target with no model at all**, through the identical
half/half offset protocol and Kupiec code:

| Fold | Raw target: coverage / p | Trained model: coverage / p | Agree? |
|---|---|---|---|
| 2008 GFC | 1.20% / 0.000966 (FAIL) | 1.59% / 0.00406 (FAIL) | yes |
| 2011 EU crisis | 0.80% / 4.15758e-06 (FAIL) | 0.80% / 4.15758e-06 (FAIL) | yes |
| 2015 selloff | 0.80% / 3.97404e-06 (FAIL) | 0.80% / 3.97404e-06 (FAIL) | yes |
| 2020 COVID | 18.73% / 7.5e-15 (FAIL) | 17.53% / 6.0e-13 (FAIL) | yes |
| 2022 bear | 3.23% / 0.334 (PASS) | 2.42% / 0.144 (PASS) | yes |

**5 of 5 folds agree**, and for EU crisis and 2015 selloff the raw target reproduces the
model's numbers to six significant figures — identical breach counts. The finding is a
property of **the parametric target definition and the offset procedure under regime shift**,
not of a neural forecaster.

This also retro-explains three things that were previously odd:
- GFC's results were **bit-for-bit identical between the buggy and corrected models** across
  four separate configurations (§4, §6, §7, §8). If the VaR head is reconstructing a
  deterministic target, model quality genuinely cannot matter.
- The 10-seed distribution had standard deviations as low as 0.08% (§11) — near-deterministic,
  because the network is approximating a closed-form quantity.
- Empirical VaR behaved so similarly under the same protocol (§9).

### What the leakage was worth: a leakage-free retrain

`scripts/leakage_free_retrain.py` retrains fold5 with each of the 8 contaminated features
replaced by its one-day lag. Same architecture, losses, weights, optimiser, early stopping,
folds, calibration protocol and seed — **only information timing changes**.

| Task | Leaky MAE | Leakage-free MAE | Ratio |
|---|---|---|---|
| log_return | 0.003859 | 0.012851 | **3.33× worse** |
| VaR₅ | 0.001315 | 0.004932 | **3.75× worse** |
| log_volume | 0.106622 | 0.107103 | 1.00× |
| CVaR₅ | 0.003304 | 0.003089 | 0.93× (**better**) |
| yz_vol | 0.000036 | 0.000032 | 0.89× (**better**) |

Every pre-registered expectation held: return degrades badly, VaR moderately, `yz_vol` barely.
Two results are worth stating plainly beyond that.

**The return head does not beat a constant.** On fold5's exact 247-day window:

| Predictor | MAE |
|---|---|
| Leaky model (reads day *t*'s close from its own input) | 0.003859 |
| **Predict zero** | **0.012040** |
| Leakage-free model | 0.012851 |
| Persistence (yesterday's return) | 0.017191 |

Once the leak is removed the transformer is *slightly worse than forecasting zero*. It beats
naive persistence, and that is the whole of its return skill. This is the expected outcome for
daily equity returns given the 0.019 R² above, but it should be stated rather than implied.

**Removing the leaky features improved two heads.** `yz_vol` and CVaR₅ both got better (0.89×,
0.93×). The contaminated features were not merely unhelpful for those targets, they were
actively costing accuracy — the persistent, lagged signal is the real driver and the day-*t*
terms were noise the model spent capacity on.

The task is also genuinely harder now, as it should be: training stopped at 16 epochs instead
of 50, with best val_loss 0.1736 against 0.0567.

**The calibration verdict is unchanged** — fold5 passes either way (leaky 2.42%, p=0.144;
leakage-free 3.23%, p=0.334). The leakage-free model lands on **exactly** the raw-target
backtest's numbers (3.23%, p=0.3336, same breach count), which is a third independent
confirmation that calibration behaviour is a property of the target and the offset procedure,
not of the network.

**Corrected framing.** This is a study of *post-hoc calibration of a parametric VaR estimator
under regime shift*, in which a transformer is used to approximate the estimator. The
calibration conclusions (§3–§9, §11) stand. The forecasting-accuracy conclusions do not.

### The complete corrected accuracy table

All five folds, the single split, and the §10 ablation were retrained leakage-free — 11
further training runs. Every figure below comes from `scripts/leakage_free_retrain.py`,
`scripts/leakage_free_ablation.py` and `scripts/leakage_free_report.py`.

**Per-fold MAE, leaky → leakage-free (ratio):**

| Fold | log_return | log_volume | yz_vol | VaR₅ | CVaR₅ |
|---|---|---|---|---|---|
| 2008 GFC | 0.014957 → 0.015541 (1.04×) | 0.571 → 0.575 (1.01×) | 2.38e-4 → 2.36e-4 (0.99×) | 0.005589 → 0.005354 (**0.96×**) | 0.016764 → 0.017443 (1.04×) |
| 2011 EU crisis | 0.006027 → 0.007770 (1.29×) | 0.170 → 0.162 (**0.95×**) | 5.1e-5 → 3.3e-5 (**0.65×**) | 0.003309 → 0.003856 (1.17×) | 0.004172 → 0.004187 (1.00×) |
| 2015 selloff | 0.002766 → 0.005349 (1.93×) | 0.121 → 0.133 (1.10×) | 1.3e-5 → 1.5e-5 (1.10×) | 0.000913 → 0.002523 (**2.76×**) | 0.001684 → 0.002489 (1.48×) |
| 2020 COVID | 0.005455 → 0.009576 (1.76×) | 0.165 → 0.145 (**0.88×**) | 5.8e-5 → 5.3e-5 (**0.92×**) | 0.001974 → 0.004242 (2.15×) | 0.005307 → 0.006324 (1.19×) |
| 2022 bear | 0.003859 → 0.012851 (**3.33×**) | 0.107 → 0.107 (1.00×) | 3.6e-5 → 3.2e-5 (**0.89×**) | 0.001315 → 0.004932 (**3.75×**) | 0.003304 → 0.003089 (**0.93×**) |

**Single split** (this one also fixes a second, separate flaw — the original fits the feature
scaler on the whole dataset, a documented quirk inherited from the notebook; the leakage-free
version fits train-only, as the folds always did, so it differs in two respects):

| Task | Leaky | Leakage-free |
|---|---|---|
| log_return | 0.00423 | 0.009684 |
| log_volume | 0.13237 | 0.156558 |
| yz_vol | 0.00004 | 0.000040 |
| VaR₅ | 0.00150 | 0.004194 |
| CVaR₅ | 0.00384 | 0.004705 |

### Three things the full re-run establishes

**1. The return head has no forecasting skill on any fold.** Against the naive baselines on
each fold's own evaluation window:

| Fold | Leakage-free MAE | Predict zero | Persistence | Beats zero? |
|---|---|---|---|---|
| 2008 GFC | 0.015541 | 0.014925 | 0.023012 | **no** |
| 2011 EU crisis | 0.007770 | 0.007250 | 0.010710 | **no** |
| 2015 selloff | 0.005349 | 0.005297 | 0.007902 | **no** |
| 2020 COVID | 0.009576 | 0.009560 | 0.014664 | **no** |
| 2022 bear | 0.012851 | 0.012040 | 0.017191 | **no** |

**0 of 5.** The model loses to a constant zero forecast on every fold, by a small but
consistent margin. It beats naive persistence everywhere, but persistence is a poor baseline
for daily returns. This is the honest summary of the return head: no skill, as the 0.019
lagged R² predicted.

**2. The leakage's cost scales with training-set size.** GFC (924 training sequences) barely
moved at 1.04×; 2022 bear (4,450) degraded 3.33×. The model needs enough data to *learn* to
exploit the leak, so the smallest fold — already performing at roughly the predict-zero level —
had the least to lose. 2020 COVID is the one inversion in an otherwise monotone pattern.
Several volatility and volume heads got *better* once the contaminated features were removed.

**3. Calibration verdicts are unchanged, 5 of 5, and still match the model-free backtest.**

| Fold | Leakage-free model | Raw target, no model | Agree |
|---|---|---|---|
| 2008 GFC | 1.59%, p=0.00406 FAIL | 1.20%, p=0.000966 FAIL | yes |
| 2011 EU crisis | 0.80%, p=4.15758e-06 FAIL | 0.80%, p=4.15758e-06 FAIL | yes |
| 2015 selloff | 1.06%, p=2.42255e-05 FAIL | 0.80%, p=3.97404e-06 FAIL | yes |
| 2020 COVID | 13.94%, p=6.95e-08 FAIL | 18.73%, p=7.55e-15 FAIL | yes |
| 2022 bear | 3.23%, p=0.3336 PASS | 3.23%, p=0.3336 PASS | yes |

The leakage-free model reproduces the *leaky* model's numbers exactly on GFC and EU crisis —
identical breach counts. Across every variant tested — the leaky model, the leakage-free
retrain, the single-task ablation and the model-free raw-target backtest — the **verdicts**
agree on all 5 folds. Breach counts are identical across all four only for EU crisis; on the
other folds the variants land on different coverages that fall on the same side of the Kupiec
line. The thesis therefore rests on verdict agreement across independent variants, not on any
one model's numbers.

*(Correction: an earlier version of this paragraph said GFC and EU crisis showed "identical
breach counts across three different models". That overstated it — on GFC the raw-target
backtest gives 1.20% against both models' 1.59%. What holds on every fold is the verdict.)*

### Two results that did change

**The single-split verdict flips: FAIL → PASS** (2.76%, p=0.0152 → **4.26%, p=0.448**). This
does not rescue the single split as evidence; it condemns it further. §11 already showed that
verdict is seed-fragile, with 3 of 4 alternate seeds flipping fail→pass. It now also flips
under a feature-timing change. The walk-forward folds remain the only trustworthy evidence.

**§10's ablation conclusion does not survive.** On the leaky features, single-task VaR-only
calibrated better on the single split and in 3 of 5 folds, which was reported as a real
finding: "the multi-task burden costs VaR calibration." Leakage-free:

| Case | 5-head MTL | Single-task | Direction |
|---|---|---|---|
| Single split | 4.26%, p=0.448 | 4.68%, p=0.748 | single better |
| 2008 GFC | 1.59%, p=0.00406 | 1.99%, p=0.0133 | single better |
| 2011 EU crisis | 0.80%, p=4.15758e-06 | 0.80%, p=4.15758e-06 | **identical** |
| 2015 selloff | 1.06%, p=2.42255e-05 | 1.06%, p=2.42255e-05 | **identical** |
| 2020 COVID | 13.94%, p=6.95e-08 | 15.94%, p=1.43e-10 | single worse |
| 2022 bear | 3.23%, p=0.3336 | 2.42%, p=0.1444 | single worse |

**1 better, 2 identical, 2 worse.** The pre-registration for this re-run said a reversal was a
real possible outcome and would be reported as one; it was. §10's conclusion was an artifact of
the leaky feature set and should be withdrawn — the honest statement is that head count makes
no consistent difference to VaR calibration, which is what the model-independence of the whole
calibration result would predict anyway.

### The recalibration experiments, re-run on the leakage-free model

Rolling recalibration (§4), the volatility-ratio monitor (§5) and the window-length sweep (§8) were
computed on the original model's predictions, so they inherited the leak. They were re-run twice.

**First on the raw VaR₅ target, with no model** (`scripts/model_free_recalibration.py`). That answers
whether a model is needed to reach the conclusions, but it is not an ex-ante backtest. The target is an
EWMA of returns *through day t* minus a volatility term that includes day t, so the value tested against
r_t carries roughly 18% of r_t itself. Two of its three pre-registered checks passed; the monitor check
failed on GFC (5.7% on elevated readings vs 6.1% on normal ones).

**Then on the leakage-free model** (`scripts/recalibration_leakage_free.py`), the ex-ante version: every
forecast of day t uses only information through day t−1. Its predictions come from
`mtl-var walk-forward --feature-set leakage_free`, gated on being bit-identical to the leakage-free runs
above — they were, 14 values on each of the 5 folds. The same three criteria were applied unchanged. They
were written before any leakage-free recalibration number existed, though after the raw-target results
had been seen.

| Pre-registered check, on the leakage-free model | Result |
|---|---|
| 1. Rolling recalibration: GFC fails, the other four pass | **pass** |
| 2. L=40 rescues GFC and regresses a fold that passes at L=125 | **pass** — rescues GFC (4.78%, p=0.873), regresses COVID (9.56%, p=0.003); no window length passes every fold |
| 3. Monitor: elevated above normal in both shock folds, below in both grind folds | **pass** — GFC 7.0% vs 5.4%, COVID 6.8% vs 5.4%; EU crisis 3.2% vs 7.2%, 2015 selloff 3.3% vs 5.9% |

Window sweep on the leakage-free model (calibrated coverage, Kupiec p):

| Fold | L=40 | L=60 | L=90 | L=125 | L=175 |
|---|---|---|---|---|---|
| 2008 GFC | **4.78%, 0.873** | 2.39%, 0.036 | 1.99%, 0.013 | 0.80%, 0.00016 | 0.00%, 3.9e-07 |
| 2011 EU crisis | 5.60%, 0.601 | 4.53%, 0.674 | 4.53%, 0.674 | 4.27%, 0.504 | 4.53%, 0.674 |
| 2015 selloff | 6.91%, 0.106 | 6.12%, 0.336 | 5.05%, 0.962 | 4.26%, 0.497 | 3.19%, 0.086 |
| 2020 COVID | 9.56%, 0.003 | 9.56%, 0.003 | 7.17%, 0.137 | 7.17%, 0.137 | 7.17%, 0.137 |
| 2022 bear | 4.03%, 0.609 | 4.84%, 0.934 | 4.03%, 0.609 | 3.28%, 0.353 | 2.78%, 0.347 |

Cell by cell, these verdicts agree with the original model in 22 of 25 cells and with the raw target in
24 of 25.

**What this changes.** The three conclusions stand on ex-ante predictions: rolling recalibration rescues
four folds but not GFC; a short window rescues GFC at the cost of COVID, and no static window passes every
fold; the monitor leads acute shocks and inverts on slow grinds. One detail is withdrawn: §8 reported that
2015 selloff fails at both ends of the sweep, and those failures were marginal (p=0.042, p=0.046) and do
not survive either the leakage-free model or the raw target. Two smaller differences are recorded rather
than smoothed over. COVID passes at L=90 on the leakage-free model where the original model failed. And the
raw target does not reproduce GFC's monitor pattern — which, since that target is not ex-ante, says more
about the raw target than about the monitor.

### Registry: the corrected model replaces the leaky one

The leakage-free fold5 model was retrained through the tracked package pipeline and came out
**bit-identical** to the run above — all 21 recorded metrics, including best val_loss
0.1735555361956358 and calibration offset −0.02196028854997308. It is registered as
`mtl-var-transformer` **version 2**, tagged `feature_set=leakage_free`,
`leakage_status=corrected`, `supersedes_version=1` and `git_commit=db12f1e`, and `@production`
points at it. The serving layer now refuses to load any version whose feature set is not
leakage-free, and serves an honest one-step-ahead forecast: bars through day D in, a forecast for
the next trading day out.

Version 1 — the original fold5 model, trained on the leaky features — was **retired because it is
superseded**, so it cannot be served by accident. It was not erased. Its training run and logged
model remain in the tracking store and still load; the registered model's own description records
the retirement and why; and this section and the git history keep the finding.

## 14. ACAI 2026 review: revisions run on the leakage-free model

The paper received conditional acceptance with six revisions. The code-side ones are below. Every
figure is computed from integer breach counts N and T and shown with them. Pre-registered readings
are in the docstrings of `scripts/reviewer_backtests.py` and `scripts/return_quantile_control.py`,
written before those scripts produced any number.

> **Superseded for every neural-model number: see §15.** The models behind this section's transformer
> and return-quantile figures used the test window for early stopping. §15 retrains all of them with
> early stopping on the training tail. Every frozen-offset, rolling and sweep verdict survives that
> retrain except one marginal sweep cell. The return-quantile head's raw coverage (C1) does **not**
> survive. The classical baselines (14.3) and the 14.1 correction don't involve training and are
> unchanged.

### 14.1 Correction: Kupiec at zero breaches is a rejection, not "n/a" (reviewer item 1)

`kupiec_from_counts` returned NaN at N = 0 and N = T. That was wrong: taking 0·log 0 = 0, the
unrestricted likelihood is exactly 1, so LR = −2T·ln(1−p) at N = 0 (−2T·ln p at N = T). For 0 breaches
in 251 days, LR = −2·251·ln 0.95 = **25.75, p = 3.9×10⁻⁷**. The function now computes every statistic in
log space with `xlogy`, which builds in that convention, and uses `chi2.sf`. Regression tests are in
`tests/test_kupiec.py`.

The cells that were "n/a" are now failures. All are 0/251 on the 2008 GFC evaluation half:

| Cell | Was | Now |
|---|---|---|
| Window sweep L=175, GFC (leakage-free model; also original model and §13 table) | 0.00%, n/a | 0/251, LR 25.75, p = 3.9×10⁻⁷, FAIL |
| Empirical VaR N=250 calibrated, GFC (§9) | 0.00%, n/a | 0/251, p = 3.9×10⁻⁷, FAIL |
| Empirical VaR N=125 calibrated, GFC (§9 sensitivity) | 0.0%, p=nan | 0/251, p = 3.9×10⁻⁷, FAIL |

The three affected scripts were re-run, and every other value in their JSON outputs was checked against
the committed version. No other number moved by more than 1e-9 relative. The pass/fail counts were
already right, because the scripts counted NaN as a failure. Only the reported p-values change.
The live breach monitor (`src/mtl_var/monitoring.py`) had its own boundary workaround, which now
just calls the corrected function. `results/empirical_var_comparison_results_buggy_loss.json` is a historical backup
and was left as it was.

### 14.2 Christoffersen independence and conditional coverage (reviewer item 3)

Added to `mtl_var.backtest`: LR_ind (first-order Markov alternative, from the 2×2 transition counts),
LR_cc = LR_POF + LR_ind on χ²(2), and a Wilson 95% interval on N/T. Tests are in
`tests/test_christoffersen.py`. The table covers the leakage-free model, frozen offset, evaluation half:

| Fold | Eval dates | N/T | Coverage [Wilson 95%] | LR_POF / p | n01, n11 | LR_ind / p | LR_cc / p | Verdict |
|---|---|---|---|---|---|---|---|---|
| 2008 GFC | 2009-01-05 – 2009-12-31 | 4/251 | 1.59% [0.6, 4.0] | 8.26 / 0.0041 | 4, 0 | 0.13 / 0.72 | 8.39 / 0.015 | reject POF, CC |
| 2011 EU crisis | 2012-07-05 – 2013-12-31 | 3/375 | 0.80% [0.3, 2.3] | 21.19 / 4.2×10⁻⁶ | 3, 0 | 0.05 / 0.83 | 21.24 / 2.4×10⁻⁵ | reject POF, CC |
| 2015 selloff | 2016-07-06 – 2017-12-29 | 4/376 | 1.06% [0.4, 2.7] | 17.82 / 2.4×10⁻⁵ | 4, 0 | 0.09 / 0.77 | 17.91 / 1.3×10⁻⁴ | reject POF, CC |
| 2020 COVID | 2020-01-06 – 2020-12-31 | 35/251 | 13.94% [10.2, 18.8] | 29.08 / 7.0×10⁻⁸ | 31, 4 | 0.23 / 0.63 | 29.31 / 4.3×10⁻⁷ | reject POF, CC |
| 2022 bear | 2022-07-07 – 2022-12-30 | 4/124 | 3.23% [1.3, 8.0] | 0.93 / 0.33 | 4, 0 | 0.27 / 0.60 | 1.20 / 0.55 | no rejection |

Rolling recalibration (L=125, R=21), same evaluation slice:

| Fold | N/T | Coverage [Wilson 95%] | p_POF | p_ind | p_cc | Verdict |
|---|---|---|---|---|---|---|
| 2008 GFC | 2/251 | 0.80% [0.2, 2.9] | 1.6×10⁻⁴ | 0.86 | 8.1×10⁻⁴ | reject |
| 2011 EU crisis | 16/375 | 4.27% [2.6, 6.8] | 0.50 | 0.71 | 0.75 | no rejection |
| 2015 selloff | 16/376 | 4.26% [2.6, 6.8] | 0.50 | 0.23 | 0.39 | no rejection |
| 2020 COVID | 18/251 | 7.17% [4.6, 11.0] | 0.14 | 0.53 | 0.27 | no rejection |
| 2022 bear | 4/122 | 3.28% [1.3, 8.1] | 0.35 | 0.60 | 0.57 | no rejection |

Against the pre-registration:

- **R3a** (folds that fail Kupiec also fail LR_cc): yes, all four. As expected, this adds little.
- **R3b(i), 2022:** no test rejects. Its Wilson interval, 1.3% to 8.0%, contains 5%. It also contains
  GFC's 1.59%, which is a clear failure. With 124 days the fold can't tell either rate apart from 5%. The
  paper should say "not rejected", never "correctly calibrated".
- **R3b(ii), rolling rescues:** all four survive LR_ind and LR_cc. The "rolling recalibration rescues
  4/5" claim stands under the fuller tests.
- **Clustering is not where the calibrated model fails.** LR_ind never rejects for the transformer after
  either calibration (smallest p = 0.23). The four failures are frequency failures: too few breaches in
  GFC, EU crisis and 2015, too many in COVID. Independence rejects only in places like the transformer's
  raw COVID output (p = 0.015) and HS-250 in COVID.
- **Exceedance severity** (mean / worst return minus VaR on breach days, frozen offset): GFC −0.40% /
  −0.80%, EU −0.44% / −0.54%, 2015 −0.39% / −1.04%, COVID −1.91% / **−10.92%**, 2022 −0.77% / −2.10%.
  A formal expected-shortfall backtest was not run. The pipeline produces no ES forecast that goes with
  the calibrated VaR (the CVaR head is trained on its own proxy and is not offset-calibrated), so the
  severity figures are descriptive only.

### 14.3 Classical VaR under the identical five-fold protocol (reviewer item 4)

`scripts/reviewer_backtests.py` runs three genuine one-step-ahead 5% quantile estimators on the
transformer's exact evaluation dates. The script asserts that the realised returns are identical. None
of the estimators is tuned to the folds:

- **HS-250**: 5th percentile of the previous 250 log returns.
- **EWMA**: RiskMetrics λ = 0.94, Gaussian, zero mean.
- **GARCH(1,1)**: Gaussian, constant mean, fitted once per fold on that fold's training period only,
  then filtered forward with parameters held fixed.

Each is run raw, with the frozen half/half offset, and with rolling L=125, R=21. Verdict at 5% (reject
if POF or CC rejects; full statistics are in `results/reviewer_backtests_results.json`):

| Fold | Transformer raw | Transformer frozen | HS raw | HS frozen | EWMA raw | EWMA frozen | GARCH raw | GARCH frozen |
|---|---|---|---|---|---|---|---|---|
| 2008 GFC | 75/251 ✗ | 4/251 ✗ | 2/251 ✗ | 0/251 ✗ | 13/251 ✓ | 8/251 ✓ | 18/251 ✓ | 8/251 ✓ |
| 2011 EU crisis | 116/375 ✗ | 3/375 ✗ | 12/375 ✓ | 6/375 ✗ | 18/375 ✓ | 8/375 ✗ | 16/375 ✓ | 10/375 ✗ (POF only, p_cc 0.058) |
| 2015 selloff | 118/376 ✗ | 4/376 ✗ | 9/376 ✗ | 8/376 ✗ | 13/376 ✓ | 12/376 ✓ | 8/376 ✗ | 6/376 ✗ |
| 2020 COVID | 92/251 ✗ | 35/251 ✗ | 21/251 ✗ | 29/251 ✗ | 15/251 ✓ | 17/251 ✗ (IND+CC) | 17/251 ✓ | 18/251 ✓ |
| 2022 bear | 51/124 ✗ | 4/124 ✓ | 3/124 ✓ | 1/124 ✗ | 5/124 ✓ | 2/124 ✗ (POF p 0.045) | 7/124 ✓ | 4/124 ✓ |
| **Rejected** | 5/5 | 4/5 | 3/5 | **5/5** | **0/5** | 3/5 | 1/5 | 2/5 |

With rolling recalibration, EWMA and GARCH are rejected in **0/5** folds, HS-250 in 3/5 (2015, COVID,
2022), and the transformer in 1/5 (GFC).

Against the pre-registration:

- **R4a, not met for EWMA or GARCH.** Under the frozen offset, HS-250 is rejected in all four folds
  where the transformer is rejected. EWMA and GARCH are rejected in only 2 of the 4. EWMA passes GFC and
  2015, and GARCH passes GFC and COVID. So the transformer's fold-by-fold failure pattern is **not**
  reproduced by the parametric baselines, and the paper cannot say "classical methods fail the same way
  under identical conditions".
- **R4b, no narrowing triggered.** No baseline goes unrejected in 4 or more folds under the frozen offset
  (most is GARCH, 3/5).
- **R4c, causal isolation: the frozen offset did the damage, for every classical method.** Applying the
  offset never turned a rejection into a non-rejection: 0 of 15 method-fold cells. It turned a
  non-rejection into a rejection in 6 cells: HS (EU, 2022), EWMA (EU, COVID, 2022) and GARCH (EU). EWMA
  is the cleanest case. On the same dates with the same returns, it goes from **0/5 rejected raw to 3/5
  rejected after the offset**. That is the direct evidence the reviewer said was missing.
- **A plain, uncalibrated RiskMetrics EWMA is not rejected in any fold.** On these folds it is
  better than the transformer pipeline, which needs an offset at all only because its raw output breaches
  30–41% of days (see 14.4). This has to be reported in the paper, not left out.

So what the reviewer asked for comes out mixed, and it should be written that way. The causal claim about
the calibration step gets stronger: a frozen offset fitted in one regime degrades well-specified
estimators in the next regime and never helps them. The "model-independent failure pattern" claim gets
weaker: which folds fail depends on the estimator. Under the frozen offset the transformer (4/5 rejected)
beats only HS-250 (5/5). EWMA (3/5) and GARCH (2/5) both do better.

**Multiplicity in the window sweep (optional item).** Five window lengths were tried per fold. At a
Bonferroni threshold of 0.05/5 = 0.01, **L=90 has no Kupiec rejection in any fold**. GFC's p = 0.013 no
longer counts as a rejection. So "no static window passes every fold" holds only at the unadjusted 5%
level. At 5%, L=90 fails GFC on Kupiec (p = 0.013) and fails GFC and COVID on LR_cc (p = 0.042, 0.034).
The sweep was also chosen after looking at the folds, so it is exploratory. One correction to the
denominators: rolling cannot start until day L, so the L=175 cell for 2022 is **2/72**, not out of 124.
The paper's 2.78% is 2/72.

### 14.4 What the VaR target actually is, and a direct return-quantile control (reviewer item 2)

`VaR_5 = EWMA_span10(log_return_t) − 1.2·yz_vol_t`. `yz_vol` is the Yang-Zhang **variance** (a 20-day
mean of squared log returns, median 5.2×10⁻⁵), not a volatility. So the 1.2 term is in return² units
and is tiny: median 6.2×10⁻⁵, against a typical daily return magnitude of about 7×10⁻³. Over the full
sample:

- the EWMA term explains **99.3%** of the target's variance;
- realised returns fall below the target on **51.0%** of days (in-sample, same day);
- a unit-consistent version, EWMA − 1.645·√yz_vol, would be breached on **4.93%** of days.

The constructed target is a smoothed return, not a 5% quantile. Pinball loss at q = 0.05 then estimates
the conditional 5% quantile *of that proxy*. That is why the transformer's raw output breaches 30–41% of
days, and why the post-hoc offset does essentially all the work. No statistical VaR guarantee holds
before calibration. The source of the 1.2 factor is not documented anywhere in the repository. Time
indexing on the leakage-free features: every input in row t is known by the close of t−1, and the target
uses information through day t, so the forecast is one step ahead.

**Control** (`scripts/return_quantile_control.py`). This is the identical single-task model (§10 body,
one head, pinball q = 0.05, seed 42) on the leakage-free features. The only change is the target: the
**realised log return**, not the proxy. The script asserts that the calendar, returns and calib_n match
the primary model. Predictions are cached in `results/predictions_return_quantile/`.

| Fold | Epochs | Raw N/T (%) | Frozen N/T (%) | Frozen p_POF / p_cc | Rolling N/T (%) | Rolling p_POF / p_cc |
|---|---|---|---|---|---|---|
| 2008 GFC | 7 | 18/251 (7.17) ✓ | 3/251 (1.20) | 9.7×10⁻⁴ / 0.0042 ✗ | 2/251 (0.80) | 1.6×10⁻⁴ / 8.1×10⁻⁴ ✗ |
| 2011 EU crisis | 13 | 7/375 (1.87) ✗ | 6/375 (1.60) | 4.6×10⁻⁴ / 0.0020 ✗ | 18/375 (4.80) | 0.86 / 0.53 ✓ |
| 2015 selloff | 8 | 12/376 (3.19) ✓ | 6/376 (1.60) | 4.4×10⁻⁴ / 0.0019 ✗ | 14/376 (3.72) | 0.24 / 0.29 ✓ |
| 2020 COVID | 15 | 23/251 (9.16) ✗ | 21/251 (8.37) | 0.025 / 0.080 ✗ (POF) | 16/251 (6.37) | 0.34 / 0.41 ✓ |
| 2022 bear | 6 | 4/124 (3.23) ✓ | 5/124 (4.03) | 0.61 / 0.34 ✓ | 6/122 (4.92) | 0.97 / 0.55 ✓ |

- **C1** (*withdrawn in §15*): raw coverage is 1.9–9.2%, against 30–41% for the proxy-trained head.
  Raw output is rejected in 2 of 5 folds: EU crisis (too few breaches) and COVID (too many). After the
  early-stopping fix, raw coverage is 1.1–14.5% and is rejected in 5/5 folds. The good raw numbers came
  from choosing the epoch by pinball loss on the test returns.
- **C2, met.** Under the frozen offset the return-quantile head is rejected in **4 of the 4** folds where
  the proxy model is rejected (GFC, EU crisis, 2015, COVID). It passes 2022, as the proxy model does.
  Rolling recalibration again rescues every fold except GFC. **The regime-conditioned failure of the
  frozen calibration persists when the head targets the return quantile directly.** The fold-level
  verdicts are the same as the proxy model's in every column.
- **C3:** the frozen offset turned a non-rejection into a rejection in GFC and 2015, and never the other
  way round. This matches the classical baselines in 14.3.

Read together with 14.3, C2 holds for the transformer under either target. For the classical methods,
the frozen offset never repaired a raw rejection, and it broke well-behaved raw output in 6 method-fold
cells. (This paragraph originally said "every method tested" and counted 8 cells. That included this
head's GFC and 2015 cells, which §15 withdraws. After the fix, the head's raw output is rejected
everywhere, and the frozen offset repairs it in 2022.) EWMA and GARCH also show that the per-fold pattern
is not universal across estimator families.

### 14.5 Reproducibility facts (reviewer item 6)

- **Data:** `SPX500-18Y_scaled_y.csv` in the repository: S&P 500 daily OHLCV with Adj Close, raw rows
  2004-01-02 to 2022-12-30. The provider is **not recorded** in the repository. The column layout
  matches a Yahoo Finance ^GSPC export, but that needs confirming before it goes in the paper. After the
  rolling-window NaN cascade, 4705 usable rows run from 2004-04-26 (one fewer for leakage-free lags).
- **Alignment:** sequences are 5 rows. Sequence j covers rows j..j+4 and predicts row j+4's targets.
  Leakage-free features replace the 8 day-t features with their one-day lags (`LEAKAGE_FREE_FEATURES`
  in `src/mtl_var/data.py`). `tests/test_leakage_free_features.py` holds that construction fixed.
- **Folds:** expanding training window from 2004-01-01. The tests are 2008–09, 2011–13, 2015–17,
  2019–20 and 2022. The first 4 trading days of each test window are used up by the sequence length.

  | Fold | Cached test window | Calibration half | Evaluation half | Frozen breaches |
  |---|---|---|---|---|
  | 2008 GFC | 2008-01-08 – 2009-12-31 | 2008-01-08 – 2009-01-02 (250 d) | 2009-01-05 – 2009-12-31 (251 d) | 4/251 |
  | 2011 EU crisis | 2011-01-07 – 2013-12-31 | 2011-01-07 – 2012-07-03 (375 d) | 2012-07-05 – 2013-12-31 (375 d) | 3/375 |
  | 2015 selloff | 2015-01-08 – 2017-12-29 | 2015-01-08 – 2016-07-05 (375 d) | 2016-07-06 – 2017-12-29 (376 d) | 4/376 |
  | 2020 COVID | 2019-01-08 – 2020-12-31 | 2019-01-08 – 2020-01-03 (250 d) | 2020-01-06 – 2020-12-31 (251 d) | 35/251 (41/251 after §15) |
  | 2022 bear | 2022-01-07 – 2022-12-30 | 2022-01-07 – 2022-07-06 (123 d) | 2022-07-07 – 2022-12-30 (124 d) | 4/124 |

- **Calibration:** frozen offset = 5th percentile of (r − VaR) on the first ⌊T/2⌋ days, added to the
  second half. Rolling: L = 125 trailing days, refreshed every R = 21 days, evaluated on the same second
  half. Window sweep: L ∈ {40, 60, 90, 125, 175}.
- **Monitor:** 10-day live volatility over the volatility of the active offset window, firing at a
  ratio ≥ 1.5. Live breach monitor: 250-day window, 100-observation minimum.
- **Seeds and determinism:** seed 42 (`mtl_var.seeding.SEED`), with `tf.config.experimental.enable_op_determinism()`.
- **Breach rule:** r_t < VaR_t, strictly.
- **Code:** github.com/ManasviMenon/Multi-Task-Model-with-Transformers. The paper should cite the
  commit that contains this section.
- **Early stopping (fixed in §15):** every neural run in this section used the fold's test window as
  its early-stopping validation set. §15 replaces that with the last 15% of each fold's training
  sequences and retrains every model.

## 15. Early stopping moved off the test window: all neural models retrained

**The problem.** `train_model` and `train_single_task` were always called with the fold's test window
as `X_val`, so early stopping and best-weight restoration saw test-period targets. Features and targets
were leakage-free, but model selection was not.

**The fix.**
- Early stopping now uses the last 15% of each fold's training sequences, in time order, with no
  shuffling (`mtl_var.model.chronological_val_split`). The 15% was fixed before any result existed.
- The pipeline trains only through `fit_with_train_tail_validation(X_train, y_train_z, n_features)`.
  That function has no parameter that could receive test data. The test window is first touched by
  `model.predict` after training ends.
- Everything else is unchanged: architecture, losses, 0.2 weights, Adam 1e-4, batch 32, patience 5,
  seed 42, 5-day sequences, leakage-free features. Scalers are still fit on the whole training period.
- `tests/test_no_test_window_in_training.py` enforces the fix in three ways:
  - no training function has a parameter or identifier containing "test";
  - the validation split is the chronological tail;
  - a stubbed end-to-end fold run hands the trainer exactly the training sequences.
- New outputs go to `*_cleanval` paths, so they cannot overwrite the recorded runs. Byte-for-byte copies
  of the old leakage-free results are in `results/leaky_earlystop/`.
- The pre-registered readings P1–P5 are in `scripts/cleanval_report.py`, written before the first
  retrain. Training ran with MLflow tracking off. The served model (registry version 2) was trained the
  old way and has **not** been re-registered.

### 15.1 Five-fold MTL retrain (frozen offset)

| Fold | Epochs old→new | n_fit / n_val | Old N/T, p | New N/T, p | Verdict |
|---|---|---|---|---|---|
| 2008 GFC | 7 → 10 | 785 / 138 | 4/251, 0.00406 | 4/251, 0.00406 | FAIL → FAIL |
| 2011 EU crisis | 19 → 24 | 1428 / 252 | 3/375, 4.16e-06 | 3/375, 4.16e-06 | FAIL → FAIL |
| 2015 selloff | 16 → 18 | 2283 / 403 | 4/376, 2.42e-05 | 4/376, 2.42e-05 | FAIL → FAIL |
| 2020 COVID | 29 → 19 | 3138 / 554 | 35/251, 6.95e-08 | **41/251, 3.8e-11** | FAIL → FAIL |
| 2022 bear | 16 → 21 | 3782 / 667 | 4/124, 0.334 | 4/124, 0.334 | pass → pass |

**P1: no verdict flips.** Four folds land on identical breach counts even though the VaR predictions
changed (VaR MAE moves 0.80× to 1.25×). That is consistent with §13: the counts are driven mainly by the
target and the offset, not the network. GFC's old "validation" loss of 2.40 was the 2008–09 crisis itself.
Its new one, on 2007, is 0.43.

Raw (uncalibrated) breaches, old → new: 75→93/251, 116→109/375, 118→88/376, 92→93/251, 51→55/124. All are
still rejected.

Per-task MAE, old → new (ratio):

| Fold | log_return | log_volume | yz_vol | VaR₅ | CVaR₅ |
|---|---|---|---|---|---|
| 2008 GFC | 0.01554 → 0.01518 (0.98×) | 0.5753 → 0.6846 (1.19×) | 2.36e-4 → 2.55e-4 (1.08×) | 0.005354 → 0.004268 (0.80×) | 0.01744 → 0.01985 (1.14×) |
| 2011 EU crisis | 0.00777 → 0.007255 (0.93×) | 0.1618 → 0.1308 (0.81×) | 3.30e-5 → 2.97e-5 (0.90×) | 0.003856 → 0.004185 (1.09×) | 0.004187 → 0.00392 (0.94×) |
| 2015 selloff | 0.005349 → 0.005392 (1.01×) | 0.1331 → 0.1125 (0.85×) | 1.46e-5 → 2.06e-5 (1.41×) | 0.002523 → 0.003162 (1.25×) | 0.002489 → 0.002915 (1.17×) |
| 2020 COVID | 0.009576 → 0.009581 (1.00×) | 0.1446 → 0.1616 (1.12×) | 5.31e-5 → 7.29e-5 (1.37×) | 0.004242 → 0.004071 (0.96×) | 0.006324 → 0.007316 (1.16×) |
| 2022 bear | 0.01285 → 0.01248 (0.97×) | 0.1071 → 0.1297 (1.21×) | 3.20e-5 → 2.34e-5 (0.73×) | 0.004932 → 0.004433 (0.90×) | 0.003089 → 0.003182 (1.03×) |

Accuracy moves both ways with no consistent direction. The return head still has no skill: GFC 0.01518 vs
predict-zero 0.014925, 2022 0.01248 vs 0.01204.

### 15.2 Rolling recalibration and the window sweep

Rolling, L=125, R=21 (old → new): GFC 2/251 → 2/251 (reject); EU 16/375 → 17/375; 2015 16/376 → 14/376;
COVID 18/251 → 19/251 (p_POF 0.082, p_cc 0.100); 2022 4/122 → 4/122. **P2a holds: rolling rescues every
fold except 2008.**

Window sweep, new (old), N/T and Kupiec p; ✗ = rejected at 5%:

| Fold | L=40 | L=60 | L=90 | L=125 | L=175 |
|---|---|---|---|---|---|
| 2008 GFC | 12/251, 0.873 (same) | 6/251, 0.036 ✗ (same) | 5/251, 0.013 ✗ (same) | 2/251, 1.6e-4 ✗ (same) | 0/251, 3.9e-7 ✗ (same) |
| 2011 EU crisis | 23/375, 0.33 (21, 0.60) | 19/375, 0.95 (17, 0.67) | 17/375, 0.67 (same) | 17/375, 0.67 (16, 0.50) | 15/375, 0.36 (17, 0.67) |
| 2015 selloff | 26/376, 0.106 (same) | 24/376, 0.24 (23, 0.34) | 19/376, 0.96 (same) | 14/376, 0.24 (16, 0.50) | 13/376, 0.15 (12, 0.086) |
| 2020 COVID | 26/251, 6.1e-4 ✗ (24, 0.003 ✗) | 24/251, 0.003 ✗ (same) | 18/251, 0.137 (same) | 19/251, 0.082 (18, 0.137) | **20/251, 0.046 ✗** (18, 0.137) |
| 2022 bear | 6/124, 0.93 (5, 0.61) | 6/124, 0.93 (same) | 4/124, 0.33 (5, 0.61) | 4/122, 0.35 (same) | 2/72, 0.35 (same) |

**P2b holds: no single window passes every fold at 5%.** One cell changed verdict: COVID at L=175 now
fails, marginally. The multiplicity caveat from 14.3 is unchanged: at a Bonferroni threshold of 0.01,
L=90 is rejected nowhere.

### 15.3 Christoffersen, frozen offset (new models)

| Fold | N/T | Coverage [Wilson 95%] | LR_POF / p | n01, n11 | LR_ind / p | LR_cc / p | Verdict |
|---|---|---|---|---|---|---|---|
| 2008 GFC | 4/251 | 1.59% [0.6, 4.0] | 8.26 / 0.0041 | 4, 0 | 0.13 / 0.72 | 8.39 / 0.015 | reject POF, CC |
| 2011 EU crisis | 3/375 | 0.80% [0.3, 2.3] | 21.19 / 4.2e-6 | 3, 0 | 0.05 / 0.83 | 21.24 / 2.4e-5 | reject POF, CC |
| 2015 selloff | 4/376 | 1.06% [0.4, 2.7] | 17.82 / 2.4e-5 | 4, 0 | 0.09 / 0.77 | 17.91 / 1.3e-4 | reject POF, CC |
| 2020 COVID | 41/251 | 16.33% [12.3, 21.4] | 43.71 / 3.8e-11 | 34, 7 | 0.02 / 0.90 | 43.73 / 3.2e-10 | reject POF, CC |
| 2022 bear | 4/124 | 3.23% [1.3, 8.0] | 0.93 / 0.33 | 4, 0 | 0.27 / 0.60 | 1.20 / 0.55 | no rejection |

**P3 holds.** 2022 is still rejected by nothing. Independence still never rejects (smallest p = 0.60).

### 15.4 Single-task heads (VaR-proxy ablation and return-quantile control)

Frozen offset, old → new:

| Fold | VaR-proxy head | Return-quantile head |
|---|---|---|
| 2008 GFC | 5/251, p 0.013 → 4/251, p 0.0041 (FAIL) | 3/251, p 9.7e-4 → 2/251, p 1.6e-4 (FAIL) |
| 2011 EU crisis | 3/375, p 4.2e-6 → 3/375, p 4.2e-6 (FAIL) | 6/375, p 4.6e-4 → 9/375, p 0.011 (FAIL) |
| 2015 selloff | 4/376, p 2.4e-5 → 3/376, p 4.0e-6 (FAIL) | 6/376, p 4.4e-4 → 8/376, p 0.0041 (FAIL) |
| 2020 COVID | 40/251, p 1.4e-10 → 42/251, p 9.8e-12 (FAIL) | 21/251, p 0.025 → 31/251, p 5.6e-6 (FAIL) |
| 2022 bear | 3/124, p 0.144 → 5/124, p 0.61 (pass) | 5/124, p 0.61 → 11/124, p 0.073 (pass) |

- **Verdicts:** every frozen-offset verdict is unchanged for both heads.
- **P4, C2 holds:** the return-quantile head is rejected in 4/4 of the folds where the clean MTL model is
  rejected. Its rolling verdicts are also unchanged: GFC 1/251 rejected; EU 20/375, 2015 18/376,
  COVID 19/251 and 2022 8/122 not rejected.
- **Single-task vs MTL (clean, frozen):** identical breach counts in GFC (4 vs 4) and EU (3 vs 3). Single
  is worse in 2015 (3 vs 4 against 5% expected) and COVID (42 vs 41). In 2022 the counts are 5 vs 4,
  both passing. Still no consistent head-count effect, as §13 concluded.

**What does not survive: the return-quantile head's raw coverage (C1 in 14.4).** Raw breaches with clean
stopping:

| Fold | Raw N/T | p_POF |
|---|---|---|
| 2008 GFC | 32/251 (12.75%) | 2.0e-6 |
| 2011 EU crisis | 32/375 (8.53%) | 0.0042 |
| 2015 selloff | 4/376 (1.06%) | 2.4e-5 |
| 2020 COVID | 27/251 (10.76%) | 2.6e-4 |
| 2022 bear | 18/124 (14.52%) | 6.4e-5 |

That is **5/5 rejected**, against 2/5 before. Picking the epoch with the lowest pinball loss on the test
returns was, in effect, tuning test-period coverage. Without it, the return-trained head is closer to 5%
than the proxy head (30–41%), but it is not a well-calibrated quantile estimator. The 14.4 statement "C3:
the frozen offset never repaired a raw rejection" also no longer holds for this head: in 2022 it turns
18/124 (rejected) into 11/124 (not rejected).

### 15.5 Unaffected by design (P5)

The classical baselines were regenerated from the new run, and every HS-250, EWMA and GARCH entry is
identical to the old file, GARCH parameters included. The model-free raw-target backtest and the 14.1
Kupiec correction read no trained model, so they were left unchanged.

### 15.6 What changes for the paper

- **Every calibration conclusion survives:**
  - frozen offset fails 4/5 and passes 2022 ("not rejected", interval 1.3–8.0%);
  - rolling rescues all but 2008;
  - no single window passes every fold at 5%;
  - no clustering;
  - the return-quantile control keeps the frozen-offset failure.
- **Update these numbers:**
  - COVID frozen: 41/251 (16.33%), was 35/251;
  - COVID L=175 cell: now a failure (20/251, p = 0.046);
  - all per-task MAEs.
- **Withdraw:** "the return-trained head is a well-behaved quantile estimator before calibration".
- **Methods text:** state that early stopping uses the last 15% of each training period, chronologically.

## What to do next

1. **The loss-shape bug is fixed and verified across five independent checks** (§0) — this
   is no longer a caveat on the rest of the document, it's closed.
2. **The central thesis holds and strengthens under correction**: 4/5 regime-aligned folds
   still fail fixed-offset calibration (3 of those got worse, not better), and GFC's result is
   bit-for-bit identical to the buggy version across four separate configurations (fixed
   offset, rolling, event-triggered, and L=40) — strong evidence this fold's resistance is
   structural, not a training artifact.
3. **Rolling recalibration is more effective than previously measured**: 4/5 now, up from 3/5.
4. **Threshold-tightening (§6a) is a cleaner result now**: 2022 rescued at every threshold
   tested, no longer just 1.5×.
5. **Window length (§8) is still the strongest single lever, but the tradeoff got sharper**:
   L=40 now regresses two folds (2015 selloff, COVID) instead of one, and 2015 selloff shows a
   "sweet spot in the middle" (L=60–125) not visible before. No single `L` works for everyone
   — more true now than before correction, not less.
6. **The reviewer objection (§9) is answered more decisively**: empirical VaR is now strictly
   worse than the corrected parametric model at the standard lookback (5/5 fail vs. 4/5), not
   merely equal to it.
7. **Adaptive window length remains the motivated, unstarted next candidate** — the mechanism
   (different folds need different `L`, for opposite and equally valid reasons) is now
   confirmed under both the buggy and corrected models, so it isn't an artifact of the bug.
8. **Single-task VaR-only calibrates measurably better than the 5-head MTL model** (§10),
   consistent on the single split and the majority of folds — a real, pre-registered finding,
   though concentrated where MTL was already closest to passing rather than a uniform fix.
9. **The walk-forward baseline is seed-robust; the single split was the fragile one** (§11).
   4/5 folds fail at all 10 seeds tested (std as low as 0.08%), 2022 bear passes at all 10 —
   likely a statistical-power artifact of its shorter test window, not better calibration.
   §3's original single-seed framing undersold this: the finding is stronger, not weaker.
10. **Not yet multi-seeded**: the rolling/triggered/threshold/window-length sweeps (§4–§8) and
    the ablation (§10) all rest on the single seed=42 model per fold. §11 shows the
    fixed-offset baseline is robust to that choice; whether the same holds for every sweep
    built on top of it hasn't been tested and would be its own multi-hour undertaking.
11. **Target leakage invalidates every accuracy number** (§13). `log_close_ret` IS the
    `log_return` target and 8 of 18 features embed day-*t* information. All five folds, the
    single split and the ablation have now been retrained leakage-free (11 runs). Outcome: the
    calibration thesis survives untouched (5/5 verdicts unchanged, still matching a backtest
    with no model at all), the return head turns out to have **no forecasting skill on any
    fold** — it loses to predicting zero 5 times out of 5 — the single-split verdict flips
    FAIL→PASS, and §10's ablation conclusion is withdrawn as a leakage artifact. The project is
    calibration research, not return forecasting.

## Files

Repo layout (cleaned and reorganized 2026-09-03 — everything below `scripts/` and `results/`
used to sit flat in the repo root; all internal path references were updated to match, and
every script was re-run from its new location to confirm nothing broke before this was
written):

```
MTL_Improved.ipynb          the notebook — patched with the loss-shape fix + shape assertions
RESULTS.md                  this file
SPX500-18Y_scaled_y.csv     raw data
pyproject.toml              packaging: pinned deps, version, `mtl-var` console entry point
src/mtl_var/                the installable package (added in the Phase 1 packaging refactor —
                             the research code was RELOCATED here, not rewritten):
                               data.py         features, targets, load_and_engineer_features
                               model.py        losses, architecture, the training loop
                               calibration.py  frozen offset + rolling/triggered recalibration
                               monitor.py      the 10-day/125-day volatility-ratio signal
                               backtest.py     the Kupiec POF test
                               pipeline.py     FOLDS, run_fold, the walk-forward campaign
                               seeding.py      the single set_seeds() utility
tests/                      pytest suite — loss verification (§0.2), Kupiec verification
                             (§0.5), and parity tests asserting the refactor reproduces the
                             recorded results exactly
scripts/                    every investigation script — run as `python scripts/<name>.py`
                             from the repo root (not from inside scripts/) so the relative
                             paths below resolve correctly. `walk_forward.py` is now a thin
                             re-export of the package, so these all still work unchanged.
results/                    every JSON result + cached prediction, corrected and buggy-loss
                             side by side (nothing overwritten, nothing deleted)
```

**`scripts/` — corrected (current) pipeline:**
- `walk_forward.py` — patched with the loss-shape fix and permanent shape assertions; the
  5-fold campaign (fixed-offset, rolling, triggered, monitor)
- (the loss verification moved to `tests/test_loss_functions.py` in the packaging refactor —
  §0.2; it runs under pytest now rather than as a standalone script)
- `verify_loss_bug_impact.py`, `sanity_check_patched_loss.py` — the decisive retrain and the
  patch-consistency check (§0.1, §0.4)
- `reanalyze_thresholds.py`, `reanalyze_window_length.py`, `empirical_var_comparison.py` —
  post-hoc sweeps (§6a, §8, §9), read from the corrected cache, no retraining needed
- `single_task_var_ablation.py` — the MTL-vs-single-task ablation (§10)
- `classical_baselines.py` — Persistence/ARIMA(1,0,1)/GARCH(1,1)/EWMA VaR baselines on the same
  single-split test window (§12); requires `statsmodels` and `arch` (not previously project
  dependencies — installed for this)
- `leakage_audit.py` — the §13 feature audit, target-recoverability R², and the decisive
  raw-target backtest that runs the calibration protocol with no model at all
- `leakage_free_retrain.py` — retrains a fold with every day-*t* feature lagged, to measure
  what the leakage was worth
- `leakage_free_ablation.py` — leakage-free single split and the §10 single-task ablation
- `leakage_free_report.py` — assembles the corrected accuracy table, checks the return head
  against naive baselines, and re-verifies verdicts against the model-free backtest
- `model_free_recalibration.py` — rolling recalibration, the monitor and the window sweep on the raw VaR
  target with no model (§13); a model-need check, not an ex-ante backtest
- `recalibration_leakage_free.py` — the same three analyses on the leakage-free model's predictions, gated
  on bit-identical parity with §13
- `verify_seed_sensitivity.py` — single-split seed-fragility check (§11 step 1); contains the
  `train_once_no_reseed` pattern to reuse for any future custom-seed training — do NOT reuse
  `verify_loss_bug_impact.train_once`, which re-seeds internally to a hardcoded 42
- `seed_distribution_walk_forward.py` — the 50-training multi-seed distribution of the
  fixed-offset baseline (§11 step 2)

**`results/` — corrected (current):**
- `walk_forward_results.json`, `predictions/*.npz` — corrected campaign results and cached
  predictions (§3–§6 read from these)
- `empirical_var_comparison_results.json` — §9 raw results
- `single_task_ablation_results.json` — §10 raw results
- `seed_distribution_results.json` — §11 raw results (all 50 fold×seed combinations)
- `classical_baselines_results.json` — §12 raw results (Persistence/ARIMA/GARCH/EWMA VaR,
  raw and calibrated variants)
- `leakage_audit_results.json`, `leakage_free_retrain_results.json`,
  `leakage_free_ablation_results.json`, `leakage_free_summary.json` — §13 raw results
- `model_free_recalibration_results.json`, `recalibration_leakage_free_results.json`,
  `walk_forward_results_leakage_free.json`, `predictions_leakage_free/` — the §13 recalibration re-runs

**`results/` — buggy-loss backups, preserved for the before/after record:**
- `walk_forward_results_buggy_loss.json` — full buggy-trained campaign (fixed-offset, rolling,
  triggered, monitor) for every fold
- `predictions_buggy_loss/` — buggy-trained per-fold predictions
- `walk_forward_results_fixed_offset_only.json`, `walk_forward_results_rolling_scheduled_only.json`
  — earlier buggy-loss snapshots, predating even the trigger mechanism
- `empirical_var_comparison_results_buggy_loss.json` — §9 under the buggy parametric model
  (empirical-VaR numbers in it are identical to the corrected version, since that calculation
  never used the trained model)

**Removed in the cleanup** (regeneratable or unrelated, not referenced by anything above):
per-run `.log` transcripts (superseded by the JSON results + this document), `concept_model.keras`
/ `best_trans_weights.weights.h5` (gitignored model artifacts the notebook recreates on every
run), `__pycache__/`, a stray duplicate `.venv-1/`, and a few files unrelated to this
investigation (two untracked Colab-exported notebooks, a personal scratch script).

## Status

Sections 0–13 done and closed. The verification checklist (§0) is complete: decisive retrain,
independent hand-computed test, permanent shape assertions in both the notebook and
`walk_forward.py`, confirmed-identical fix in both places, and a manual spot-check of the
evaluation pipeline. The central thesis survives correction and strengthens in most respects:
"4/5 fail, 2022 passes" (§3) turned out to be robust across 10 seeds per fold, not a single
lucky/unlucky draw (§11) — the one place seed-fragility actually showed up was the single
split, which doesn't generalize to the walk-forward methodology. Rolling recalibration now
rescues 4/5 (§4). GFC's resistance to every intervention except a short window is confirmed
under both the buggy and corrected models, across four independent configurations. Single-task
VaR-only calibrates measurably better than the 5-head MTL model, consistent on the single
split and the majority of folds (§10). Adaptive window length remains the next candidate, not
yet started. The rolling/triggered/threshold/window-length sweeps and the ablation still rest
on one seed per fold — §11 established the fixed-offset baseline is seed-robust, not that
everything built on top of it is. Classical baselines (§12) are now real: MTL wins both MAE
metrics decisively but loses the single-split VaR-coverage race to both ARIMA-residual and
EWMA parametric VaR — a finding that fits the paper's thesis rather than undermining it.
§13 then found target leakage: `log_close_ret` is bit-identical to the `log_return` target, so
that MAE win is not a like-for-like comparison and the accuracy results measure reconstruction
rather than forecasting. The calibration thesis was tested against this and survives — all five
fold verdicts reproduce with no model in the loop — but the framing is now explicitly
"calibration of a parametric VaR estimator", not "a transformer that forecasts VaR".
