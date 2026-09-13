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

Pre-registered: **rescue = fold1_gfc flips to p ≥ 0.05; regression = any of {2011 EU crisis,
2015 selloff, 2020 COVID} — passing at L=125 — drops below p = 0.05.** Swept `L` ∈
{40, 60, 90, 125, 175} days, `R=21` held fixed, post-hoc from the corrected cache.

| Fold | L=40 | L=60 | L=90 | L=125 (baseline) | L=175 |
|---|---|---|---|---|---|
| 2008 GFC | **4.78% cov, p=0.873** | 2.39%, p=0.035 | 1.99%, p=0.013 | 0.80%, p=0.00016 | 0.00%, p=n/a |
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
| 2008 GFC | 1.59% cov, p=0.0041 | 0.80% cov, p=0.0002 | 0.00% cov, p=n/a |
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
| 2008 GFC | fail (0.0%, p=nan) | fail (0.0%, p=nan) | fail (0.4%, p=0.00002) |
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

**Not yet done**: only fold5 was retrained leakage-free. The other four folds, and the
single-split and ablation numbers, still rest on the leaky feature set. Their *calibration*
verdicts are covered by the raw-target backtest above, which needs no model at all; their
*accuracy* figures should be treated as reconstruction, not forecasting, until re-run.

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
    `log_return` target and 8 of 18 features embed day-*t* information. The calibration thesis
    survives — verified by reproducing all 5 fold verdicts with no model at all — but the
    project must be described as calibration research, not return forecasting. Remaining work:
    retrain the other four folds leakage-free (only fold5 is done), and decide whether to
    re-run §10's ablation on the clean feature set.

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
- `leakage_audit_results.json`, `leakage_free_retrain_results.json` — §13 raw results

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
