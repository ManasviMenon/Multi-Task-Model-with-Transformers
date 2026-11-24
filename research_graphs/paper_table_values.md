# Paper table values (clean-validation models)

Sources: `results/` files named per table. Coverage = N/T computed from integer counts.

## tab:data: fold calendar (evaluation half = second half of each cached test window)

| Fold | Train | Test (cached) | Calibration half | Evaluation half |
|---|---|---|---|---|
| 2008 GFC | 2004-01-01 – 2007-12-31 (923 seq; fit 785, val 138) | 2008-01-08 – 2009-12-31 (501 d) | 2008-01-08 – 2009-01-02 (250 d) | 2009-01-05 – 2009-12-31 (251 d) |
| 2011 EU crisis | 2004-01-01 – 2010-12-31 (1680 seq; fit 1428, val 252) | 2011-01-07 – 2013-12-31 (750 d) | 2011-01-07 – 2012-07-03 (375 d) | 2012-07-05 – 2013-12-31 (375 d) |
| 2015 selloff | 2004-01-01 – 2014-12-31 (2686 seq; fit 2283, val 403) | 2015-01-08 – 2017-12-29 (751 d) | 2015-01-08 – 2016-07-05 (375 d) | 2016-07-06 – 2017-12-29 (376 d) |
| 2020 COVID | 2004-01-01 – 2018-12-31 (3692 seq; fit 3138, val 554) | 2019-01-08 – 2020-12-31 (501 d) | 2019-01-08 – 2020-01-03 (250 d) | 2020-01-06 – 2020-12-31 (251 d) |
| 2022 bear | 2004-01-01 – 2021-12-31 (4449 seq; fit 3782, val 667) | 2022-01-07 – 2022-12-30 (247 d) | 2022-01-07 – 2022-07-06 (123 d) | 2022-07-07 – 2022-12-30 (124 d) |

Source: walk_forward_results_leakage_free_cleanval.json, reviewer_backtests_cleanval_results.json

## tab:walkforward (Table V per the review): frozen offset, evaluation half

| Fold | Uncalibrated | Calibrated (frozen) | Verdict | Offset | Epochs |
|---|---|---|---|---|---|
| 2008 GFC | 93/251 = 37.05%, p = 1.13e-54 | 4/251 = 1.59%, p = 0.004 | FAIL | -0.04314 | 10 |
| 2011 EU crisis | 109/375 = 29.07%, p = 1.40e-51 | 3/375 = 0.80%, p = 4.16e-06 | FAIL | -0.01733 | 24 |
| 2015 selloff | 88/376 = 23.40%, p = 5.75e-34 | 4/376 = 1.06%, p = 2.42e-05 | FAIL | -0.01235 | 18 |
| 2020 COVID | 93/251 = 37.05%, p = 1.13e-54 | 41/251 = 16.33%, p = 3.80e-11 | FAIL | -0.00860 | 19 |
| 2022 bear | 55/124 = 44.35%, p = 4.77e-38 | 4/124 = 3.23%, p = 0.334 | not rejected | -0.02282 | 21 |

Source: walk_forward_results_leakage_free_cleanval.json. The ten-seed column in the paper is from the ORIGINAL feature set (seed_distribution_results.json) and was not re-run; label it as such.

## tab:rolling: rolling recalibration L=125, R=21, same evaluation half

| Fold | Frozen | Rolling | p_cc (rolling) | Rolling verdict |
|---|---|---|---|---|
| 2008 GFC | 4/251 = 1.59%, p = 0.004 | 2/251 = 0.80%, p = 1.63e-04 | 8.06e-04 | FAIL |
| 2011 EU crisis | 3/375 = 0.80%, p = 4.16e-06 | 17/375 = 4.53%, p = 0.674 | 0.885 | not rejected |
| 2015 selloff | 4/376 = 1.06%, p = 2.42e-05 | 14/376 = 3.72%, p = 0.235 | 0.287 | not rejected |
| 2020 COVID | 41/251 = 16.33%, p = 3.80e-11 | 19/251 = 7.57%, p = 0.081 | 0.100 | not rejected |
| 2022 bear | 4/124 = 3.23%, p = 0.334 | 4/122 = 3.28%, p = 0.353 | 0.567 | not rejected |

Source: walk_forward_results_leakage_free_cleanval.json (rolling_calibration.eval_slice_only); p_cc from reviewer_backtests_cleanval_results.json. Rolling starts at day L, so 2022's slice is 122 days, not 124.

## tab:window (Table VII per the review): window sweep, Kupiec

| Fold | L=40 | L=60 | L=90 | L=125 | L=175 |
|---|---|---|---|---|---|
| 2008 GFC | 12/251 = 4.78%, p = 0.873 | 6/251 = 2.39%, p = 0.035 ✗ | 5/251 = 1.99%, p = 0.013 ✗ | 2/251 = 0.80%, p = 1.63e-04 ✗ | 0/251 = 0.00%, p = 3.89e-07 ✗ |
| 2011 EU crisis | 23/375 = 6.13%, p = 0.330 | 19/375 = 5.07%, p = 0.953 | 17/375 = 4.53%, p = 0.674 | 17/375 = 4.53%, p = 0.674 | 15/375 = 4.00%, p = 0.358 |
| 2015 selloff | 26/376 = 6.91%, p = 0.106 | 24/376 = 6.38%, p = 0.237 | 19/376 = 5.05%, p = 0.962 | 14/376 = 3.72%, p = 0.235 | 13/376 = 3.46%, p = 0.147 |
| 2020 COVID | 26/251 = 10.36%, p = 6.09e-04 ✗ | 24/251 = 9.56%, p = 0.003 ✗ | 18/251 = 7.17%, p = 0.137 | 19/251 = 7.57%, p = 0.081 | 20/251 = 7.97%, p = 0.046 ✗ |
| 2022 bear | 6/124 = 4.84%, p = 0.934 | 6/124 = 4.84%, p = 0.934 | 4/124 = 3.23%, p = 0.334 | 4/122 = 3.28%, p = 0.353 | 2/72 = 2.78%, p = 0.347 |

✗ = rejected at 5%. Source: reviewer_backtests_cleanval_results.json (window_sweep). Exploratory sweep; at Bonferroni 0.01, L=90 is rejected in no fold.

## tab:monitor: volatility-ratio monitor (rolling L=125), all valid days of the walk

| Fold | Elevated days: breaches | Normal days: breaches | Breaches preceded by elevated (5 d) | Baseline |
|---|---|---|---|---|
| 2008 GFC | 16/227 = 7.05% | 8/148 = 5.41% | 83.3% (of 24 breaches) | 71.0% |
| 2020 COVID | 19/263 = 7.22% | 6/112 = 5.36% | 96.0% (of 25 breaches) | 80.1% |
| 2011 EU crisis | 10/248 = 4.03% | 28/376 = 7.45% | 36.8% (of 38 breaches) | 54.2% |
| 2015 selloff | 9/271 = 3.32% | 19/354 = 5.37% | 53.6% (of 28 breaches) | 60.5% |
| 2022 bear | 1/21 = 4.76% | 3/100 = 3.00% | 25.0% (of 4 breaches) | 35.6% |

Source: walk_forward_results_leakage_free_cleanval.json (rolling_calibration.monitor). Breach counts are rate × days, rounded; the file stores rates and day counts.

## tab:empirical (Table IX per the review): historical VaR, N=250, vs parametric (frozen)

| Fold | Parametric, calibrated | Historical raw | Historical calibrated |
|---|---|---|---|
| 2008 GFC | 4/251 = 1.59%, p = 0.004 | 2/251 = 0.80%, p = 1.63e-04 | 0/251 = 0.00%, p = 3.89e-07 |
| 2011 EU crisis | 3/375 = 0.80%, p = 4.16e-06 | 12/375 = 3.20%, p = 0.088 | 6/375 = 1.60%, p = 4.58e-04 |
| 2015 selloff | 4/376 = 1.06%, p = 2.42e-05 | 9/376 = 2.39%, p = 0.010 | 8/376 = 2.13%, p = 0.004 |
| 2020 COVID | 41/251 = 16.33%, p = 3.80e-11 | 21/251 = 8.37%, p = 0.025 | 29/251 = 11.55%, p = 4.06e-05 |
| 2022 bear | 4/124 = 3.23%, p = 0.334 | 3/124 = 2.42%, p = 0.144 | 1/124 = 0.81%, p = 0.008 |

Sources: walk_forward_results_leakage_free_cleanval.json; empirical_var_comparison_results.json (no model, unchanged).

## tab:ablation: 5-head MTL vs single-task VaR head, frozen offset

| Case | MTL | Single-task |
|---|---|---|
| Single split | 17/470 = 3.62%, p = 0.149 | 17/470 = 3.62%, p = 0.149 |
| 2008 GFC | 4/251 = 1.59%, p = 0.004 | 4/251 = 1.59%, p = 0.004 |
| 2011 EU crisis | 3/375 = 0.80%, p = 4.16e-06 | 3/375 = 0.80%, p = 4.16e-06 |
| 2015 selloff | 4/376 = 1.06%, p = 2.42e-05 | 3/376 = 0.80%, p = 3.97e-06 |
| 2020 COVID | 41/251 = 16.33%, p = 3.80e-11 | 42/251 = 16.73%, p = 9.82e-12 |
| 2022 bear | 4/124 = 3.23%, p = 0.334 | 5/124 = 4.03%, p = 0.609 |

Sources: walk_forward_results_leakage_free_cleanval.json, cleanval_single_task_results.json, cleanval_single_split_results.json, cleanval_single_split_single_task_results.json.

## Return-quantile control (section 14.4 / 15.4), frozen and raw

| Fold | Raw | Frozen | Rolling |
|---|---|---|---|
| 2008 GFC | 32/251 = 12.75%, p = 1.96e-06 | 2/251 = 0.80%, p = 1.63e-04 | 1/251 = 0.40%, p = 1.62e-05 |
| 2011 EU crisis | 32/375 = 8.53%, p = 0.004 | 9/375 = 2.40%, p = 0.010 | 20/375 = 5.33%, p = 0.769 |
| 2015 selloff | 4/376 = 1.06%, p = 2.42e-05 | 8/376 = 2.13%, p = 0.004 | 18/376 = 4.79%, p = 0.849 |
| 2020 COVID | 27/251 = 10.76%, p = 2.56e-04 | 31/251 = 12.35%, p = 5.57e-06 | 19/251 = 7.57%, p = 0.081 |
| 2022 bear | 18/124 = 14.52%, p = 6.36e-05 | 11/124 = 8.87%, p = 0.073 | 8/122 = 6.56%, p = 0.450 |

Source: cleanval_single_task_results.json

## tab:accuracy: per-task MAE (clean validation)

| Case (n test) | log_return | log_volume | yz_vol | VaR_5 | CVaR_5 | Predict zero | Persistence |
|---|---|---|---|---|---|---|---|
| 2008 GFC (501) | 0.01518 | 0.6846 | 0.0002545 | 0.004268 | 0.01985 | 0.01492 | 0.02301 |
| 2011 EU crisis (750) | 0.007255 | 0.1308 | 2.965e-05 | 0.004185 | 0.00392 | 0.00725 | 0.01071 |
| 2015 selloff (751) | 0.005392 | 0.1125 | 2.064e-05 | 0.003162 | 0.002915 | 0.005297 | 0.007902 |
| 2020 COVID (501) | 0.009581 | 0.1616 | 7.286e-05 | 0.004071 | 0.007316 | 0.00956 | 0.01466 |
| 2022 bear (247) | 0.01248 | 0.1297 | 2.341e-05 | 0.004433 | 0.003182 | 0.01204 | 0.01719 |
| Single split (940) | 0.009696 | 0.1386 | 6.037e-05 | 0.004055 | 0.006011 | 0.009621 | 0.0143 |

Single-split volatility: GARCH(1,1) 5.868e-05 (classical_baselines_results.json, 941-day window) vs transformer 6.037e-05. Sources: walk_forward_results_leakage_free_cleanval.json, cleanval_single_split_results.json, leakage_free_summary.json.

## Single split, frozen offset (clean validation)

Uncalibrated 174/470 = 37.02%, p = 1.34e-100; calibrated 17/470 = 3.62%, p = 0.149, p_cc = 0.316; epochs 14; test 2019-04-10 – 2022-12-30. Source: cleanval_single_split_results.json

## tab:replay (Table XI per the review): registry v4 (@cleanval), clean validation

| Phase | Dates | N/T | Kupiec p |
|---|---|---|---|
| calm, pre-shock | 2019-06-04 – 2020-02-19 | 3/180 = 1.67% | 0.018 |
| COVID shock | 2020-02-20 – 2020-04-30 | 12/50 = 24.00% | 5.41e-06 |
| recovery | 2020-05-01 – 2020-12-31 | 5/170 = 2.94% | 0.184 |
| whole window | 2019-06-04 – 2020-12-31 | 20/400 = 5.00% | 1.000 |

Source: drift_demo_cleanval_results.json

## Unchanged tables (no neural training involved)

- tab:leakage (R²): leakage_audit_results.json, unchanged.
