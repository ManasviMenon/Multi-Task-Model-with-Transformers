"""
ACAI 2026 review, revisions 3, 4 and 6: full-fold backtest diagnostics, classical VaR baselines
under the identical five-fold protocol, and the per-fold reproducibility facts.

Inputs (all read-only):
  - results/predictions_leakage_free/*.npz : the leakage-free transformer's test-period dates,
    realised returns, VaR predictions and calib_n - the primary specification
  - SPX500-18Y_scaled_y.csv                : raw prices, for the classical baselines
  - mtl_var.pipeline.FOLDS                 : fold calendar (training periods for GARCH fits)

Protocol, identical for every method:
  - same evaluation dates (the transformer's cached dates, so the calendar matches by construction)
  - "raw": the method's own VaR, no adjustment
  - "frozen": calib_n = T//2; 5th percentile of (return - VaR) on the first half is added to the
    second half's VaR; second half evaluated (calibration.fixed_offset_calibration)
  - "rolling": calibration.rolling_calibration_analysis, L=125, R=21, reported on the same
    second-half slice (the paper's rolling table)
  - backtests on integer counts: Kupiec POF, Christoffersen independence and conditional
    coverage, Wilson 95% interval on N/T, exceedance severity (mtl_var.backtest.full_backtest)

Classical baselines (revision 4), each a genuine one-step-ahead 5% return quantile:
  - HS-250   : 5th percentile of the 250 log returns strictly before day t
  - EWMA     : RiskMetrics, lambda = 0.94, sigma2_t = lam*sigma2_{t-1} + (1-lam)*r_{t-1}^2,
               VaR_t = z_0.05 * sigma_t (zero mean), recursion run from the first data row
  - GARCH    : GARCH(1,1), constant mean, Gaussian, parameters fitted ONCE per fold on that fold's
               training period only (same expanding window the transformer is trained on), then
               held fixed and filtered through the test period; VaR_t = mu + z_0.05 * sigma_t
None of these is tuned to the folds. No transformer is retrained.

------------------------------------------------------------------------------------------
PRE-REGISTERED READINGS (written before any number from this script was computed):

Revision 3 (clustering), transformer, frozen offset:
  R3a. Folds failing Kupiec will also fail conditional coverage (LR_cc includes LR_POF, so
       this is close to automatic - not informative on its own).
  R3b. The informative questions: (i) does 2022 (Kupiec non-rejection, p=0.334) also avoid
       rejection on LR_ind and LR_cc? If it rejects either, the paper says 2022 fails too. If
       not, the paper says "not rejected", with the Wilson interval, never "correctly calibrated".
       (ii) Do the folds that rolling recalibration "rescues" on Kupiec survive LR_ind and LR_cc?
       Any that reject are withdrawn from the "rolling recalibration fixes 4/5" claim.

Revision 4 (baselines), frozen offset:
  R4a. If the classical baselines, under the frozen offset, are rejected (Kupiec or LR_cc) in at
       least 3 of the 4 folds where the transformer is rejected, the failure is attributed to the
       frozen calibration under regime shift and not to the transformer. Supports the paper.
  R4b. If any classical baseline under the frozen offset is not rejected (Kupiec AND LR_cc) in
       4 or 5 folds, the paper's claim narrows to "this estimator under this protocol".
  R4c. Causal isolation (the reviewer's point): if a baseline's RAW VaR is not rejected in a fold
       where its FROZEN-calibrated VaR is rejected, the calibration step itself did the damage
       there. If raw and frozen are both rejected, the controls do not isolate calibration as the
       cause, and the paper must say so.

Multiplicity (optional item): the window sweep (5 lengths x 5 folds) is re-read with a
Bonferroni threshold of 0.05/5 = 0.01 per fold. If any window then has no rejection on any fold,
the claim "no static window passes every fold" holds only at the unadjusted 5% level and must be
stated that way.
------------------------------------------------------------------------------------------
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import norm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mtl_var.backtest import full_backtest  # noqa: E402
from mtl_var.calibration import fixed_offset_calibration, rolling_calibration_analysis  # noqa: E402
from mtl_var.pipeline import FOLDS  # noqa: E402

CSV = "SPX500-18Y_scaled_y.csv"
# Defaults reproduce section 14 (models trained with test-window early stopping). Section 15's
# clean-validation run: --pred-dir results/predictions_leakage_free_cleanval
#                       --out results/reviewer_backtests_cleanval_results.json
PRED_DIR = "results/predictions_leakage_free"
OUT = "results/reviewer_backtests_results.json"
if "--pred-dir" in sys.argv:
    PRED_DIR = sys.argv[sys.argv.index("--pred-dir") + 1]
if "--out" in sys.argv:
    OUT = sys.argv[sys.argv.index("--out") + 1]
Z05 = norm.ppf(0.05)
WINDOWS = [40, 60, 90, 125, 175]
FOLD_LABEL = {"fold1_gfc": "2008 GFC", "fold2_eu_crisis": "2011 EU crisis",
              "fold3_2015_selloff": "2015 selloff", "fold4_covid": "2020 COVID",
              "fold5_2022_bear": "2022 bear"}


def load_raw():
    data = pd.read_csv(CSV, parse_dates=['observation_date'], dayfirst=True)
    data = data.dropna(axis=1, how='all').drop(index=0)
    data['log_return'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1))
    data = data.dropna(subset=['log_return']).sort_values('observation_date').reset_index(drop=True)
    return data['observation_date'].values.astype('datetime64[D]'), data['log_return'].values


def hs_var(raw_r, idx, n=250):
    return np.array([np.quantile(raw_r[i - n:i], 0.05) for i in idx])


def ewma_var(raw_r, idx, lam=0.94):
    s2 = np.empty(len(raw_r))
    s2[0] = raw_r[:30].var()
    for t in range(1, len(raw_r)):
        s2[t] = lam * s2[t - 1] + (1 - lam) * raw_r[t - 1] ** 2  # uses only r_{t-1}
    return Z05 * np.sqrt(s2[idx])


def garch_var(raw_d, raw_r, idx, fold):
    train = (raw_d >= np.datetime64(fold["train_start"])) & (raw_d <= np.datetime64(fold["train_end"]))
    scale = 100.0
    fit = arch_model(raw_r[train] * scale, mean="Constant", vol="GARCH", p=1, q=1,
                     dist="normal").fit(disp="off")
    upto = idx.max() + 1
    first = int(np.argmax(train))
    fixed = arch_model(raw_r[first:upto] * scale, mean="Constant", vol="GARCH", p=1, q=1,
                       dist="normal").fix(fit.params)
    sigma = np.asarray(fixed.conditional_volatility)  # sigma_t uses information through t-1
    mu = float(fit.params["mu"])
    var = (mu + Z05 * sigma) / scale
    return var[idx - first], {k: float(v) for k, v in fit.params.items()}


def frozen_eval(returns, var, calib_n):
    fo = fixed_offset_calibration(returns, var)
    assert fo["calib_n"] == calib_n
    off = fo["offset"]
    r_e, v_e = returns[calib_n:], var[calib_n:]
    return {"offset": off,
            "raw": full_backtest(r_e, v_e),
            "frozen": full_backtest(r_e, v_e + off)}


def rolling_eval(dates, returns, var, calib_n, L=125, R=21):
    rc = rolling_calibration_analysis(dates, returns, var, eval_start_idx=calib_n, L=L, R=R)
    if rc is None:
        return None
    idx = np.arange(L, len(returns))
    off = np.array(rc["timeline"]["offset"], dtype=float)
    # recompute from unrounded offsets rather than the rounded timeline copy
    offs = np.empty(len(idx))
    current = None
    refresh = set(rc["refresh_days"])
    for k, t in enumerate(idx):
        if t in refresh:
            current = float(np.quantile(returns[t - L:t] - var[t - L:t], 0.05))
        offs[k] = current
    assert np.allclose(offs, off, atol=1e-6)
    m = idx >= calib_n
    bt = full_backtest(returns[idx[m]], var[idx[m]] + offs[m])
    assert bt["N"] == rc["eval_slice_only"]["breaches"] and bt["T"] == rc["eval_slice_only"]["n"]
    return bt


def fmt_p(p):
    return f"{p:.3g}"


def main():
    print(__doc__)
    raw_d, raw_r = load_raw()
    date_idx = {d: i for i, d in enumerate(raw_d)}
    out = {"protocol": {"csv": CSV, "predictions": PRED_DIR, "rolling": {"L": 125, "R": 21},
                        "baselines": {"HS": 250, "EWMA_lambda": 0.94, "GARCH": "(1,1) normal, fit on fold train"},
                        "first_raw_date": str(raw_d[0]), "last_raw_date": str(raw_d[-1])},
           "folds": {}}

    for fold in FOLDS:
        name = fold["name"]
        d = np.load(os.path.join(PRED_DIR, f"{name}.npz"), allow_pickle=True)
        dates = d["dates"].astype("datetime64[D]")
        returns = d["returns"].astype(float)
        var_tf = d["var_pred"].astype(float)
        calib_n = int(d["calib_n"])
        idx = np.array([date_idx[x] for x in dates])
        # the transformer's realised returns are the raw log returns on those dates, exactly
        assert np.allclose(raw_r[idx], returns, rtol=0, atol=1e-12), name

        g_var, g_params = garch_var(raw_d, raw_r, idx, fold)
        methods = {"Transformer (leakage-free)": var_tf,
                   "HS-250": hs_var(raw_r, idx),
                   "EWMA (0.94)": ewma_var(raw_r, idx),
                   "GARCH(1,1)": g_var}
        fres = {"label": FOLD_LABEL[name],
                "test_window": [str(dates[0]), str(dates[-1])],
                "calibration_half": [str(dates[0]), str(dates[calib_n - 1]), calib_n],
                "evaluation_half": [str(dates[calib_n]), str(dates[-1]), len(dates) - calib_n],
                "garch_params": g_params, "methods": {}}
        for mname, v in methods.items():
            r = frozen_eval(returns, v, calib_n)
            r["rolling"] = rolling_eval(dates, returns, v, calib_n)
            fres["methods"][mname] = r

        # window sweep for the transformer, with Christoffersen on every cell
        sweep = {}
        for L in WINDOWS:
            sweep[L] = rolling_eval(dates, returns, var_tf, calib_n, L=L)
        fres["window_sweep"] = sweep
        out["folds"][name] = fres

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"written {OUT}\n")
    report(out)


def verdict(bt):
    if bt is None:
        return "-"
    rej = []
    if bt["p_pof"] < 0.05:
        rej.append("POF")
    if bt["p_ind"] < 0.05:
        rej.append("IND")
    if bt["p_cc"] < 0.05:
        rej.append("CC")
    return "reject " + "+".join(rej) if rej else "no reject"


def row(label, bt):
    lo, hi = bt["wilson95"]
    t = bt["transitions"]
    ind = f"{bt['lr_ind']:.2f} / {fmt_p(bt['p_ind'])}" + ("" if bt["ind_testable"] else " (untestable)")
    return (f"| {label} | {bt['N']}/{bt['T']} | {bt['coverage']*100:.2f}% [{lo*100:.1f}, {hi*100:.1f}] | "
            f"{bt['lr_pof']:.2f} / {fmt_p(bt['p_pof'])} | {t['n01']},{t['n11']} | {ind} | "
            f"{bt['lr_cc']:.2f} / {fmt_p(bt['p_cc'])} | {verdict(bt)} |")


HEADER = ("| {first} | N/T | Coverage [Wilson 95%] | LR_POF / p | n01,n11 | LR_ind / p | LR_cc / p | Verdict (5%) |\n"
          "|---|---|---|---|---|---|---|---|")


def report(out):
    folds = out["folds"]
    tf = "Transformer (leakage-free)"
    print("## Revision 6: per-fold calendar\n")
    print("| Fold | Test window (cached) | Calibration half | Evaluation half | Frozen breaches |")
    print("|---|---|---|---|---|")
    for f in folds.values():
        c, e = f["calibration_half"], f["evaluation_half"]
        bt = f["methods"][tf]["frozen"]
        print(f"| {f['label']} | {f['test_window'][0]} to {f['test_window'][1]} | {c[0]} to {c[1]} ({c[2]}d) | "
              f"{e[0]} to {e[1]} ({e[2]}d) | {bt['N']}/{bt['T']} |")

    for block, title in [("frozen", "frozen offset"), ("rolling", "rolling recalibration L=125, R=21"),
                         ("raw", "raw, no calibration")]:
        print(f"\n## Revision 3: transformer, {title}\n")
        print(HEADER.format(first="Fold"))
        for f in folds.values():
            print(row(f["label"], f["methods"][tf][block]))
        print("\nExceedance severity (mean / worst return minus VaR on breach days):")
        for f in folds.values():
            bt = f["methods"][tf][block]
            if bt["N"]:
                print(f"  {f['label']}: {bt['mean_exceedance']*100:.2f}% / {bt['max_exceedance']*100:.2f}%")
            else:
                print(f"  {f['label']}: no breaches")

    for block, title in [("raw", "raw (method's own VaR)"), ("frozen", "frozen offset"),
                         ("rolling", "rolling recalibration L=125, R=21")]:
        print(f"\n## Revision 4: all methods, {title}\n")
        print(HEADER.format(first="Fold / method"))
        for f in folds.values():
            for m, r in f["methods"].items():
                print(row(f"{f['label']} / {m}", r[block]))

    print("\n## Window sweep (transformer), Christoffersen added; Bonferroni threshold 0.01\n")
    print("| Fold | " + " | ".join(f"L={L}" for L in WINDOWS) + " |")
    print("|---|" + "---|" * len(WINDOWS))
    for f in folds.values():
        cells = []
        for L in WINDOWS:
            bt = f["window_sweep"][str(L)] if str(L) in f["window_sweep"] else f["window_sweep"][L]
            cells.append(f"{bt['N']}/{bt['T']} pof {fmt_p(bt['p_pof'])} cc {fmt_p(bt['p_cc'])}")
        print(f"| {f['label']} | " + " | ".join(cells) + " |")

    print("\n## Pre-registered checks\n")
    names = list(folds)
    tf_rej = {n: folds[n]["methods"][tf]["frozen"]["p_pof"] < 0.05 or folds[n]["methods"][tf]["frozen"]["p_cc"] < 0.05
              for n in names}
    print(f"R3a: transformer frozen rejected (POF or CC): {[folds[n]['label'] for n in names if tf_rej[n]]}")
    b22 = folds["fold5_2022_bear"]["methods"][tf]["frozen"]
    print(f"R3b(i): 2022 frozen p_pof={fmt_p(b22['p_pof'])} p_ind={fmt_p(b22['p_ind'])} p_cc={fmt_p(b22['p_cc'])} "
          f"Wilson={[round(x*100, 2) for x in b22['wilson95']]}%")
    for n in names:
        rb = folds[n]["methods"][tf]["rolling"]
        print(f"R3b(ii): rolling {folds[n]['label']}: {verdict(rb)}")
    failing = [n for n in names if tf_rej[n]]
    for m in ["HS-250", "EWMA (0.94)", "GARCH(1,1)"]:
        rej = {n: (folds[n]["methods"][m]["frozen"]["p_pof"] < 0.05 or folds[n]["methods"][m]["frozen"]["p_cc"] < 0.05)
               for n in names}
        raw_rej = {n: (folds[n]["methods"][m]["raw"]["p_pof"] < 0.05 or folds[n]["methods"][m]["raw"]["p_cc"] < 0.05)
                   for n in names}
        both = sum(rej[n] for n in failing)
        print(f"R4a {m}: frozen rejected in {both}/{len(failing)} transformer-failing folds -> "
              f"{'met' if both >= 3 else 'NOT met'}")
        print(f"R4b {m}: frozen not rejected in {sum(not rej[n] for n in names)}/5 folds -> "
              f"{'claim must narrow' if sum(not rej[n] for n in names) >= 4 else 'no narrowing needed'}")
        hurt = [folds[n]["label"] for n in names if rej[n] and not raw_rej[n]]
        helped = [folds[n]["label"] for n in names if raw_rej[n] and not rej[n]]
        bothrej = [folds[n]["label"] for n in names if raw_rej[n] and rej[n]]
        print(f"R4c {m}: calibration turned pass->reject in {hurt}; reject->pass in {helped}; "
              f"rejected both raw and frozen in {bothrej}")
    for L in WINDOWS:
        rej_bonf = [folds[n]["label"] for n in names
                    if (folds[n]["window_sweep"].get(str(L)) or folds[n]["window_sweep"][L])["p_pof"] < 0.01]
        rej_cc = [folds[n]["label"] for n in names
                  if (folds[n]["window_sweep"].get(str(L)) or folds[n]["window_sweep"][L])["p_cc"] < 0.05]
        print(f"Multiplicity L={L}: Kupiec rejections at 0.01 = {rej_bonf}; LR_cc rejections at 0.05 = {rej_cc}")


if __name__ == "__main__":
    main()
