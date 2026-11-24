"""Print every paper-table value from the results files, with N/T, as markdown.

Run from the repository root:  python research_graphs/paper_table_values.py > research_graphs/paper_table_values.md
Tables are keyed by the LaTeX label the figure script uses; Roman numerals are given only where
the ACAI review confirms them.
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
RES = Path(__file__).resolve().parent.parent / "results"
F = ["fold1_gfc", "fold2_eu_crisis", "fold3_2015_selloff", "fold4_covid", "fold5_2022_bear"]
LBL = dict(zip(F, ["2008 GFC", "2011 EU crisis", "2015 selloff", "2020 COVID", "2022 bear"]))


def load(name):
    with open(RES / name) as fh:
        return json.load(fh)


def p(v):
    return "n/a" if v is None else (f"{v:.2e}" if v < 1e-3 else f"{v:.3f}")


def cell(n, t, pv=None):
    s = f"{n}/{t} = {100 * n / t:.2f}%"
    return s + (f", p = {p(pv)}" if pv is not None else "")


def verdict(pv):
    return "FAIL" if pv < 0.05 else "not rejected"


wf = {r["fold"]: r for r in load("walk_forward_results_leakage_free_cleanval.json")}
bt = load("reviewer_backtests_cleanval_results.json")["folds"]
st = load("cleanval_single_task_results.json")
ss = load("cleanval_single_split_results.json")
emp = load("empirical_var_comparison_results.json")
summ = load("leakage_free_summary.json")
cls = load("classical_baselines_results.json")
sst = load("cleanval_single_split_single_task_results.json")["frozen_backtest"]
TF = "Transformer (leakage-free)"

print("# Paper table values (clean-validation models)\n")
print("Sources: `results/` files named per table. Coverage = N/T computed from integer counts.\n")

print("## tab:data: fold calendar (evaluation half = second half of each cached test window)\n")
print("| Fold | Train | Test (cached) | Calibration half | Evaluation half |")
print("|---|---|---|---|---|")
for f in F:
    b = bt[f]
    c, e = b["calibration_half"], b["evaluation_half"]
    print(f"| {LBL[f]} | {wf[f]['train_start']} – {wf[f]['train_end']} ({wf[f]['n_train']} seq; fit {wf[f]['n_fit']}, val {wf[f]['n_val']}) | "
          f"{b['test_window'][0]} – {b['test_window'][1]} ({wf[f]['n_test']} d) | {c[0]} – {c[1]} ({c[2]} d) | {e[0]} – {e[1]} ({e[2]} d) |")
print("\nSource: walk_forward_results_leakage_free_cleanval.json, reviewer_backtests_cleanval_results.json\n")

print("## tab:walkforward (Table V per the review): frozen offset, evaluation half\n")
print("| Fold | Uncalibrated | Calibrated (frozen) | Verdict | Offset | Epochs |")
print("|---|---|---|---|---|---|")
for f in F:
    vc = wf[f]["var_calibration"]
    u, c = vc["uncalibrated"], vc["calibrated"]
    print(f"| {LBL[f]} | {cell(u['breaches'], vc['eval_n'], u['kupiec_p'])} | {cell(c['breaches'], vc['eval_n'], c['kupiec_p'])} | "
          f"{verdict(c['kupiec_p'])} | {vc['offset']:.5f} | {wf[f]['epochs_run']} |")
print("\nSource: walk_forward_results_leakage_free_cleanval.json. The ten-seed column in the paper is from the "
      "ORIGINAL feature set (seed_distribution_results.json) and was not re-run; label it as such.\n")

print("## tab:rolling: rolling recalibration L=125, R=21, same evaluation half\n")
print("| Fold | Frozen | Rolling | p_cc (rolling) | Rolling verdict |")
print("|---|---|---|---|---|")
for f in F:
    vc = wf[f]["var_calibration"]
    es = wf[f]["rolling_calibration"]["eval_slice_only"]
    rb = bt[f]["methods"][TF]["rolling"]
    print(f"| {LBL[f]} | {cell(vc['calibrated']['breaches'], vc['eval_n'], vc['calibrated']['kupiec_p'])} | "
          f"{cell(es['breaches'], es['n'], es['kupiec_p'])} | {p(rb['p_cc'])} | {verdict(min(es['kupiec_p'], rb['p_cc']))} |")
print("\nSource: walk_forward_results_leakage_free_cleanval.json (rolling_calibration.eval_slice_only); p_cc from "
      "reviewer_backtests_cleanval_results.json. Rolling starts at day L, so 2022's slice is 122 days, not 124.\n")

print("## tab:window (Table VII per the review): window sweep, Kupiec\n")
Ls = ["40", "60", "90", "125", "175"]
print("| Fold | " + " | ".join(f"L={L}" for L in Ls) + " |")
print("|---|" + "---|" * len(Ls))
for f in F:
    cells = []
    for L in Ls:
        c = bt[f]["window_sweep"][L]
        cells.append(cell(c["N"], c["T"], c["p_pof"]) + (" ✗" if c["p_pof"] < 0.05 else ""))
    print(f"| {LBL[f]} | " + " | ".join(cells) + " |")
print("\n✗ = rejected at 5%. Source: reviewer_backtests_cleanval_results.json (window_sweep). Exploratory sweep; "
      "at Bonferroni 0.01, L=90 is rejected in no fold.\n")

print("## tab:monitor: volatility-ratio monitor (rolling L=125), all valid days of the walk\n")
print("| Fold | Elevated days: breaches | Normal days: breaches | Breaches preceded by elevated (5 d) | Baseline |")
print("|---|---|---|---|---|")
for f in ["fold1_gfc", "fold4_covid", "fold2_eu_crisis", "fold3_2015_selloff", "fold5_2022_bear"]:
    m = wf[f]["rolling_calibration"]["monitor"]
    ne, nv = m["n_elevated_days"], m["n_valid_days"]
    nn = nv - ne
    be, bn = round(m["elevated_breach_rate"] * ne), round(m["normal_breach_rate"] * nn)
    fw = wf[f]["rolling_calibration"]["full_walk"]
    print(f"| {LBL[f]} | {cell(be, ne)} | {cell(bn, nn)} | {100 * m['breach_preceded_by_elevated_rate']:.1f}% "
          f"(of {fw['breaches']} breaches) | {100 * m['baseline_preceded_by_elevated_rate']:.1f}% |")
print("\nSource: walk_forward_results_leakage_free_cleanval.json (rolling_calibration.monitor). Breach counts are "
      "rate × days, rounded; the file stores rates and day counts.\n")

print("## tab:empirical (Table IX per the review): historical VaR, N=250, vs parametric (frozen)\n")
print("| Fold | Parametric, calibrated | Historical raw | Historical calibrated |")
print("|---|---|---|---|")
for f in F:
    vc = wf[f]["var_calibration"]
    e = emp[f]["empirical"]["250"]
    print(f"| {LBL[f]} | {cell(vc['calibrated']['breaches'], vc['eval_n'], vc['calibrated']['kupiec_p'])} | "
          f"{cell(e['raw']['breaches'], e['raw']['n'], e['raw']['kupiec_p'])} | "
          f"{cell(e['calibrated']['breaches'], e['calibrated']['n'], e['calibrated']['kupiec_p'])} |")
print("\nSources: walk_forward_results_leakage_free_cleanval.json; empirical_var_comparison_results.json (no model, unchanged).\n")

print("## tab:ablation: 5-head MTL vs single-task VaR head, frozen offset\n")
print("| Case | MTL | Single-task |")
print("|---|---|---|")
sb = ss["frozen_backtest"]
print(f"| Single split | {cell(sb['N'], sb['T'], sb['p_pof'])} | {cell(sst['N'], sst['T'], sst['p_pof'])} |")
for f in F:
    vc = wf[f]["var_calibration"]
    s = st[f]["var_proxy"]["frozen"]
    print(f"| {LBL[f]} | {cell(vc['calibrated']['breaches'], vc['eval_n'], vc['calibrated']['kupiec_p'])} | {cell(s['N'], s['T'], s['p_pof'])} |")
print("\nSources: walk_forward_results_leakage_free_cleanval.json, cleanval_single_task_results.json, "
      "cleanval_single_split_results.json, cleanval_single_split_single_task_results.json.\n")

print("## Return-quantile control (section 14.4 / 15.4), frozen and raw\n")
print("| Fold | Raw | Frozen | Rolling |")
print("|---|---|---|---|")
for f in F:
    r = st[f]["return_quantile"]
    print(f"| {LBL[f]} | {cell(r['raw']['N'], r['raw']['T'], r['raw']['p_pof'])} | {cell(r['frozen']['N'], r['frozen']['T'], r['frozen']['p_pof'])} | "
          f"{cell(r['rolling']['N'], r['rolling']['T'], r['rolling']['p_pof'])} |")
print("\nSource: cleanval_single_task_results.json\n")

print("## tab:accuracy: per-task MAE (clean validation)\n")
T5 = ["log_return", "log_volume", "yz_vol", "VaR_5", "CVaR_5"]
print("| Case (n test) | " + " | ".join(T5) + " | Predict zero | Persistence |")
print("|---|" + "---|" * (len(T5) + 2))
for f in F:
    a = wf[f]["accuracy"]
    nb = summ["folds"][f]["naive_baselines"]
    print(f"| {LBL[f]} ({wf[f]['n_test']}) | " + " | ".join(f"{a[t]['mae']:.4g}" for t in T5)
          + f" | {nb['predict_zero']:.4g} | {nb['persistence']:.4g} |")
a = ss["accuracy"]
print(f"| Single split ({ss['n_test']}) | " + " | ".join(f"{a[t]['mae']:.4g}" for t in T5)
      + f" | {ss['naive_baselines']['predict_zero']:.4g} | {ss['naive_baselines']['persistence']:.4g} |")
print(f"\nSingle-split volatility: GARCH(1,1) {cls['garch_1_1']['vol_mae']:.4g} (classical_baselines_results.json, "
      f"{cls['test_period']['n']}-day window) vs transformer {a['yz_vol']['mae']:.4g}. "
      "Sources: walk_forward_results_leakage_free_cleanval.json, cleanval_single_split_results.json, leakage_free_summary.json.\n")

print("## Single split, frozen offset (clean validation)\n")
vc = ss["var_calibration"]
print(f"Uncalibrated {cell(vc['uncalibrated']['breaches'], vc['eval_n'], vc['uncalibrated']['kupiec_p'])}; "
      f"calibrated {cell(sb['N'], sb['T'], sb['p_pof'])}, p_cc = {p(sb['p_cc'])}; epochs {ss['epochs_run']}; "
      f"test {ss['test_window'][0]} – {ss['test_window'][1]}. Source: cleanval_single_split_results.json\n")

rp = load("drift_demo_cleanval_results.json")["replays"]["2019_2020_covid_shock"]
print("## tab:replay (Table XI per the review): registry v4 (@cleanval), clean validation\n")
print("| Phase | Dates | N/T | Kupiec p |")
print("|---|---|---|---|")
for ph in rp["phases"]:
    print(f"| {ph['phase']} | {ph['first']} – {ph['last']} | {cell(ph['breaches'], ph['days'])} | {p(ph['kupiec_p'])} |")
ww = rp["whole_window"]
print(f"| whole window | {rp['window']['first_forecast']} – {rp['window']['last_forecast']} | "
      f"{cell(ww['breaches'], ww['days'])} | {p(ww['kupiec_p'])} |")
print("\nSource: drift_demo_cleanval_results.json\n")

print("## Unchanged tables (no neural training involved)\n")
print("- tab:leakage (R²): leakage_audit_results.json, unchanged.")
