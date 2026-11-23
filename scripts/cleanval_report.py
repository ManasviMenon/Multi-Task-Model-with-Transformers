"""
Before/after report for the early-stopping fix (RESULTS.md section 15).

Old: every leakage-free neural model used the fold's TEST window for early stopping and
best-weight restoration (copies in results/leaky_earlystop/). New: early stopping on the last 15%
of each fold's training sequences, chronological (mtl_var.model.fit_with_train_tail_validation).
Nothing else changes: architecture, losses, 0.2 weights, Adam 1e-4, batch 32, patience 5,
seed 42, 5-day sequences, leakage-free features.

Inputs:
  old MTL  results/leaky_earlystop/walk_forward_results_leakage_free.json
  new MTL  results/walk_forward_results_leakage_free_cleanval.json   (mtl-var walk-forward)
  old/new  results/leaky_earlystop/reviewer_backtests_results.json vs
           results/reviewer_backtests_cleanval_results.json          (rolling, sweep, Christoffersen,
                                                                       baselines)
  old/new  single-task VaR-proxy and return-quantile heads:
           results/leaky_earlystop/{leakage_free_ablation,return_quantile_control}_results.json vs
           results/cleanval_single_task_results.json

------------------------------------------------------------------------------------------
PRE-REGISTERED (written 2026-11-23, before any clean-validation model was trained):

  P1 (step 4). A fold's verdict "changed" if its frozen-offset Kupiec decision at 5% flips.
      Expectation, not a criterion: no flips, because the model-free raw-target backtest already
      reproduces every fold verdict (section 13). Epochs, validation losses and MAEs will change
      and are reported without a pass/fail.
  P2 (step 5). "Rolling rescues all but 2008" still holds iff, under rolling L=125/R=21, GFC is
      rejected (Kupiec or LR_cc at 5%) and the other four are not. "No single window passes every
      fold at 5%" still holds iff every L in {40,60,90,125,175} has at least one fold rejected by
      Kupiec at 5%.
  P3 (step 6). Reported per fold with N/T. The 2022 statement ("not rejected, interval includes
      5%") still holds iff 2022 is rejected by none of POF, IND, CC.
  P4 (step 7). The return-quantile conclusion (C2) still holds iff the return-quantile head under
      the frozen offset is rejected in at least 3 of the folds where the clean MTL model is
      rejected. The single-task-vs-MTL comparison is reported fold by fold, with no criterion
      (section 13 already withdrew the ablation's conclusion).
  P5 (step 8). The classical baselines and the model-free backtest do not involve training. Check:
      every baseline entry in the regenerated reviewer-backtest JSON is identical to the old one.
------------------------------------------------------------------------------------------
"""
import json
import math
import os

OLD_MTL = "results/leaky_earlystop/walk_forward_results_leakage_free.json"
NEW_MTL = "results/walk_forward_results_leakage_free_cleanval.json"
OLD_RB = "results/leaky_earlystop/reviewer_backtests_results.json"
NEW_RB = "results/reviewer_backtests_cleanval_results.json"
OLD_ABL = "results/leaky_earlystop/leakage_free_ablation_results.json"
OLD_RQ = "results/leaky_earlystop/return_quantile_control_results.json"
NEW_ST = "results/cleanval_single_task_results.json"
TASKS = ["log_return", "log_volume", "yz_vol", "VaR_5", "CVaR_5"]
TF = "Transformer (leakage-free)"
LABEL = {"fold1_gfc": "2008 GFC", "fold2_eu_crisis": "2011 EU crisis", "fold3_2015_selloff": "2015 selloff",
         "fold4_covid": "2020 COVID", "fold5_2022_bear": "2022 bear"}


def load(p):
    with open(p) as f:
        return json.load(f)


def pv(p):
    return "nan" if p is None or (isinstance(p, float) and math.isnan(p)) else f"{p:.3g}"


def rejects(bt):
    return bt["p_pof"] < 0.05 or bt["p_cc"] < 0.05


def step4():
    old = {r["fold"]: r for r in load(OLD_MTL)}
    new = {r["fold"]: r for r in load(NEW_MTL)}
    print("## Step 4: MTL retrain, frozen offset\n")
    print("| Fold | Epochs old→new | Best val loss old (test win) → new (train tail) | n_fit / n_val |")
    print("|---|---|---|---|")
    for f in LABEL:
        o, n = old[f], new[f]
        print(f"| {LABEL[f]} | {o['epochs_run']} → {n['epochs_run']} | {o['best_val_loss']:.4f} → "
              f"{n['best_val_loss']:.4f} | {n['n_fit']} / {n['n_val']} |")
    print("\n| Fold | " + " | ".join(f"{t} MAE old → new" for t in TASKS) + " |")
    print("|---|" + "---|" * len(TASKS))
    for f in LABEL:
        cells = [f"{old[f]['accuracy'][t]['mae']:.4g} → {new[f]['accuracy'][t]['mae']:.4g} "
                 f"({new[f]['accuracy'][t]['mae'] / old[f]['accuracy'][t]['mae']:.2f}×)" for t in TASKS]
        print(f"| {LABEL[f]} | " + " | ".join(cells) + " |")
    print("\n| Fold | Old N/T, cov, p | New N/T, cov, p | Old verdict | New verdict | Changed? |")
    print("|---|---|---|---|---|---|")
    flips = []
    for f in LABEL:
        oc, nc = old[f]["var_calibration"], new[f]["var_calibration"]
        o, n = oc["calibrated"], nc["calibrated"]
        T_o, T_n = oc["eval_n"], nc["eval_n"]
        assert abs(o["coverage"] - o["breaches"] / T_o) < 1e-15 and abs(n["coverage"] - n["breaches"] / T_n) < 1e-15
        vo, vn = ("FAIL" if o["kupiec_p"] < 0.05 else "pass"), ("FAIL" if n["kupiec_p"] < 0.05 else "pass")
        if vo != vn:
            flips.append(LABEL[f])
        print(f"| {LABEL[f]} | {o['breaches']}/{T_o}, {o['breaches']/T_o*100:.2f}%, {pv(o['kupiec_p'])} | "
              f"{n['breaches']}/{T_n}, {n['breaches']/T_n*100:.2f}%, {pv(n['kupiec_p'])} | {vo} | {vn} | "
              f"{'**yes**' if vo != vn else 'no'} |")
    print(f"\nP1: verdict flips: {flips or 'none'}")
    print("\nUncalibrated (raw) coverage old → new:")
    for f in LABEL:
        o, n = old[f]["var_calibration"]["uncalibrated"], new[f]["var_calibration"]["uncalibrated"]
        print(f"  {LABEL[f]}: {o['breaches']}/{old[f]['var_calibration']['eval_n']} → "
              f"{n['breaches']}/{new[f]['var_calibration']['eval_n']}")


def bt_cell(bt):
    return f"{bt['N']}/{bt['T']} ({bt['coverage']*100:.2f}%) pof {pv(bt['p_pof'])} cc {pv(bt['p_cc'])}"


def steps5_6_8():
    old, new = load(OLD_RB), load(NEW_RB)
    print("\n## Step 5: rolling recalibration L=125, R=21 (eval half)\n")
    print("| Fold | Old | New | Old verdict | New verdict |")
    print("|---|---|---|---|---|")
    roll_rej = {}
    for f in LABEL:
        o, n = old["folds"][f]["methods"][TF]["rolling"], new["folds"][f]["methods"][TF]["rolling"]
        roll_rej[f] = rejects(n)
        print(f"| {LABEL[f]} | {bt_cell(o)} | {bt_cell(n)} | {'reject' if rejects(o) else 'no reject'} | "
              f"{'reject' if rejects(n) else 'no reject'} |")
    holds = roll_rej["fold1_gfc"] and not any(roll_rej[f] for f in LABEL if f != "fold1_gfc")
    print(f"\nP2a 'rolling rescues all but 2008': {'HOLDS' if holds else 'DOES NOT HOLD'} "
          f"(rejected: {[LABEL[f] for f in LABEL if roll_rej[f]]})")

    print("\n## Step 5: window sweep (new; old in parentheses), Kupiec p at 5%\n")
    Ls = ["40", "60", "90", "125", "175"]
    print("| Fold | " + " | ".join(f"L={L}" for L in Ls) + " |")
    print("|---|" + "---|" * len(Ls))
    for f in LABEL:
        cells = []
        for L in Ls:
            n, o = new["folds"][f]["window_sweep"][L], old["folds"][f]["window_sweep"][L]
            mark = " ✗" if n["p_pof"] < 0.05 else ""
            flip = " **changed**" if (n["p_pof"] < 0.05) != (o["p_pof"] < 0.05) else ""
            cells.append(f"{n['N']}/{n['T']}, {pv(n['p_pof'])}{mark} ({o['N']}/{o['T']}, {pv(o['p_pof'])}){flip}")
        print(f"| {LABEL[f]} | " + " | ".join(cells) + " |")
    all_have_reject = all(any(new["folds"][f]["window_sweep"][L]["p_pof"] < 0.05 for f in LABEL) for L in Ls)
    for L in Ls:
        rej = [LABEL[f] for f in LABEL if new["folds"][f]["window_sweep"][L]["p_pof"] < 0.05]
        rej01 = [LABEL[f] for f in LABEL if new["folds"][f]["window_sweep"][L]["p_pof"] < 0.01]
        print(f"  L={L}: rejected at 5% {rej}; at Bonferroni 0.01 {rej01}")
    print(f"P2b 'no single window passes every fold at 5%': {'HOLDS' if all_have_reject else 'DOES NOT HOLD'}")

    print("\n## Step 6: Christoffersen, frozen offset (new)\n")
    print("| Fold | N/T | Coverage [Wilson 95%] | LR_POF / p | n01,n11 | LR_ind / p | LR_cc / p | Verdict |")
    print("|---|---|---|---|---|---|---|---|")
    for f in LABEL:
        b = new["folds"][f]["methods"][TF]["frozen"]
        lo, hi = b["wilson95"]
        t = b["transitions"]
        rej = [k for k, key in (("POF", "p_pof"), ("IND", "p_ind"), ("CC", "p_cc")) if b[key] < 0.05]
        print(f"| {LABEL[f]} | {b['N']}/{b['T']} | {b['coverage']*100:.2f}% [{lo*100:.1f}, {hi*100:.1f}] | "
              f"{b['lr_pof']:.2f} / {pv(b['p_pof'])} | {t['n01']},{t['n11']} | {b['lr_ind']:.2f} / {pv(b['p_ind'])} | "
              f"{b['lr_cc']:.2f} / {pv(b['p_cc'])} | {'reject ' + '+'.join(rej) if rej else 'no reject'} |")
    b22 = new["folds"]["fold5_2022_bear"]["methods"][TF]["frozen"]
    ok22 = min(b22["p_pof"], b22["p_ind"], b22["p_cc"]) >= 0.05
    print(f"P3 2022 not rejected by any test: {'HOLDS' if ok22 else 'DOES NOT HOLD'}")

    same = True
    for f in LABEL:
        for m in ["HS-250", "EWMA (0.94)", "GARCH(1,1)"]:
            if old["folds"][f]["methods"][m] != new["folds"][f]["methods"][m]:
                same = False
                print(f"  baseline differs: {f} {m}")
        if old["folds"][f]["garch_params"] != new["folds"][f]["garch_params"]:
            same = False
    print(f"\n## Step 8\nP5 classical baselines identical old vs new: {'YES' if same else 'NO'}")
    return {f for f in LABEL if rejects(new["folds"][f]["methods"][TF]["frozen"])}


def step7(mtl_rejected):
    if not os.path.exists(NEW_ST):
        print("\n(step 7 results not present yet)")
        return
    new = load(NEW_ST)
    old_abl = {r["fold"]: r for r in load(OLD_ABL) if r.get("fold")}
    old_rq = load(OLD_RQ)
    print("\n## Step 7: single-task heads, frozen offset (old → new)\n")
    print("| Fold | VaR-proxy head old | new | Return-quantile head old | new |")
    print("|---|---|---|---|---|")
    for f in LABEL:
        oa = old_abl[f]["var_calibration"]
        oac = oa["calibrated"]
        na = new[f]["var_proxy"]["frozen"]
        orq, nrq = old_rq[f]["frozen"], new[f]["return_quantile"]["frozen"]
        print(f"| {LABEL[f]} | {oac['breaches']}/{oa['eval_n']}, p {pv(oac['kupiec_p'])} | "
              f"{na['N']}/{na['T']}, p {pv(na['p_pof'])} cc {pv(na['p_cc'])} | "
              f"{orq['N']}/{orq['T']}, p {pv(orq['p_pof'])} | {nrq['N']}/{nrq['T']}, p {pv(nrq['p_pof'])} cc {pv(nrq['p_cc'])} |")
    print("\nReturn-quantile head, raw and rolling (new):")
    for f in LABEL:
        r = new[f]["return_quantile"]
        print(f"  {LABEL[f]}: epochs {r['epochs_run']}  raw {bt_cell(r['raw'])}  rolling {bt_cell(r['rolling'])}")
    print("VaR-proxy single-task epochs: " + ", ".join(f"{LABEL[f]} {new[f]['var_proxy']['epochs_run']}" for f in LABEL))
    rq_rej = [f for f in mtl_rejected if rejects(new[f]["return_quantile"]["frozen"])]
    print(f"\nP4: return-quantile frozen rejected in {len(rq_rej)}/{len(mtl_rejected)} MTL-rejected folds -> "
          f"{'C2 HOLDS' if len(rq_rej) >= 3 else 'C2 DOES NOT HOLD'}")


if __name__ == "__main__":
    print(__doc__)
    step4()
    rejected = steps5_6_8()
    step7(rejected)
