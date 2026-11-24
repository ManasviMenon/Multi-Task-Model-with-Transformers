"""Create all seven research figures from the committed results files.

Install: python -m pip install numpy matplotlib
Run:     python research_graphs.py          (from the repository root)
Output:  ./research_graphs/figures/ (PDF, SVG, 600 dpi PNG)

Every plotted number is read from results/*.json at run time - nothing is hardcoded - so the
figures cannot drift from the data. Neural-model numbers come from the clean-validation runs
(early stopping on the training tail; RESULTS.md section 15). Breach rates are computed from the
integer counts N/T in those files. No retraining, dataset or API key is needed.

The retrospective replay (fig05) is scripts/drift_demo.py run against registry version 4
(@cleanval): the fold-5 model retrained with clean validation, bit-identical in metrics to the
walk-forward run. The served @production model (v2) is not used anywhere here.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle
REPO = Path(__file__).resolve().parent
RES = REPO / 'results'
ROOT = Path.cwd() / 'research_graphs'
ROOT.mkdir(parents=True, exist_ok=True)

FOLD_KEYS = ['fold1_gfc', 'fold2_eu_crisis', 'fold3_2015_selloff', 'fold4_covid', 'fold5_2022_bear']
SOURCES = {
    'walk_forward': 'walk_forward_results_leakage_free_cleanval.json',
    'backtests': 'reviewer_backtests_cleanval_results.json',
    'single_task': 'cleanval_single_task_results.json',
    'single_split': 'cleanval_single_split_results.json',
    'empirical': 'empirical_var_comparison_results.json',
    'summary': 'leakage_free_summary.json',
    'audit': 'leakage_audit_results.json',
    'classical': 'classical_baselines_results.json',
    'replay': 'drift_demo_cleanval_results.json',
    'single_split_single_task': 'cleanval_single_split_single_task_results.json',
}


def _load(key):
    with open(RES / SOURCES[key]) as fh:
        return json.load(fh)


def pct(n, t):
    return 100.0 * n / t


def r2_label(v):
    if v == 1.0:
        return '1.0000'
    if 0.9999 < v < 1.0:
        return f'{v:.5f}'
    return f'{v:.3f}'


def load_data():
    wf = {r['fold']: r for r in _load('walk_forward')}
    bt = _load('backtests')['folds']
    st = _load('single_task')
    ss = _load('single_split')
    emp = _load('empirical')
    summ = _load('summary')
    audit = _load('audit')['target_recoverability_r2']
    classical = _load('classical')
    replay = _load('replay')['replays']['2019_2020_covid_shock']
    sst = _load('single_split_single_task')['frozen_backtest']
    tf_key = 'Transformer (leakage-free)'

    def frozen(f):
        vc = wf[f]['var_calibration']
        return vc['calibrated']['breaches'], vc['eval_n'], vc['calibrated']['kupiec_p']

    def rolling(f):
        es = wf[f]['rolling_calibration']['eval_slice_only']
        return es['breaches'], es['n'], es['kupiec_p']

    # the backtest file must have been built from the same predictions as the walk-forward file
    for f in FOLD_KEYS:
        b = bt[f]['methods'][tf_key]
        assert (b['frozen']['N'], b['frozen']['T']) == frozen(f)[:2], f
        assert (b['rolling']['N'], b['rolling']['T']) == rolling(f)[:2], f

    fr, ro = [frozen(f) for f in FOLD_KEYS], [rolling(f) for f in FOLD_KEYS]
    sweep_days = [40, 60, 90, 125, 175]
    sweep = [[bt[f]['window_sweep'][str(L)] for L in sweep_days] for f in FOLD_KEYS]
    mon_folds = ['fold1_gfc', 'fold4_covid', 'fold2_eu_crisis', 'fold3_2015_selloff']
    mon = [wf[f]['rolling_calibration']['monitor'] for f in mon_folds]
    e250 = [emp[f]['empirical']['250'] for f in FOLD_KEYS]
    stf = [st[f]['var_proxy']['frozen'] for f in FOLD_KEYS]
    ssb = ss['frozen_backtest']
    phases = replay['phases'] + [dict(replay['whole_window'])]
    tasks = ['log_return', 'log_volume', 'yz_vol', 'VaR_5', 'CVaR_5']
    acc = {t: [wf[f]['accuracy'][t]['mae'] for f in FOLD_KEYS] + [ss['accuracy'][t]['mae']] for t in tasks}

    return {
        'folds': ['2008 GFC', '2011 EU crisis', '2015 selloff', '2020 COVID', '2022 bear'],
        'frozen_rolling': {
            'source': SOURCES['walk_forward'],
            'frozen_evaluation_days': [t for _, t, _ in fr],
            'frozen_breaches': [n for n, _, _ in fr],
            'uncalibrated_pct': [pct(wf[f]['var_calibration']['uncalibrated']['breaches'],
                                     wf[f]['var_calibration']['eval_n']) for f in FOLD_KEYS],
            'frozen_pct': [pct(n, t) for n, t, _ in fr],
            'frozen_p': [p for _, _, p in fr],
            'rolling_days': [t for _, t, _ in ro],
            'rolling_breaches': [n for n, _, _ in ro],
            'rolling_pct': [pct(n, t) for n, t, _ in ro],
            'rolling_p': [p for _, _, p in ro],
            'rolling_calibration_days': 125,
            'rolling_refresh_days': 21,
        },
        'window_sweep': {
            'source': SOURCES['backtests'],
            'days': sweep_days,
            'breaches': [[c['N'] for c in row] for row in sweep],
            'denominators': [[c['T'] for c in row] for row in sweep],
            'breach_pct': [[pct(c['N'], c['T']) for c in row] for row in sweep],
            'p': [[c['p_pof'] for c in row] for row in sweep],
        },
        'monitor': {
            'source': SOURCES['walk_forward'],
            'folds': ['2008 GFC', '2020 COVID', '2011 EU crisis', '2015 selloff'],
            'type': ['Acute shock', 'Acute shock', 'Slow grind', 'Slow grind'],
            'elevated_breach_pct': [100 * m['elevated_breach_rate'] for m in mon],
            'normal_breach_pct': [100 * m['normal_breach_rate'] for m in mon],
            'lead_pct': [100 * m['breach_preceded_by_elevated_rate'] for m in mon],
            'base_pct': [100 * m['baseline_preceded_by_elevated_rate'] for m in mon],
        },
        'empirical': {
            'source': [SOURCES['walk_forward'], SOURCES['empirical']],
            'lookback_days': 250,
            'parametric_calibrated_pct': [pct(n, t) for n, t, _ in fr],
            'parametric_calibrated_p': [p for _, _, p in fr],
            'empirical_raw_pct': [pct(e['raw']['breaches'], e['raw']['n']) for e in e250],
            'empirical_raw_p': [e['raw']['kupiec_p'] for e in e250],
            'empirical_calibrated_pct': [pct(e['calibrated']['breaches'], e['calibrated']['n']) for e in e250],
            'empirical_calibrated_p': [e['calibrated']['kupiec_p'] for e in e250],
        },
        'ablation': {
            'source': [SOURCES['walk_forward'], SOURCES['single_task'], SOURCES['single_split']],
            'folds': ['Single split', '2008 GFC', '2011 EU crisis', '2015 selloff', '2020 COVID', '2022 bear'],
            'mtl_pct': [pct(ssb['N'], ssb['T'])] + [pct(n, t) for n, t, _ in fr],
            'mtl_p': [ssb['p_pof']] + [p for _, _, p in fr],
            'single_pct': [pct(sst['N'], sst['T'])] + [pct(b['N'], b['T']) for b in stf],
            'single_p': [sst['p_pof']] + [b['p_pof'] for b in stf],
        },
        'replay': {
            'source': SOURCES['replay'],
            'phases': ['Calm / pre-shock', 'COVID shock', 'Recovery', 'Whole window'],
            'days': [p['days'] for p in phases],
            'breaches': [p['breaches'] for p in phases],
            'breach_pct': [pct(p['breaches'], p['days']) for p in phases],
            'p': [p['kupiec_p'] for p in phases],
            'model': replay['model'],
            'caveat': 'Retrospective replay over 2019-2020 using registry model v4 (trained through '
                      '2021, clean validation) and an offset fitted on the first half of 2022; not a '
                      'prospective out-of-sample forecast test.',
        },
        'leakage': {
            'source': SOURCES['audit'],
            'targets': ['Log-return', 'Log-volume', 'Range volatility', 'VaR (5%)', 'CVaR (5%)'],
            'original_r2': [audit[t]['r2_all_features'] for t in tasks],
            'lagged_r2': [audit[t]['r2_lagged_only'] for t in tasks],
            'original_labels': [r2_label(audit[t]['r2_all_features']) for t in tasks],
            'lagged_labels': [r2_label(audit[t]['r2_lagged_only']) for t in tasks],
            'caveat': 'Full-sample in-sample OLS diagnostic, not Transformer predictive performance.',
        },
        'forecast_accuracy': {
            'source': [SOURCES['walk_forward'], SOURCES['single_split'], SOURCES['summary'], SOURCES['classical']],
            'folds': ['2008 GFC', '2011 EU crisis', '2015 selloff', '2020 COVID', '2022 bear', 'Single split'],
            'return_mae': acc['log_return'],
            'volume_mae': acc['log_volume'],
            'range_volatility_mae': acc['yz_vol'],
            'var_mae': acc['VaR_5'],
            'cvar_mae': acc['CVaR_5'],
            'zero_return_mae': [summ['folds'][f]['naive_baselines']['predict_zero'] for f in FOLD_KEYS]
                               + [ss['naive_baselines']['predict_zero']],
            'persistence_return_mae': [summ['folds'][f]['naive_baselines']['persistence'] for f in FOLD_KEYS]
                                      + [ss['naive_baselines']['persistence']],
            'single_split_volatility_mae_e5': {'GARCH': classical['garch_1_1']['vol_mae'] * 1e5,
                                               'Transformer': ss['accuracy']['yz_vol']['mae'] * 1e5},
        },
    }


D = load_data()

FIG = ROOT / 'figures'
FIG.mkdir(exist_ok=True)
BLUE, ORANGE, GREEN, GRAY, INK = ('#0072B2', '#D55E00', '#009E73', '#7A8693', '#202C38')
FOLDS = D['folds']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8, 'xtick.labelsize': 7.5, 'ytick.labelsize': 8, 'axes.titleweight': 'bold', 'axes.labelcolor': INK, 'text.color': INK, 'axes.edgecolor': '#82909E', 'axes.linewidth': 0.65, 'xtick.color': INK, 'ytick.color': INK, 'grid.color': '#DDE3E9', 'grid.linewidth': 0.5, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'savefig.facecolor': 'white', 'figure.facecolor': 'white', 'legend.frameon': False, 'legend.fontsize': 7.5, 'lines.linewidth': 1.5, 'axes.spines.top': False, 'axes.spines.right': False})

def style(ax, grid='x'):
    ax.set_axisbelow(True)
    ax.grid(axis=grid, alpha=0.8)
    ax.tick_params(length=3, width=0.6)

def rows(ax, labels=FOLDS):
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.tick_params(axis='y', length=0, pad=7)

def target(ax):
    ax.axvline(5, color=INK, ls=(0, (4, 3)), lw=1, zorder=1)

def pformat(p):
    if p is None:
        return 'n/a'
    if p < 0.001:
        return f'{p:.1e}'.replace('e-0', 'e-')
    return f'{p:.3f}'

def save(fig, stem):
    from io import BytesIO
    for fmt in ('pdf', 'png', 'svg'):
        stream = BytesIO()
        fig.savefig(stream, format=fmt, dpi=600 if fmt == 'png' else 150)
        (FIG / f'{stem}.{fmt}').write_bytes(stream.getvalue())
    plt.close(fig)
    print(f'Saved {stem}: PDF, SVG and 600 dpi PNG')
f = D['frozen_rolling']
fig, (a, b) = plt.subplots(1, 2, figsize=(7.16, 3.35), sharey=True, gridspec_kw={'width_ratios': [1.2, 1]})
fig.subplots_adjust(left=0.17, right=0.98, bottom=0.19, top=0.81, wspace=0.19)
y = np.arange(5)
for vals, ps, off, col, marker, label in [(f['frozen_pct'], f['frozen_p'], -0.13, ORANGE, 'o', 'Frozen offset'), (f['rolling_pct'], f['rolling_p'], 0.13, BLUE, 's', 'Rolling, 125 days')]:
    a.scatter(vals, y + off, s=30, c=col, marker=marker, label=label, zorder=4)
    b.scatter(ps, y + off, s=30, c=col, marker=marker, zorder=4)
    for i, v in enumerate(vals):
        a.annotate(f'{v:.2f}', (v, i + off), xytext=(5, 0), textcoords='offset points', va='center', fontsize=7, color=col)
for i in range(5):
    a.plot([f['frozen_pct'][i], f['rolling_pct'][i]], [i - 0.13, i + 0.13], color='#B7C1CB', lw=0.8, zorder=2)
rows(a)
style(a)
style(b)
a.set_xlim(0, max(16.8, max(f['frozen_pct'] + f['rolling_pct']) + 2.6))
a.set_xticks([0, 5, 10, 15])
target(a)
a.set_xlabel('VaR breach rate (%)')
a.set_title('(a) Observed breach rates', loc='left', pad=10)
b.set_xscale('log')
# log axis sized from the data so no p-value can fall off the chart (COVID frozen p is ~1e-11)
p_floor = 10.0 ** np.floor(np.log10(min(f['frozen_p'] + f['rolling_p'])) - 0.5)
b.set_xlim(p_floor, 2)
b.set_xticks([10.0 ** k for k in range(0, int(np.log10(p_floor)) - 1, -3)])
b.axvspan(0.05, 2, color=GREEN, alpha=0.065, zorder=0)
b.axvline(0.05, color=INK, ls=(0, (4, 3)), lw=1)
b.set_xlabel('Reported Kupiec p-value (log scale)')
b.set_title('(b) Unconditional coverage test', loc='left', pad=10)
fig.legend(*a.get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(0.58, 0.985), ncol=2)
fig.text(0.17, 0.035, 'Dashed lines: 5% nominal breach rate (a), and p = 0.05 (b).', fontsize=7, color=GRAY)
save(fig, 'fig01_calibration_across_regimes')
s = D['window_sweep']
rates = np.array(s['breach_pct'])
ps = s['p']
fig, ax = plt.subplots(figsize=(7.16, 3.65))
fig.subplots_adjust(left=0.19, right=0.88, bottom=0.2, top=0.84)
cmap = LinearSegmentedColormap.from_list('coverage_balance', ['#B8D7E8', '#FFFFFF', '#EAC1AE'])
im = ax.imshow(rates - 5, cmap=cmap, norm=TwoSlopeNorm(vmin=-5, vcenter=0, vmax=5), aspect='auto')
for i in range(5):
    for j in range(5):
        p = ps[i][j]
        ax.text(j, i - 0.13, f'{rates[i, j]:.2f}%', ha='center', va='center', fontsize=9, fontweight='bold')
        ax.text(j, i + 0.2, 'p = ' + pformat(p) if p is not None else 'p = n/a', ha='center', va='center', fontsize=6.9, color=INK)
        if p is not None and p >= 0.05:
            ax.add_patch(Rectangle((j - 0.455, i - 0.435), 0.91, 0.87, fill=False, ec=INK, lw=1.05))
ax.set_xticks(range(5), [str(x) for x in s['days']])
ax.set_yticks(range(5), FOLDS)
ax.set_xlabel('Calibration-window length (trading days)', labelpad=8)
ax.tick_params(length=0, pad=7)
ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
ax.grid(which='minor', color='white', lw=2)
ax.tick_params(which='minor', length=0)
for spine in ax.spines.values():
    spine.set_visible(False)
cax = fig.add_axes([0.91, 0.2, 0.019, 0.64])
cbar = fig.colorbar(im, cax=cax, ticks=[-5, 0, 5])
cbar.ax.set_yticklabels(['-5', '0', '+5'])
cbar.set_label('Breach rate minus 5% (pp)', fontsize=7.5, labelpad=8)
cbar.outline.set_visible(False)
fig.text(0.19, 0.93, 'Window length creates a trade-off between the two shock folds', fontsize=9, fontweight='bold')
fig.text(0.19, 0.047, 'Outlined cells: Kupiec p >= 0.05.  Clean-validation models (early stopping on the training tail).', fontsize=7, color=GRAY)
save(fig, 'fig02_calibration_window_sweep')
m = D['monitor']
y = np.arange(4)
fig, (a, b) = plt.subplots(1, 2, figsize=(7.16, 3.25), sharey=True)
fig.subplots_adjust(left=0.17, right=0.98, bottom=0.21, top=0.77, wspace=0.2)
for ax, key1, key2, xlim, title, xlabel in [(a, 'elevated_breach_pct', 'normal_breach_pct', (0, 9), '(a) Conditional breach rate', 'VaR breach rate (%)'), (b, 'lead_pct', 'base_pct', (0, 110), '(b) Reading before a breach', 'Elevated-reading frequency (%)')]:
    for i in range(4):
        ax.plot([m[key1][i], m[key2][i]], [i - 0.12, i + 0.12], color='#B7C1CB', lw=0.9)
    ax.scatter(m[key1], y - 0.12, color=ORANGE, marker='o', s=28, label='Elevated' if ax == a else 'Within 5 days before breach', zorder=3)
    ax.scatter(m[key2], y + 0.12, color=BLUE, marker='s', s=28, label='Normal' if ax == a else 'Baseline frequency', zorder=3)
    for k, off, col in [(key1, -0.12, ORANGE), (key2, 0.12, BLUE)]:
        for i, v in enumerate(m[k]):
            ax.annotate(f'{v:.1f}', (v, i + off), xytext=(5, 0), textcoords='offset points', va='center', fontsize=7, color=col)
    rows(ax, m['folds'])
    style(ax)
    ax.set_xlim(*xlim)
    ax.axhline(1.5, color='#AAB5BF', lw=0.7, ls=':')
    ax.set_title(title, loc='left', pad=30)
    ax.set_xlabel(xlabel)
    ax.legend(loc='lower left', bbox_to_anchor=(-0.02, 1.015), fontsize=6.7, ncol=1, borderaxespad=0, handletextpad=0.5, labelspacing=0.25)
a.set_xticks([0, 2, 4, 6, 8])
b.set_xticks([0, 25, 50, 75, 100])
fig.text(0.17, 0.065, 'Above divider: acute shocks.  Below divider: slow grinds.', fontsize=7, color=GRAY)
fig.text(0.17, 0.022, '2022 is omitted in the source table because it has too few breaches.', fontsize=7, color=GRAY)
save(fig, 'fig03_monitor_regime_dependence')
e = D['empirical']
h = D['ablation']
y = np.arange(5)
fig, (a, b) = plt.subplots(1, 2, figsize=(7.16, 3.65), sharey=True)
fig.subplots_adjust(left=0.17, right=0.985, bottom=0.3, top=0.86, wspace=0.19)
for vals, off, col, marker, label in [(e['parametric_calibrated_pct'], -0.22, BLUE, 's', 'Parametric model, calibrated'), (e['empirical_raw_pct'], 0, GRAY, '^', 'Historical VaR, raw'), (e['empirical_calibrated_pct'], 0.22, ORANGE, 'o', 'Historical VaR, calibrated')]:
    a.scatter(vals, y + off, s=25, c=col, marker=marker, label=label, zorder=3)
    for i, v in enumerate(vals):
        a.annotate(f'{v:.2f}', (v, i + off), xytext=(5, 0), textcoords='offset points', va='center', fontsize=6.6, color=col)
for vals, off, col, marker, label in [(h['mtl_pct'][1:], -0.13, BLUE, 's', 'Multi-task (five heads)'), (h['single_pct'][1:], 0.13, ORANGE, 'o', 'Single-task (VaR only)')]:
    b.scatter(vals, y + off, s=28, c=col, marker=marker, label=label, zorder=3)
    for i, v in enumerate(vals):
        b.annotate(f'{v:.2f}', (v, i + off), xytext=(5, 0), textcoords='offset points', va='center', fontsize=6.6, color=col)
for ax in [a, b]:
    rows(ax)
    style(ax)
    target(ax)
    ax.set_xlim(-0.4, 18.7)
    ax.set_xticks([0, 5, 10, 15])
    ax.set_xlabel('VaR breach rate (%)')
    ax.legend(loc='upper left', bbox_to_anchor=(-0.025, -0.18), fontsize=6.8, labelspacing=0.5, handletextpad=0.5)
a.set_title('(a) Model-free historical VaR', loc='left', pad=10)
b.set_title('(b) Number of prediction heads', loc='left', pad=10)
fig.text(0.17, 0.022, 'Dashed line: nominal 5% breach rate. Historical VaR lookback: 250 days.', fontsize=7, color=GRAY)
save(fig, 'fig04_definition_and_architecture_controls')
r = D['replay']
y = np.array([0, 1, 2, 3.4])
fig, ax = plt.subplots(figsize=(7.16, 3.2))
fig.subplots_adjust(left=0.22, right=0.95, bottom=0.19, top=0.85)
colors = [BLUE, ORANGE, BLUE, GRAY]
ax.barh(y, r['breach_pct'], height=0.5, color=colors, alpha=0.94, zorder=3)
for i, v in enumerate(r['breach_pct']):
    ax.text(v + 0.28, y[i] - 0.09, f"{v:.2f}%  ({r['breaches'][i]}/{r['days'][i]})", va='center', fontsize=8, fontweight='bold', color=colors[i], bbox=dict(facecolor='white', edgecolor='none', pad=0.35))
    ax.text(v + 0.28, y[i] + 0.19, 'Reported p = ' + pformat(r['p'][i]), va='center', fontsize=7, color=GRAY, bbox=dict(facecolor='white', edgecolor='none', pad=0.25))
ax.set_yticks(y, r['phases'])
ax.set_ylim(4, -0.55)
ax.tick_params(axis='y', length=0, pad=8)
# leave room right of the longest bar for its two-line label
ax.set_xlim(0, max(27, max(r['breach_pct']) * 1.45))
ax.set_xticks(list(range(0, int(ax.get_xlim()[1]) + 1, 5)))
target(ax)
style(ax)
ax.axhline(2.72, color='#AAB5BF', ls=':', lw=0.8)
ax.set_xlabel('VaR breach rate (%)')
ax.set_title('Phase-level failures can disappear in the whole-window result', loc='left', pad=12)
fig.text(0.22, 0.025, 'Retrospective replay: 2019-2020; model trained through 2021; offset fitted in 2022.', fontsize=6.9, color=GRAY)
save(fig, 'fig05_retrospective_monitoring_replay')
l = D['leakage']
y = np.arange(5)
fig, ax = plt.subplots(figsize=(7.16, 3.1))
fig.subplots_adjust(left=0.21, right=0.95, bottom=0.2, top=0.8)
for i in range(5):
    ax.plot([l['original_r2'][i], l['lagged_r2'][i]], [i - 0.13, i + 0.13], color='#B7C1CB', lw=0.9)
for key, labels, off, col, marker, label in [('original_r2', 'original_labels', -0.13, ORANGE, 'o', 'Original features'), ('lagged_r2', 'lagged_labels', 0.13, BLUE, 's', 'Lagged features')]:
    ax.scatter(l[key], y + off, color=col, marker=marker, s=29, label=label, zorder=3)
    for i, v in enumerate(l[key]):
        ax.annotate(l[labels][i], (v, i + off), xytext=(5, 0), textcoords='offset points', va='center', fontsize=7, color=col)
rows(ax, l['targets'])
style(ax)
ax.set_xlim(-0.01, 1.14)
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
ax.set_xlabel('In-sample OLS $R^2$')
fig.legend(*ax.get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(0.58, 0.98), ncol=2)
fig.text(0.21, 0.045, 'Diagnostic of information in the inputs; not held-out Transformer performance.', fontsize=7, color=GRAY)
save(fig, 'fig06_feature_leakage_diagnostic')
f = D['forecast_accuracy']
labels = ['2008 GFC', '2011 EU', '2015 selloff', '2020 COVID', '2022 bear', 'Single split']
x = np.arange(6)
fig, (a, b) = plt.subplots(1, 2, figsize=(7.16, 3.35), gridspec_kw={'width_ratios': [2.1, 1]})
fig.subplots_adjust(left=0.085, right=0.975, bottom=0.28, top=0.81, wspace=0.4)
for key, col, marker, label in [('return_mae', BLUE, 's', 'Transformer'), ('zero_return_mae', ORANGE, 'o', 'Zero forecast'), ('persistence_return_mae', GRAY, '^', 'Persistence')]:
    a.scatter(x, np.array(f[key]) * 1000.0, s=29, marker=marker, color=col, label=label, zorder=3)
a.set_xticks(x, labels, rotation=32, ha='right')
a.set_ylim(0, 26)
a.set_yticks([0, 5, 10, 15, 20, 25])
style(a, 'y')
a.set_ylabel('Return MAE ($\\times 10^{-3}$)')
a.set_title('(a) Return forecasts by fold / split', loc='left', pad=10)
a.axvline(4.5, color='#AAB5BF', ls=':', lw=0.8)
a.legend(loc='lower left', bbox_to_anchor=(-0.01, 1.15), ncol=3, fontsize=6.8, handletextpad=0.3, columnspacing=0.8, borderaxespad=0)
vol_e5 = f['single_split_volatility_mae_e5']
vol_vals = [vol_e5['GARCH'], vol_e5['Transformer']]
b.bar([0, 1], vol_vals, width=0.52, color=[GRAY, BLUE], zorder=3)
for i, v in enumerate(vol_vals):
    b.text(i, v + 0.17, f'{v:.2f}', ha='center', fontsize=8, fontweight='bold')
b.set_xticks([0, 1], ['GARCH', 'Transformer'], rotation=25, ha='right')
b.set_ylim(0, 7)
b.set_yticks([0, 2, 4, 6])
style(b, 'y')
b.set_ylabel('Volatility MAE ($\\times 10^{-5}$)')
b.set_title('(b) Single-split volatility', loc='left', pad=10)
fig.text(0.085, 0.04, 'Lower is better. In (b), the model and classical baseline windows differ by one day.', fontsize=7, color=GRAY)
save(fig, 'fig07_forecast_accuracy')

print(f'All seven figures are in: {FIG.resolve()}')

