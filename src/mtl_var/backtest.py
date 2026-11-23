"""VaR backtests: Kupiec proportion-of-failures (POF) and Christoffersen (1998) independence and
conditional coverage, plus a Wilson interval on the breach rate.

Every coverage verdict in RESULTS.md comes from `kupiec_from_counts`, including the
hand-verified fold5 spot check (3 breaches / 124 days).

Boundary handling (corrected in response to the ACAI 2026 review, RESULTS.md section 14): with
zero breaches, or with every day a breach, the Kupiec likelihood ratio is NOT undefined. Taking
0*log(0) = 0, the unrestricted likelihood p_hat^N (1-p_hat)^(T-N) is exactly 1, so
LR = -2*T*ln(1-p) at N = 0 and LR = -2*T*ln(p) at N = T. For N = 0, T = 251, p = 0.05 that gives
LR = 25.75 and p = 3.9e-7: a rejection. This module previously returned NaN for both cases, and
two table cells were published as "n/a" as a result; they are failures.

All statistics are computed in log space with `scipy.special.xlogy`, which implements the
0*log(0) = 0 convention, and tail probabilities use `chi2.sf` rather than `1 - chi2.cdf` so very
small p-values keep their precision. For interior cases (0 < N < T) this reproduces the recorded
p-values to within 1e-9 relative (tests/test_refactor_parity.py).
"""
import numpy as np
from scipy.special import xlogy
from scipy.stats import chi2, norm


def _bernoulli_loglik(n_hit, n_miss, prob):
    """log of prob^n_hit (1-prob)^n_miss, with 0*log(0) = 0."""
    return xlogy(n_hit, prob) + xlogy(n_miss, 1.0 - prob)


def kupiec_lr(N, T, p=0.05):
    """Kupiec LR_POF statistic. Defined for every 0 <= N <= T with T > 0; NaN only if T == 0."""
    if T == 0:
        return float('nan')
    p_hat = N / T
    lr = -2.0 * (_bernoulli_loglik(N, T - N, p) - _bernoulli_loglik(N, T - N, p_hat))
    return float(max(lr, 0.0)) + 0.0  # clip rounding below zero; +0.0 turns -0.0 into 0.0


def kupiec_from_counts(N, T, p=0.05):
    """Kupiec POF p-value (chi-square, 1 dof). NaN only when there are no observations."""
    lr = kupiec_lr(N, T, p)
    if np.isnan(lr):
        return float('nan')
    return float(chi2.sf(lr, 1))


def kupiec_test(returns, var_pred, p=0.05):
    T = len(var_pred)
    N = int(np.sum(returns < var_pred))
    return N, T, kupiec_from_counts(N, T, p)


def transition_counts(hits):
    """2x2 transition counts of a breach indicator sequence: (n00, n01, n10, n11), where n_ij is
    the number of days in state j whose previous day was in state i (1 = breach)."""
    h = np.asarray(hits, dtype=int)
    prev, curr = h[:-1], h[1:]
    n00 = int(np.sum((prev == 0) & (curr == 0)))
    n01 = int(np.sum((prev == 0) & (curr == 1)))
    n10 = int(np.sum((prev == 1) & (curr == 0)))
    n11 = int(np.sum((prev == 1) & (curr == 1)))
    return n00, n01, n10, n11


def christoffersen_independence_lr(hits):
    """Christoffersen (1998) LR_ind: first-order Markov alternative against independence.

    pi01 = n01/(n00+n01), pi11 = n11/(n10+n11), pi = (n01+n11)/(T-1). With 0*log(0) = 0 the
    statistic is defined in every case; when a row of the transition matrix is empty (e.g. no
    breaches at all, so no day follows a breach) that row contributes nothing and LR_ind = 0 -
    there is no clustering to detect, and the test has no power, which is reported alongside.
    """
    n00, n01, n10, n11 = transition_counts(hits)
    n0, n1 = n00 + n01, n10 + n11
    total = n0 + n1
    if total == 0:
        return float('nan')
    pi = (n01 + n11) / total
    pi01 = n01 / n0 if n0 else 0.0
    pi11 = n11 / n1 if n1 else 0.0
    ll_restricted = _bernoulli_loglik(n01 + n11, n00 + n10, pi)
    ll_markov = _bernoulli_loglik(n01, n00, pi01) + _bernoulli_loglik(n11, n10, pi11)
    return float(max(-2.0 * (ll_restricted - ll_markov), 0.0)) + 0.0  # +0.0 turns -0.0 into 0.0


def wilson_interval(N, T, conf=0.95):
    """Wilson score interval for a binomial proportion N/T."""
    if T == 0:
        return (float('nan'), float('nan'))
    z = norm.ppf(0.5 + conf / 2)
    phat = N / T
    denom = 1 + z ** 2 / T
    centre = (phat + z ** 2 / (2 * T)) / denom
    half = z * np.sqrt(phat * (1 - phat) / T + z ** 2 / (4 * T ** 2)) / denom
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def full_backtest(returns, var_pred, p=0.05):
    """Kupiec POF, Christoffersen independence and conditional coverage, from integer counts.

    LR_cc = LR_POF + LR_ind ~ chi2(2). Following Christoffersen (1998), LR_POF is computed on
    all T days and LR_ind on the T-1 transitions; both are reported so the arithmetic can be
    checked by hand from N, T and the transition counts.
    """
    returns = np.asarray(returns, dtype=float)
    var_pred = np.asarray(var_pred, dtype=float)
    hits = returns < var_pred
    T = int(len(hits))
    N = int(hits.sum())
    n00, n01, n10, n11 = transition_counts(hits)
    lr_pof = kupiec_lr(N, T, p)
    lr_ind = christoffersen_independence_lr(hits)
    lr_cc = lr_pof + lr_ind
    lo, hi = wilson_interval(N, T)
    exceed = returns[hits] - var_pred[hits]  # negative: how far past the VaR each breach went
    return {
        "T": T, "N": N, "coverage": N / T if T else None,
        "wilson95": [lo, hi],
        "transitions": {"n00": n00, "n01": n01, "n10": n10, "n11": n11},
        "lr_pof": lr_pof, "p_pof": float(chi2.sf(lr_pof, 1)),
        "lr_ind": lr_ind, "p_ind": float(chi2.sf(lr_ind, 1)),
        "lr_cc": lr_cc, "p_cc": float(chi2.sf(lr_cc, 2)),
        "ind_testable": bool(n10 + n11 > 0),
        "mean_exceedance": float(exceed.mean()) if N else None,
        "max_exceedance": float(exceed.min()) if N else None,
    }
