"""Kupiec proportion-of-failures (POF) backtest.

Moved verbatim from scripts/walk_forward.py. Every coverage verdict in RESULTS.md comes from
`kupiec_from_counts`, including the hand-verified fold5 spot check (3 breaches / 124 days),
so the boundary handling here - N == 0 or N == T returns NaN rather than underflowing - is
part of the published behaviour, not an implementation detail to clean up.
"""
import numpy as np
from scipy.stats import chi2


def kupiec_from_counts(N, T, p=0.05):
    if T == 0 or N == 0 or N == T:
        return float('nan')
    lr = -2 * np.log(((1 - p) ** (T - N) * p ** N) / ((1 - N / T) ** (T - N) * (N / T) ** N))
    return 1 - chi2.cdf(lr, 1)


def kupiec_test(returns, var_pred, p=0.05):
    T = len(var_pred)
    N = int(np.sum(returns < var_pred))
    return N, T, kupiec_from_counts(N, T, p)
