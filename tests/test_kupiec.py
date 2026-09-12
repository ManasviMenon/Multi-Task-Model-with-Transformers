"""Kupiec POF test verification.

Formalises the hand-check recorded in RESULTS.md section 0.5, where fold5's breach count was
counted with a plain Python loop over the cached predictions and matched the pipeline's
reported coverage and Kupiec p exactly. That check lived only in the write-up; here it is a
test that runs.

The oracle values are computed from the likelihood-ratio formula directly with `math`, and
the tail probability via `chi2.sf` rather than the implementation's `1 - chi2.cdf`, so this
is an independent path to the same number rather than a restatement of the code.
"""
import math

import numpy as np
import pytest
from scipy.stats import chi2

from mtl_var.backtest import kupiec_from_counts, kupiec_test

# fold5_2022_bear, calibrated, as recorded in results/walk_forward_results.json
FOLD5_BREACHES = 3
FOLD5_DAYS = 124
FOLD5_COVERAGE = 0.024193548387096774
FOLD5_KUPIEC_P = 0.1443909755485061


def oracle_kupiec_p(N, T, p=0.05):
    """LR_POF from the formula, tail probability via chi2.sf - an independent path."""
    lr = -2 * math.log(
        ((1 - p) ** (T - N) * p ** N) / ((1 - N / T) ** (T - N) * (N / T) ** N)
    )
    return float(chi2.sf(lr, 1))


def test_fold5_spot_check_reproduces_recorded_result():
    """The RESULTS.md section 0.5 hand-count, as a test: 3 breaches / 124 days."""
    assert FOLD5_BREACHES / FOLD5_DAYS == pytest.approx(FOLD5_COVERAGE)
    assert kupiec_from_counts(FOLD5_BREACHES, FOLD5_DAYS) == pytest.approx(FOLD5_KUPIEC_P, rel=1e-9)


def test_matches_independent_oracle():
    for N, T in [(3, 124), (15, 471), (28, 471), (12, 251), (30, 376)]:
        assert kupiec_from_counts(N, T) == pytest.approx(oracle_kupiec_p(N, T), rel=1e-9)


def test_perfect_coverage_gives_p_of_one():
    """Observed rate exactly on target -> LR = 0 -> p = 1."""
    assert kupiec_from_counts(5, 100) == pytest.approx(1.0)


def test_undercoverage_and_overcoverage_both_rejected():
    """The test is two-sided in effect: far too few and far too many breaches both reject."""
    assert kupiec_from_counts(1, 400) < 0.05    # 0.25% vs 5% target
    assert kupiec_from_counts(60, 400) < 0.05   # 15% vs 5% target


@pytest.mark.parametrize("N,T", [(0, 100), (100, 100), (0, 0)])
def test_boundary_cases_return_nan_not_a_number(N, T):
    """Zero or all breaches are undefined, and must not underflow into a bogus p-value.

    This is published behaviour - RESULTS.md reports 'p=n/a' for the L=175 GFC cell and the
    empirical-VaR 2008 cell, both of which are the N == 0 case.
    """
    assert math.isnan(kupiec_from_counts(N, T))


def test_kupiec_test_counts_breaches_as_strictly_below():
    """A return exactly equal to the VaR level is not a breach; strictly below is."""
    returns = np.array([-0.05, -0.02, 0.01, -0.03])
    var_pred = np.array([-0.03, -0.02, -0.03, -0.02])
    #   idx 0: -0.05 <  -0.03  -> breach
    #   idx 1: -0.02 == -0.02  -> NOT a breach (the case this test exists for)
    #   idx 2:  0.01 >  -0.03  -> no breach
    #   idx 3: -0.03 <  -0.02  -> breach
    N, T, _ = kupiec_test(returns, var_pred)
    assert (N, T) == (2, 4)
