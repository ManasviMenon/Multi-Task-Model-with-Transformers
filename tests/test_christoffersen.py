"""Christoffersen (1998) independence / conditional coverage, and the Wilson interval.

Oracles are written out with `math` from the textbook formulas, independent of the
implementation's xlogy-based code path.
"""
import math

import numpy as np
import pytest
from scipy.stats import chi2

from mtl_var.backtest import (
    christoffersen_independence_lr,
    full_backtest,
    kupiec_lr,
    transition_counts,
    wilson_interval,
)


def oracle_lr_ind(n00, n01, n10, n11):
    def ll(k, m, p):  # log p^k (1-p)^m with 0 log 0 = 0
        return (k * math.log(p) if k else 0.0) + (m * math.log(1 - p) if m else 0.0)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    pi01 = n01 / (n00 + n01)
    pi11 = n11 / (n10 + n11)
    return -2 * (ll(n01 + n11, n00 + n10, pi) - ll(n01, n00, pi01) - ll(n11, n10, pi11))


def test_transition_counts_by_hand():
    hits = [0, 1, 1, 0, 0, 0, 1, 0]
    # pairs: 01 11 10 00 00 01 10
    assert transition_counts(hits) == (2, 2, 2, 1)


def test_independence_lr_matches_oracle():
    hits = [0, 1, 1, 0, 0, 0, 1, 0]
    assert christoffersen_independence_lr(hits) == pytest.approx(oracle_lr_ind(2, 2, 2, 1), rel=1e-12)


def test_clustered_breaches_reject_independence():
    hits = np.zeros(250, dtype=int)
    hits[100:112] = 1  # 12 breaches in one block: right frequency, maximal clustering
    bt = full_backtest(np.where(hits, -1.0, 0.0), np.full(250, -0.5))
    assert bt["N"] == 12
    assert bt["p_pof"] > 0.05       # Kupiec sees nothing wrong
    assert bt["p_ind"] < 1e-6       # independence test does
    assert bt["p_cc"] < 1e-6


def test_conditional_coverage_is_sum_on_two_dof():
    rng = np.random.default_rng(0)
    r = rng.standard_normal(400)
    v = np.full(400, -1.645)
    bt = full_backtest(r, v)
    assert bt["lr_cc"] == pytest.approx(bt["lr_pof"] + bt["lr_ind"], rel=1e-12)
    assert bt["lr_pof"] == pytest.approx(kupiec_lr(bt["N"], bt["T"]), rel=1e-12)
    assert bt["p_cc"] == pytest.approx(float(chi2.sf(bt["lr_cc"], 2)), rel=1e-12)


def test_no_breaches_gives_zero_ind_and_flags_untestable():
    bt = full_backtest(np.zeros(251), np.full(251, -1.0))
    assert bt["N"] == 0
    assert bt["lr_ind"] == 0.0
    assert not bt["ind_testable"]
    assert bt["lr_pof"] == pytest.approx(-2 * 251 * math.log(0.95), rel=1e-12)
    assert bt["p_cc"] < 1e-5


def test_wilson_interval_by_hand():
    # 4 / 124 (the 2022 frozen-offset cell): z = 1.959964
    N, T, z = 4, 124, 1.959963984540054
    p = N / T
    c = (p + z * z / (2 * T)) / (1 + z * z / T)
    h = z * math.sqrt(p * (1 - p) / T + z * z / (4 * T * T)) / (1 + z * z / T)
    lo, hi = wilson_interval(N, T)
    assert lo == pytest.approx(c - h, rel=1e-12)
    assert hi == pytest.approx(c + h, rel=1e-12)
    assert lo < 0.05 < hi
