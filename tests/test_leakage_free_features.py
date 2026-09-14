"""Tests for the leakage-free feature set (RESULTS.md section 13).

Two properties matter. The package's feature construction must be byte-identical to the script
that produced the section 13 numbers - otherwise the served model would be fed features it was
never evaluated on. And the construction must actually be leakage-free, which is tested by its
defining property rather than by inspecting column names: perturb day t's own bar, and row t's
features must not move at all.
"""
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

from mtl_var.data import (
    CONTAMINATED_FEATURES,
    FEATURE_SETS,
    FEATURES,
    LEAKAGE_FREE_FEATURES,
    build_leakage_free_frame,
    engineer_features,
    frame_for_feature_set,
    load_and_engineer_features,
)

CSV = "SPX500-18Y_scaled_y.csv"
SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture(scope="module")
def engineered():
    return load_and_engineer_features(CSV)


@pytest.fixture(scope="module")
def raw_bars():
    raw = pd.read_csv(CSV, parse_dates=["observation_date"], dayfirst=True)
    raw = raw.dropna(axis=1, how="all").drop(index=0).dropna(subset=["observation_date"])
    return raw.reset_index(drop=True)


def test_feature_set_is_a_one_for_one_shift():
    assert len(LEAKAGE_FREE_FEATURES) == len(FEATURES) == 18
    assert not set(CONTAMINATED_FEATURES) & set(LEAKAGE_FREE_FEATURES)
    for original, clean in zip(FEATURES, LEAKAGE_FREE_FEATURES):
        assert clean == (f"{original}_lag1" if original in CONTAMINATED_FEATURES else original)
    assert FEATURE_SETS == {"original": FEATURES, "leakage_free": LEAKAGE_FREE_FEATURES}


def test_matches_the_script_that_produced_section_13(engineered):
    sys.path.insert(0, str(SCRIPTS))
    try:
        from leakage_free_retrain import build_clean_frame
    finally:
        sys.path.remove(str(SCRIPTS))

    script_frame, script_names = build_clean_frame(engineered)
    assert script_names == LEAKAGE_FREE_FEATURES
    pd.testing.assert_frame_equal(build_leakage_free_frame(engineered), script_frame)


def test_lagged_columns_are_exact_one_day_shifts(engineered):
    clean = build_leakage_free_frame(engineered)
    assert len(clean) == len(engineered) - 1  # the first row has no lag

    for f in CONTAMINATED_FEATURES:
        np.testing.assert_array_equal(clean[f"{f}_lag1"].to_numpy(), engineered[f].to_numpy()[:-1])

    # The accepted redundancy: yesterday's close log-ratio is exactly yesterday's return.
    np.testing.assert_array_equal(clean["log_close_ret_lag1"].to_numpy(), clean["ret_lag1"].to_numpy())


def test_row_t_is_invariant_to_day_t_prices(raw_bars):
    """The defining property of a leakage-free feature row. Move day t's entire bar - prices up
    5%, volume doubled - and the leakage-free features for row t must be bit-identical, while
    the original set and the targets move."""
    window = raw_bars.tail(200).reset_index(drop=True)
    bumped = window.copy()
    last = bumped.index[-1]
    for col in ("Open", "High", "Low", "Close", "Adj Close"):
        bumped.loc[last, col] *= 1.05
    bumped.loc[last, "Volume"] *= 2.0

    base = build_leakage_free_frame(engineer_features(window))
    moved = build_leakage_free_frame(engineer_features(bumped))
    assert base["observation_date"].iloc[-1] == moved["observation_date"].iloc[-1]

    np.testing.assert_array_equal(base[LEAKAGE_FREE_FEATURES].iloc[-1].to_numpy(),
                                  moved[LEAKAGE_FREE_FEATURES].iloc[-1].to_numpy())

    # ...and the leak in the original set is real: the same perturbation moves it.
    assert base["log_close_ret"].iloc[-1] != moved["log_close_ret"].iloc[-1]
    assert base["log_return"].iloc[-1] != moved["log_return"].iloc[-1]
    assert base["log_close_ret"].iloc[-1] == base["log_return"].iloc[-1]


def test_frame_for_feature_set_dispatch(engineered):
    assert frame_for_feature_set(engineered, "original") is engineered
    clean = frame_for_feature_set(engineered, "leakage_free")
    assert set(LEAKAGE_FREE_FEATURES) <= set(clean.columns)
    with pytest.raises(ValueError):
        frame_for_feature_set(engineered, "not-a-feature-set")
