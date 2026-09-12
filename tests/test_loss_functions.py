"""The check that would have caught the broadcast bug in five minutes (RESULTS.md section 0.2).

Moved from scripts/test_loss_functions.py into pytest form during the Phase 1 packaging
refactor. Same three hand-picked data points, same independently-computed NumPy oracle
values - the expectations are still calculated here in plain NumPy rather than by calling
anything in the package, which is the whole point: it proves the loss functions compute the
right number, not just "a plausibly small number".
"""
import numpy as np
import pytest
import tensorflow as tf

from mtl_var.model import huber_loss, log_cosh, quantile_loss_05, weighted_loss

# Deliberately includes a near-zero residual, a small residual, and one past the huber
# delta=1.0 boundary, so both huber branches and both asymmetric-quantile branches fire.
Y_TRUE = [1.0, 2.0, 3.0]
Y_PRED = [1.5, 2.0, 5.0]  # residuals: -0.5, 0.0, -2.0
TASK_WEIGHT = 0.2  # WEIGHTS[task] is 0.2 for every task in this pipeline


def expected_huber(y_true, y_pred, delta=1.0):
    e = np.array(y_true) - np.array(y_pred)
    per_sample = np.where(np.abs(e) < delta, 0.5 * e ** 2, delta * (np.abs(e) - 0.5 * delta))
    return float(np.mean(per_sample))


def expected_quantile_05(y_true, y_pred, q=0.05):
    e = np.array(y_true) - np.array(y_pred)
    per_sample = np.maximum(q * e, (q - 1) * e)
    return float(np.mean(per_sample))


def expected_log_cosh(y_true, y_pred):
    d = np.array(y_pred) - np.array(y_true) + 1e-12
    per_sample = np.log(np.cosh(d))
    return float(np.mean(per_sample))


@pytest.fixture
def y_true_t():
    return tf.constant(Y_TRUE, dtype=tf.float32)


@pytest.fixture
def y_pred_t():
    return tf.constant(Y_PRED, dtype=tf.float32)


@pytest.fixture
def y_pred_col():
    """Shape (3,1), i.e. what a Dense(1) head actually emits in the training loop."""
    return tf.constant([[p] for p in Y_PRED], dtype=tf.float32)


def test_huber_matches_numpy_oracle(y_true_t, y_pred_t):
    assert float(huber_loss(y_true_t, y_pred_t)) == pytest.approx(expected_huber(Y_TRUE, Y_PRED), rel=1e-5, abs=1e-6)


def test_quantile_05_matches_numpy_oracle(y_true_t, y_pred_t):
    assert float(quantile_loss_05(y_true_t, y_pred_t)) == pytest.approx(
        expected_quantile_05(Y_TRUE, Y_PRED), rel=1e-5, abs=1e-6)


def test_log_cosh_matches_numpy_oracle(y_true_t, y_pred_t):
    assert float(log_cosh(y_true_t, y_pred_t)) == pytest.approx(
        expected_log_cosh(Y_TRUE, Y_PRED), rel=1e-5, abs=1e-6)


@pytest.mark.parametrize("task,oracle", [
    ("log_return", expected_huber(Y_TRUE, Y_PRED)),
    ("log_volume", expected_huber(Y_TRUE, Y_PRED)),
    ("yz_vol", expected_huber(Y_TRUE, Y_PRED)),
    ("VaR_5", expected_quantile_05(Y_TRUE, Y_PRED)),
    ("CVaR_5", expected_log_cosh(Y_TRUE, Y_PRED)),
])
def test_weighted_loss_squeezes_column_predictions(y_true_t, y_pred_col, task, oracle):
    """weighted_loss must squeeze (N,1) -> (N,) and return weight * the per-sample loss.

    This is the exact call shape the training loop uses; before the fix it broadcast to
    (N,N) and silently averaged over every cross-pair instead.
    """
    actual = float(weighted_loss(y_true_t, y_pred_col, task))
    assert actual == pytest.approx(TASK_WEIGHT * oracle, rel=1e-5, abs=1e-6)


def test_shape_assertion_fires_on_the_bug_pattern(y_true_t, y_pred_col):
    """(N,) against unsquashed (N,1) must raise, not silently broadcast."""
    with pytest.raises(AssertionError, match="loss shape mismatch"):
        huber_loss(y_true_t, y_pred_col)


def test_shape_assertion_silent_on_correct_input(y_true_t, y_pred_t):
    huber_loss(y_true_t, y_pred_t)
    quantile_loss_05(y_true_t, y_pred_t)
    log_cosh(y_true_t, y_pred_t)
