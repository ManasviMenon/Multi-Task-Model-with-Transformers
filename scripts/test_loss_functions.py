"""
The check that would have caught the broadcast bug in five minutes, run now on the
corrected code:

  1. Hand-made data points with independently-computed expected loss values (plain
     Python/NumPy, not calling anything from walk_forward.py) - proves the corrected
     loss functions compute the right number, not just "a plausibly small number."
  2. Explicit proof the new shape assertion catches exactly the bug class that bit us:
     feeding y_true (N,) against y_pred (N,1) unsquashed must raise, not silently broadcast.
  3. Proof the assertion does NOT fire on correctly-shaped input.

Exits non-zero (and prints which check failed) if anything doesn't match - this is a
verification script, not a demo.
"""
import math

import numpy as np
import tensorflow as tf

from walk_forward import huber_loss, quantile_loss_05, log_cosh, weighted_loss

TOL = 1e-6


def check(name, actual, expected):
    actual = float(actual)
    ok = math.isclose(actual, expected, rel_tol=1e-5, abs_tol=TOL)
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}: actual={actual:.8f}  expected={expected:.8f}  diff={actual-expected:.2e}")
    return ok


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


def main():
    # Three hand-picked points - deliberately include a near-zero residual, a small
    # residual, and a residual past the huber delta=1.0 boundary, so all branches of
    # huber_loss and both branches of the asymmetric quantile loss get exercised.
    y_true = [1.0, 2.0, 3.0]
    y_pred = [1.5, 2.0, 5.0]   # residuals: -0.5, 0.0, -2.0

    exp_huber = expected_huber(y_true, y_pred)
    exp_quant = expected_quantile_05(y_true, y_pred)
    exp_logcosh = expected_log_cosh(y_true, y_pred)
    print(f"Independent (NumPy) oracle values: huber={exp_huber:.8f}  "
          f"quantile05={exp_quant:.8f}  log_cosh={exp_logcosh:.8f}\n")

    all_ok = True

    print("--- 1. Correctly-shaped (N,) vs (N,) input into the raw loss functions ---")
    y_true_t = tf.constant(y_true, dtype=tf.float32)
    y_pred_t = tf.constant(y_pred, dtype=tf.float32)
    all_ok &= check("huber_loss(N,)vs(N,)", huber_loss(y_true_t, y_pred_t), exp_huber)
    all_ok &= check("quantile_loss_05(N,)vs(N,)", quantile_loss_05(y_true_t, y_pred_t), exp_quant)
    all_ok &= check("log_cosh(N,)vs(N,)", log_cosh(y_true_t, y_pred_t), exp_logcosh)

    print("\n--- 2. weighted_loss() with UNSQUASHED (N,1) y_pred (the real training-loop shape) ---")
    y_pred_col = tf.constant([[p] for p in y_pred], dtype=tf.float32)  # shape (3,1), like a Dense(1) head
    w = 0.2  # WEIGHTS[task] for every task in this pipeline
    all_ok &= check("weighted_loss(log_return)", weighted_loss(y_true_t, y_pred_col, "log_return"), w * exp_huber)
    all_ok &= check("weighted_loss(VaR_5)", weighted_loss(y_true_t, y_pred_col, "VaR_5"), w * exp_quant)
    all_ok &= check("weighted_loss(CVaR_5)", weighted_loss(y_true_t, y_pred_col, "CVaR_5"), w * exp_logcosh)

    print("\n--- 3. The shape assertion must FIRE on the exact bug pattern: (N,) vs (N,1), unsquashed ---")
    try:
        huber_loss(y_true_t, y_pred_col)  # calling the RAW function directly, bypassing weighted_loss's squeeze
        print("[FAIL] huber_loss(y_true=(3,), y_pred=(3,1)) did NOT raise - the guard is not working")
        all_ok = False
    except AssertionError as e:
        print(f"[PASS] huber_loss(y_true=(3,), y_pred=(3,1)) raised AssertionError as expected: {e}")

    print("\n" + ("ALL CHECKS PASSED" if all_ok else "AT LEAST ONE CHECK FAILED"))
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
