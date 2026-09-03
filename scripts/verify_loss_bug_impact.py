"""
Decisive test: how much does the training-loss broadcast bug (walk_forward.py:142-148,
weighted_loss called with y_true shape (batch,) against y_pred shape (batch,1), which
broadcasts to (batch,batch) cross-pairs instead of the intended (batch,) per-sample
residual) actually change the trained model?

Retrains the 5-head MTL transformer ONCE on the single 80/20 split (matching
MTL_Improved.ipynb's methodology exactly - same features, same standardization, same
equal task weights, same architecture, same seed/init), twice:

  - "buggy":  exact reproduction of the current weighted_loss call (unsquashed y_pred)
  - "fixed":  identical in every other respect, y_pred squeezed to (batch,) before the
              same underlying loss functions (huber_loss/quantile_loss_05/log_cosh,
              imported unmodified from walk_forward.py) are applied

Both runs use set_seeds() immediately before training so weight initialization is
identical - the ONLY variable that differs between the two runs is the loss computation.

Compared against the numbers already recorded in RESULTS.md sections 1-2 (from the
buggy-trained model): per-task MAE/RMSE, and VaR coverage/Kupiec p (uncalibrated and
post-hoc calibrated, same split-in-half offset procedure).

Decision rule (given, not invented here): if the fixed-loss numbers are essentially the
same as the buggy ones, the bug was benign and gets a footnote. If they're materially
different, sections 3-9 (the whole walk-forward campaign) need re-running on the
corrected loss before the paper's central claim can be trusted.

Does not modify MTL_Improved.ipynb or walk_forward.py.
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import chi2

from walk_forward import (
    FEATURES, SEQUENCE_LENGTH, TARGETS, WEIGHTS,
    huber_loss, log_cosh, quantile_loss_05,
    build_transformer_model, load_and_engineer_features,
    kupiec_from_counts, make_sequences, set_seeds,
)

CSV_PATH = "SPX500-18Y_scaled_y.csv"


def weighted_loss_buggy(y_true, y_pred, task):
    """Exact reproduction of walk_forward.py:142-148 - y_pred is NOT squeezed."""
    w = WEIGHTS[task]
    if task in ("log_return", "log_volume", "yz_vol"):
        return w * huber_loss(y_true, y_pred)
    if task == "VaR_5":
        return w * quantile_loss_05(y_true, y_pred)
    return w * log_cosh(y_true, y_pred)


def weighted_loss_fixed(y_true, y_pred, task):
    """Same loss functions, y_pred squeezed to (batch,) first - correct per-sample loss."""
    y_pred = tf.squeeze(y_pred, axis=-1)
    w = WEIGHTS[task]
    if task in ("log_return", "log_volume", "yz_vol"):
        return w * huber_loss(y_true, y_pred)
    if task == "VaR_5":
        return w * quantile_loss_05(y_true, y_pred)
    return w * log_cosh(y_true, y_pred)


def build_single_split():
    """Reproduces MTL_Improved.ipynb's single 80/20 split methodology exactly, including
    its existing (pre-existing, not introduced here) quirk of fitting the FEATURE scaler
    on the full dataset rather than train-only - kept as-is because this test isolates
    ONLY the loss-computation variable; changing anything else would confound it."""
    data = load_and_engineer_features(CSV_PATH)

    feat_scaler = StandardScaler()
    X_scaled = feat_scaler.fit_transform(data[FEATURES])
    X_seq = make_sequences(X_scaled, SEQUENCE_LENGTH)

    y_raw = {t: data[t].values[SEQUENCE_LENGTH - 1:] for t in TARGETS}

    split_index = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split_index], X_seq[split_index:]
    y_train_raw = {t: y_raw[t][:split_index] for t in TARGETS}
    y_test_raw = {t: y_raw[t][split_index:] for t in TARGETS}

    target_scalers = {}
    y_train_z, y_test_z = {}, {}
    for t in TARGETS:
        ts = StandardScaler()
        y_train_z[t] = ts.fit_transform(y_train_raw[t].reshape(-1, 1)).flatten()
        y_test_z[t] = ts.transform(y_test_raw[t].reshape(-1, 1)).flatten()
        target_scalers[t] = ts

    return X_train, X_test, y_train_raw, y_test_raw, y_train_z, y_test_z, target_scalers


def train_once(loss_fn, X_train, y_train_z, X_test, y_test_z, n_features,
                batch_size=32, max_epochs=50, patience=5):
    tf.keras.backend.clear_session()
    set_seeds()  # identical weight init across both runs - the only thing that should differ is loss_fn

    model = build_transformer_model(SEQUENCE_LENGTH, n_features)
    optimizer = tf.keras.optimizers.Adam(1e-4)

    train_ds = tf.data.Dataset.from_tensor_slices(
        (X_train, {t: tf.cast(y_train_z[t], tf.float32) for t in TARGETS})
    ).batch(batch_size)
    val_ds = tf.data.Dataset.from_tensor_slices(
        (X_test, {t: tf.cast(y_test_z[t], tf.float32) for t in TARGETS})
    ).batch(batch_size)

    best_val_loss = float('inf')
    wait = 0
    best_weights = None
    epochs_run = 0

    for epoch in range(max_epochs):
        for xb, yb in train_ds:
            with tf.GradientTape() as tape:
                mu_p, vol_p, yz_p, var_p, cvar_p = model(xb, training=True)
                L = sum(loss_fn(yb[t], p, t) for t, p in
                        zip(TARGETS, [mu_p, vol_p, yz_p, var_p, cvar_p]))
            grads = tape.gradient(L, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        val_losses = []
        for xb, yb in val_ds:
            mu_p, vol_p, yz_p, var_p, cvar_p = model(xb, training=False)
            vL = sum(loss_fn(yb[t], p, t) for t, p in
                     zip(TARGETS, [mu_p, vol_p, yz_p, var_p, cvar_p]))
            val_losses.append(float(vL))
        val_loss = float(np.mean(val_losses))
        epochs_run = epoch + 1

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = model.get_weights()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.set_weights(best_weights)
    print(f"  trained {epochs_run} epochs, best val_loss={best_val_loss:.6f}")
    return model


def evaluate(model, X_test, y_test_raw, target_scalers):
    preds = model.predict(X_test, verbose=0)
    native = {}
    for t, p in zip(TARGETS, preds):
        native[t] = target_scalers[t].inverse_transform(p.reshape(-1, 1)).flatten()

    accuracy = {}
    for t in TARGETS:
        y_true, y_pred = y_test_raw[t], native[t]
        accuracy[t] = {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        }

    returns = y_test_raw['log_return']
    var_pred = native['VaR_5']
    calib_n = len(var_pred) // 2
    returns_calib, returns_eval = returns[:calib_n], returns[calib_n:]
    var_calib, var_eval = var_pred[:calib_n], var_pred[calib_n:]
    offset = float(np.quantile(returns_calib - var_calib, 0.05))
    var_eval_cal = var_eval + offset

    def kupiec_eval(returns_e, var_e):
        T = len(var_e)
        N = int(np.sum(returns_e < var_e))
        return N, T, kupiec_from_counts(N, T)

    N0, T0, p0 = kupiec_eval(returns_eval, var_eval)
    N1, T1, p1 = kupiec_eval(returns_eval, var_eval_cal)

    var_result = {
        "uncalibrated": {"coverage": N0 / T0, "kupiec_p": p0},
        "calibrated": {"coverage": N1 / T1, "kupiec_p": p1},
        "offset": offset,
    }
    return accuracy, var_result


def main():
    print(__doc__)
    print("Building single 80/20 split (matches MTL_Improved.ipynb)...")
    X_train, X_test, y_train_raw, y_test_raw, y_train_z, y_test_z, target_scalers = build_single_split()
    n_features = len(FEATURES)
    print(f"n_train={len(X_train)}  n_test={len(X_test)}  n_features={n_features}\n")

    print("=== Training with the BUGGY loss (exact reproduction of current code) ===")
    model_buggy = train_once(weighted_loss_buggy, X_train, y_train_z, X_test, y_test_z, n_features)
    acc_buggy, var_buggy = evaluate(model_buggy, X_test, y_test_raw, target_scalers)

    print("\n=== Training with the FIXED loss (y_pred squeezed, everything else identical) ===")
    model_fixed = train_once(weighted_loss_fixed, X_train, y_train_z, X_test, y_test_z, n_features)
    acc_fixed, var_fixed = evaluate(model_fixed, X_test, y_test_raw, target_scalers)

    print("\n" + "=" * 90)
    print("PER-TASK MAE  (RESULTS.md section 1 recorded, buggy-trained: log_return 0.0096, "
          "log_volume 0.1529, yz_vol 0.00004, VaR5 0.0058, CVaR5 0.0063)")
    print(f"{'task':14s} {'buggy (this run)':>18s} {'fixed (this run)':>18s} {'ratio fixed/buggy':>20s}")
    for t in TARGETS:
        b, f = acc_buggy[t]['mae'], acc_fixed[t]['mae']
        print(f"{t:14s} {b:>18.5f} {f:>18.5f} {f/b if b else float('nan'):>20.2f}")

    print("\nVaR coverage / Kupiec p  (RESULTS.md section 2 recorded, buggy-trained: "
          "29.5% uncal -> 4.25% cal, p=0.44)")
    print(f"{'':14s} {'uncal cov':>10s} {'uncal p':>10s} {'cal cov':>10s} {'cal p':>10s}")
    print(f"{'buggy':14s} {var_buggy['uncalibrated']['coverage']*100:>9.2f}% "
          f"{var_buggy['uncalibrated']['kupiec_p']:>10.4g} "
          f"{var_buggy['calibrated']['coverage']*100:>9.2f}% "
          f"{var_buggy['calibrated']['kupiec_p']:>10.4g}")
    print(f"{'fixed':14s} {var_fixed['uncalibrated']['coverage']*100:>9.2f}% "
          f"{var_fixed['uncalibrated']['kupiec_p']:>10.4g} "
          f"{var_fixed['calibrated']['coverage']*100:>9.2f}% "
          f"{var_fixed['calibrated']['kupiec_p']:>10.4g}")


if __name__ == "__main__":
    main()
