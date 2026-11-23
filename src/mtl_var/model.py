"""Transformer architecture, loss functions, and the training loop.

Losses and architecture are moved verbatim from scripts/walk_forward.py, including the
squeeze in `weighted_loss` that fixed the broadcast bug (RESULTS.md section 0) and the
permanent shape assertions guarding against its return. `train_model` is the manual
GradientTape loop lifted out of `run_fold` unchanged - same optimizer, same batch size, same
early-stopping rule, same order of operations, so the RNG draw sequence and therefore the
trained weights are identical to the published runs.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model, layers

from .data import SEQUENCE_LENGTH, TARGETS

# Called here rather than in __init__ so it lands immediately after the TensorFlow import,
# exactly as it did at the top of scripts/walk_forward.py.
tf.config.experimental.enable_op_determinism()

WEIGHTS = {"log_return": 0.2, "log_volume": 0.2, "yz_vol": 0.2, "VaR_5": 0.2, "CVaR_5": 0.2}


def _assert_same_shape(y_true, y_pred):
    # Permanent guard against the broadcast bug found in this file: y_true (batch,) minus
    # y_pred (batch,1) doesn't error, it silently broadcasts to (batch,batch). This assert
    # makes that class of bug impossible to reintroduce silently - it fails loudly instead.
    assert y_true.shape == y_pred.shape, (
        f"loss shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape} - "
        f"these must match exactly or the subtraction below silently broadcasts"
    )


def huber_loss(y_true, y_pred, delta=1.0):
    _assert_same_shape(y_true, y_pred)
    e = y_true - y_pred
    return tf.reduce_mean(tf.where(tf.abs(e) < delta, 0.5 * e ** 2, delta * (tf.abs(e) - 0.5 * delta)))


def log_cosh(y_true, y_pred):
    _assert_same_shape(y_true, y_pred)
    return tf.reduce_mean(tf.math.log(tf.cosh(y_pred - y_true + 1e-12)))


def quantile_loss_05(y_true, y_pred):
    q = 0.05
    _assert_same_shape(y_true, y_pred)
    e = y_true - y_pred
    return tf.reduce_mean(tf.maximum(q * e, (q - 1) * e))


def weighted_loss(y_true, y_pred, task):
    # FIX: y_pred comes in as (batch,1) (Dense(1) head output) while y_true is (batch,).
    # Without squeezing, y_true - y_pred broadcasts to (batch,batch) - every cross-pair
    # of a true value against every OTHER sample's prediction - instead of the (batch,)
    # per-sample residual. Verified materially different via verify_loss_bug_impact.py:
    # same seed/architecture/data, only this squeeze differs, and every task's MAE drops
    # 16-74% while the single-split Kupiec result flips from p=0.44 (pass) to p=0.015 (fail).
    y_pred = tf.squeeze(y_pred, axis=-1)
    w = WEIGHTS[task]
    if task in ("log_return", "log_volume", "yz_vol"):
        return w * huber_loss(y_true, y_pred)
    if task == "VaR_5":
        return w * quantile_loss_05(y_true, y_pred)
    return w * log_cosh(y_true, y_pred)


def transformer_block(x, heads=4, ff_dim=128):
    attn = layers.MultiHeadAttention(num_heads=heads, key_dim=ff_dim)(x, x)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization()(x)
    f = layers.Dense(ff_dim, activation="relu")(x)
    f = layers.Dense(x.shape[-1])(f)
    x = layers.Add()([x, f])
    x = layers.LayerNormalization()(x)
    return x


def build_transformer_model(seq_len, n_features):
    inp = Input(shape=(seq_len, n_features))
    x = layers.Dense(64, activation=None)(inp)
    for _ in range(4):
        x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    out_mu = layers.Dense(1, name="log_return")(x)
    out_vol = layers.Dense(1, name="log_volume")(x)
    out_yz = layers.Dense(1, name="yz_vol")(x)
    out_var = layers.Dense(1, name="VaR_5")(x)
    out_cvar = layers.Dense(1, name="CVaR_5")(x)
    return Model(inp, [out_mu, out_vol, out_yz, out_var, out_cvar])


def train_model(X_fit, y_fit_z, X_val, y_val_z, n_features,
                seq_len=SEQUENCE_LENGTH, targets=TARGETS, loss_fn=weighted_loss,
                learning_rate=1e-4, batch_size=32, max_epochs=50, patience=5):
    """The manual GradientTape training loop, lifted out of `run_fold` unchanged.

    `X_val`/`y_val_z` are used only for early stopping and best-weight restoration. They must
    come from the training period: call this through `fit_with_train_tail_validation`, which
    takes no test-window argument at all. Until RESULTS.md section 15 every caller passed the
    fold's test window here.

    Builds the model as its first action so the RNG state at initialisation matches the
    original inline version - callers must `set_seeds()` immediately before, with nothing
    RNG-consuming in between. Returns (model, epochs_run, best_val_loss) with the best
    weights restored.
    """
    model = build_transformer_model(seq_len, n_features)
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    train_ds = tf.data.Dataset.from_tensor_slices(
        (X_fit, {t: tf.cast(y_fit_z[t], tf.float32) for t in targets})
    ).batch(batch_size)
    val_ds = tf.data.Dataset.from_tensor_slices(
        (X_val, {t: tf.cast(y_val_z[t], tf.float32) for t in targets})
    ).batch(batch_size)

    best_val_loss = float('inf')
    wait = 0
    best_weights = None
    epochs_run = 0

    for epoch in range(max_epochs):
        for xb, yb in train_ds:
            with tf.GradientTape() as tape:
                preds = model(xb, training=True)
                L = sum(loss_fn(yb[t], p, t) for t, p in zip(targets, preds))
            grads = tape.gradient(L, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        val_losses = []
        for xb, yb in val_ds:
            preds = model(xb, training=False)
            vL = sum(loss_fn(yb[t], p, t) for t, p in zip(targets, preds))
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
    return model, epochs_run, best_val_loss


def build_single_task_model(seq_len, n_features):
    """Identical transformer body, one Dense(1) head. Copied unchanged from
    scripts/single_task_var_ablation.py so package code does not import from scripts/."""
    inp = Input(shape=(seq_len, n_features))
    x = layers.Dense(64, activation=None)(inp)
    for _ in range(4):
        x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    out_var = layers.Dense(1, name="VaR_5")(x)
    return Model(inp, out_var)


def train_single_task(X_fit, y_fit_z, X_val, y_val_z, n_features,
                      batch_size=32, max_epochs=50, patience=5):
    """Single-head pinball-loss (q=0.05) loop, unchanged from
    scripts/single_task_var_ablation.train_single_task apart from parameter names. Like the
    original it clears the session and seeds (seed 42) itself. `X_val` is for early stopping
    only and must come from the training period."""
    from .seeding import set_seeds

    tf.keras.backend.clear_session()
    set_seeds()

    model = build_single_task_model(SEQUENCE_LENGTH, n_features)
    optimizer = tf.keras.optimizers.Adam(1e-4)

    fit_ds = tf.data.Dataset.from_tensor_slices((X_fit, tf.cast(y_fit_z, tf.float32))).batch(batch_size)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, tf.cast(y_val_z, tf.float32))).batch(batch_size)

    best_val_loss = float('inf')
    wait = 0
    best_weights = None
    epochs_run = 0

    for epoch in range(max_epochs):
        for xb, yb in fit_ds:
            with tf.GradientTape() as tape:
                pred = tf.squeeze(model(xb, training=True), axis=-1)
                loss = quantile_loss_05(yb, pred)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        val_losses = []
        for xb, yb in val_ds:
            pred = tf.squeeze(model(xb, training=False), axis=-1)
            val_losses.append(float(quantile_loss_05(yb, pred)))
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
    return model, epochs_run, best_val_loss


# Share of each fold's training sequences, taken from the chronological END, held out for early
# stopping. Fixed at 0.15 before any clean-validation result existed (RESULTS.md section 15).
VAL_FRACTION = 0.15


def chronological_val_split(X, y, val_fraction=VAL_FRACTION):
    """Split training sequences into (fit, validation) by time: the last `val_fraction` of
    sequences become the validation set. No shuffling. `y` is an array or a dict of arrays
    aligned with X. Returns X_fit, y_fit, X_val, y_val."""
    n = len(X)
    n_val = int(round(n * val_fraction))
    if not 0 < n_val < n:
        raise ValueError(f"validation split of {n_val} from {n} sequences is empty or total")
    cut = n - n_val
    for a in (y.values() if isinstance(y, dict) else [y]):
        assert len(a) == n, "targets must align with sequences"

    def part(a, sl):
        return {k: v[sl] for k, v in a.items()} if isinstance(a, dict) else a[sl]

    return X[:cut], part(y, slice(None, cut)), X[cut:], part(y, slice(cut, None))


def fit_with_train_tail_validation(X_train, y_train_z, n_features, trainer=None,
                                   val_fraction=VAL_FRACTION):
    """Train with early stopping on the chronological tail of the TRAINING sequences.

    The only training entry point the pipeline uses. It takes no test-window argument, so the
    test window cannot reach the training loop, early stopping or best-weight restoration
    (tests/test_no_test_window_in_training.py). Returns
    (model, epochs_run, best_val_loss, n_fit, n_val).
    """
    trainer = trainer if trainer is not None else train_model
    X_fit, y_fit, X_val, y_val = chronological_val_split(X_train, y_train_z, val_fraction)
    model, epochs_run, best_val_loss = trainer(X_fit, y_fit, X_val, y_val, n_features)
    return model, epochs_run, best_val_loss, len(X_fit), len(X_val)
