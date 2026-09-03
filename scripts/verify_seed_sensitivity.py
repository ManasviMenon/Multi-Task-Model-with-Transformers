"""
Before writing the strong claim ("single-split Kupiec verdicts are seed-fragile"), verify
it with more than one data point. The notebook's full pipeline (concept model trained first,
consuming RNG state, then transformer trained without re-seeding) gave a PASS (p=0.740) on
the single split, while the "clean" pipeline used throughout the walk-forward campaign
(transformer only, freshly seeded with set_seeds() -> SEED=42) gave a FAIL (p=0.0152) on the
identical data and loss. This tests two things:

  1. General seed-sensitivity: does the CLEAN pipeline (no concept model, matching
     walk_forward.py's methodology) itself swing between pass/fail just from which seed is
     used, independent of any concept-model involvement? Runs seed=43 and seed=44 in
     addition to the already-known seed=42 result.

  2. The specific mechanism: does training an unrelated dummy model first (consuming global
     RNG draws), THEN building the real transformer WITHOUT re-seeding in between - mirroring
     exactly what the notebook's cell structure does - reproduce a result closer to the
     notebook's pass, using seed=42 throughout (so seed value itself isn't the variable)?

Does not modify walk_forward.py, MTL_Improved.ipynb, or any existing results file.
"""
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input

from walk_forward import SEQUENCE_LENGTH, TARGETS, build_transformer_model
from verify_loss_bug_impact import build_single_split, evaluate, weighted_loss_fixed

FEATURES_N = 18


def set_seeds_custom(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def train_once_no_reseed(loss_fn, X_train, y_train_z, X_test, y_test_z, n_features,
                          batch_size=32, max_epochs=50, patience=5):
    """Identical to verify_loss_bug_impact.train_once EXCEPT it does not call set_seeds()
    internally. That internal re-seed is exactly what this script needs to NOT do - it
    would silently reset the RNG stream back to a fresh SEED=42 draw sequence right before
    building the model, undoing whatever seed or dummy-model-training happened just before
    calling this function. (Caught this via a bogus result: seed=43 reproduced seed=42's
    numbers bit-for-bit, which is what led to writing this function instead of reusing
    train_once as-is.)"""
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


def build_dummy_model(seq_len, n_features):
    """An architecturally irrelevant model, just to consume global RNG draws the way the
    notebook's concept model does - a Dense-only network, nothing like the transformer."""
    inp = Input(shape=(seq_len, n_features))
    x = layers.Flatten()(inp)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)
    out = layers.Dense(1)(x)
    return Model(inp, out)


def run_trial(label, seed, train_dummy_first):
    print(f"\n=== {label} ===")
    set_seeds_custom(seed)
    tf.keras.backend.clear_session()

    X_train, X_test, y_train_raw, y_test_raw, y_train_z, y_test_z, target_scalers = build_single_split()

    if train_dummy_first:
        print("  Building + briefly training a dummy model first (consumes RNG draws, "
              "no re-seed before the real model - mirrors the notebook's cell structure)...")
        dummy = build_dummy_model(SEQUENCE_LENGTH, FEATURES_N)
        dummy.compile(optimizer="adam", loss="mse")
        dummy.fit(X_train, y_train_z["log_return"], epochs=3, batch_size=32, verbose=0)
        tf.keras.backend.clear_session()
        # NOTE: no set_seeds_custom() call here - matches the notebook not re-seeding
        # between the concept model and the transformer.
        # clear_session() resets the Keras graph/name state but NOT the RNG draw sequence.

    model = train_once_no_reseed(weighted_loss_fixed, X_train, y_train_z, X_test, y_test_z, FEATURES_N)
    acc, var = evaluate(model, X_test, y_test_raw, target_scalers)

    print(f"  uncal={var['uncalibrated']['coverage']*100:.2f}%  "
          f"cal={var['calibrated']['coverage']*100:.2f}%  p={var['calibrated']['kupiec_p']:.5g}  "
          f"verdict={'PASS' if var['calibrated']['kupiec_p'] >= 0.05 else 'FAIL'}")
    return var


def main():
    print(__doc__)
    results = {}

    # Known already, not re-run (already reproduced bit-for-bit twice):
    print("=== seed=42, clean (no dummy) [ALREADY ESTABLISHED, not re-run] ===")
    print("  uncal=48.20%  cal=2.76%  p=0.015156  verdict=FAIL")
    results["seed42_clean"] = {"coverage": 0.0276, "kupiec_p": 0.015156}

    results["seed43_clean"] = run_trial("seed=43, clean (no dummy)", 43, train_dummy_first=False)
    results["seed44_clean"] = run_trial("seed=44, clean (no dummy)", 44, train_dummy_first=False)
    results["seed42_dummy_first"] = run_trial("seed=42, dummy model trained first (no re-seed)", 42, train_dummy_first=True)

    print("\n" + "=" * 90)
    print("SUMMARY")
    print(f"{'trial':30s} {'cal cov':>10s} {'kupiec p':>12s} {'verdict':>8s}")
    print(f"{'seed=42, clean':30s} {'2.76%':>10s} {'0.015156':>12s} {'FAIL':>8s}")
    for name in ["seed43_clean", "seed44_clean", "seed42_dummy_first"]:
        r = results[name]
        v = "PASS" if r["calibrated"]["kupiec_p"] >= 0.05 else "FAIL"
        print(f"{name:30s} {r['calibrated']['coverage']*100:>9.2f}% "
              f"{r['calibrated']['kupiec_p']:>12.5g} {v:>8s}")
    print(f"\n(for reference: the notebook's own full pipeline - concept model + transformer, "
          f"seed=42 - gave cal=4.67%, p=0.740, PASS)")


if __name__ == "__main__":
    main()
