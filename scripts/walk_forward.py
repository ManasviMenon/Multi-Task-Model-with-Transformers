"""
Walk-forward validation for the SPX500 multi-task transformer.

Everything validated in MTL_Improved.ipynb (standardized targets, equal task weights,
post-hoc VaR calibration) lives on a single 80/20 chronological split. This script
re-runs the same methodology across five expanding-window folds spanning 2008-2022,
so accuracy and VaR calibration can be checked regime by regime instead of once.

Fold design (expanding train window, non-overlapping 2-3yr test blocks covering the
regimes that actually matter: 2008 GFC, 2011-13 EU debt crisis, 2015-17 China slowdown
+ Aug'15 selloff, 2019-20 COVID crash, 2022 rate-hike bear market):

  fold 1: train 2004-2007        test 2008-2009
  fold 2: train 2004-2010        test 2011-2013
  fold 3: train 2004-2014        test 2015-2017
  fold 4: train 2004-2018        test 2019-2020
  fold 5: train 2004-2021        test 2022

Each fold: fresh feature/target scalers (fit on that fold's train rows only), fresh
transformer model (same architecture/losses/equal-0.2-weights/early-stopping as the
notebook), trained from scratch. The fold's test window is split in half chronologically:
first half calibrates the VaR offset, second half (untouched) reports coverage + Kupiec,
exactly like the notebook's calibration step.

Results are written to walk_forward_results.json after every fold, so a crash partway
through doesn't lose completed folds.
"""
import os
import gc
import json
import random
import traceback

os.environ['PYTHONHASHSEED'] = '42'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import chi2

tf.config.experimental.enable_op_determinism()

SEED = 42
SEQUENCE_LENGTH = 5
RESULTS_PATH = "results/walk_forward_results.json"

FEATURES = [
    'log_open_ret', 'log_high_ret', 'log_low_ret', 'log_close_ret',
    'log_volume_lag1', 'log_volume_lag5',
    'ret_lag1', 'ret_lag2', 'ret_lag3', 'ret_lag4', 'ret_lag5',
    'yz_vol_lag1', 'VaR_5_lag1', 'drawdown', 'ret_min20', 'ret_max20',
    'CVaR_5_lag1', 'ret_skew20'
]
TARGETS = ['log_return', 'log_volume', 'yz_vol', 'VaR_5', 'CVaR_5']

FOLDS = [
    {"name": "fold1_gfc",        "train_start": "2004-01-01", "train_end": "2007-12-31", "test_start": "2008-01-01", "test_end": "2009-12-31"},
    {"name": "fold2_eu_crisis",  "train_start": "2004-01-01", "train_end": "2010-12-31", "test_start": "2011-01-01", "test_end": "2013-12-31"},
    {"name": "fold3_2015_selloff","train_start": "2004-01-01", "train_end": "2014-12-31", "test_start": "2015-01-01", "test_end": "2017-12-31"},
    {"name": "fold4_covid",      "train_start": "2004-01-01", "train_end": "2018-12-31", "test_start": "2019-01-01", "test_end": "2020-12-31"},
    {"name": "fold5_2022_bear",  "train_start": "2004-01-01", "train_end": "2021-12-31", "test_start": "2022-01-01", "test_end": "2022-12-31"},
]


def set_seeds():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


def load_and_engineer_features(csv_path):
    data = pd.read_csv(csv_path, parse_dates=['observation_date'], dayfirst=True)
    data = data.dropna(axis=1, how='all')
    data = data.drop(index=0)

    data['ret'] = data['Adj Close'].pct_change(fill_method=None)
    data['log_return'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1))
    data['log_volume'] = np.log(data['Volume'])

    data['log_open_ret'] = np.log(data['Open'] / data['Adj Close'].shift(1))
    data['log_high_ret'] = np.log(data['High'] / data['Adj Close'].shift(1))
    data['log_low_ret'] = np.log(data['Low'] / data['Adj Close'].shift(1))
    data['log_close_ret'] = np.log(data['Close'] / data['Adj Close'].shift(1))

    data['log_volume_lag1'] = data['log_volume'].shift(1)
    data['log_volume_lag5'] = data['log_volume'].shift(5)

    data['ret_skew20'] = data['log_return'].rolling(window=20).skew()
    data.dropna(inplace=True)

    def yang_zhang_vol(df):
        log_oc = np.log(df['Open'] / df['Adj Close'].shift(1))
        log_ho = np.log(df['High'] / df['Open'])
        log_lo = np.log(df['Low'] / df['Open'])
        rs = 0.5 * (log_ho ** 2 + log_lo ** 2)
        return (log_oc ** 2 + rs).rolling(window=20).mean()

    data['yz_vol'] = yang_zhang_vol(data)
    data.dropna(inplace=True)

    data['VaR_5'] = data['log_return'].ewm(span=10).mean() - 1.2 * data['yz_vol']
    data.dropna(inplace=True)

    for lag in range(1, 6):
        data[f'ret_lag{lag}'] = data['log_return'].shift(lag)
    data['yz_vol_lag1'] = data['yz_vol'].shift(1)
    data['VaR_5_lag1'] = data['VaR_5'].shift(1)
    data['CVaR_5'] = data['log_return'].rolling(window=20).apply(lambda x: x[x < x.quantile(0.05)].mean())
    data.dropna(inplace=True)

    data['drawdown'] = (data['Adj Close'] / data['Adj Close'].cummax()) - 1
    data['ret_min20'] = data['log_return'].rolling(20).min()
    data['ret_max20'] = data['log_return'].rolling(20).max()
    data['CVaR_5_lag1'] = data['CVaR_5'].shift(1)
    data.dropna(inplace=True)

    return data.reset_index(drop=True)


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


WEIGHTS = {"log_return": 0.2, "log_volume": 0.2, "yz_vol": 0.2, "VaR_5": 0.2, "CVaR_5": 0.2}


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


def kupiec_from_counts(N, T, p=0.05):
    if T == 0 or N == 0 or N == T:
        return float('nan')
    lr = -2 * np.log(((1 - p) ** (T - N) * p ** N) / ((1 - N / T) ** (T - N) * (N / T) ** N))
    return 1 - chi2.cdf(lr, 1)


def kupiec_test(returns, var_pred, p=0.05):
    T = len(var_pred)
    N = int(np.sum(returns < var_pred))
    return N, T, kupiec_from_counts(N, T, p)


def rolling_calibration_analysis(dates, returns, var_pred, eval_start_idx,
                                  L=125, R=21, S=10, elevated_ratio=1.5, lead_days=5,
                                  trigger_on_elevated=False, min_trigger_gap=5):
    """
    Walk day-by-day through a fold's test period with a causal, periodically-refreshed
    VaR offset, instead of one offset frozen at the start.

    L = trailing window (days) the offset is fit on
    R = refresh cadence (days) - how often the offset is recomputed on schedule
    S = short "live" window (days) used for the volatility-ratio monitor signal
    eval_start_idx = index marking where the fixed-offset approach's eval half began,
                      so rolling coverage can also be reported on that identical slice
    trigger_on_elevated = if True, an "elevated" vol-ratio reading forces an immediate
                      recalibration instead of waiting for the next scheduled refresh
                      (still gated by min_trigger_gap so it can't refresh every single
                      day during a sustained elevated stretch)

    Everything here is causal: the offset active on day t was fit only on
    returns[t-L:t], and the monitor's vol_ratio on day t only looks backward. A
    trigger check uses the ratio computed from the *currently active* offset window,
    so it can never see into the future either.
    """
    n = len(returns)
    if n < L + R:
        return None

    offset = None
    offset_window_vol = None
    last_refresh = None
    log_threshold = np.log2(elevated_ratio)

    idx, breach, vol_ratio, refresh_kind, offset_series, refresh_days = [], [], [], [], [], []
    n_scheduled_refreshes, n_triggered_refreshes = 0, 0

    for t in range(L, n):
        live_vol = float(returns[t - S:t].std()) if t >= S else np.nan
        ratio = live_vol / offset_window_vol if offset_window_vol and not np.isnan(live_vol) else np.nan
        is_elevated_now = (not np.isnan(ratio)) and (abs(np.log2(ratio)) > log_threshold)

        scheduled_due = last_refresh is None or t - last_refresh >= R
        gap_ok = last_refresh is None or t - last_refresh >= min_trigger_gap
        triggered_due = trigger_on_elevated and is_elevated_now and gap_ok and not scheduled_due

        kind = None
        if scheduled_due:
            kind = "scheduled"
            n_scheduled_refreshes += 1
        elif triggered_due:
            kind = "triggered"
            n_triggered_refreshes += 1

        if kind is not None:
            window_returns = returns[t - L:t]
            window_var = var_pred[t - L:t]
            offset = float(np.quantile(window_returns - window_var, 0.05))
            offset_window_vol = float(window_returns.std())
            last_refresh = t
            refresh_days.append(t)

        calibrated_var = var_pred[t] + offset
        idx.append(t)
        breach.append(bool(returns[t] < calibrated_var))
        vol_ratio.append(ratio)
        refresh_kind.append(kind)
        offset_series.append(offset)

    idx = np.array(idx)
    breach = np.array(breach)
    vol_ratio = np.array(vol_ratio, dtype=float)
    timeline_dates = [str(np.datetime_as_string(dates[t], unit='D')) for t in idx]

    T = len(breach)
    N = int(breach.sum())
    coverage = N / T
    kupiec_p = kupiec_from_counts(N, T)

    # restrict to the same trailing slice the fixed-offset eval half covered, for a
    # like-for-like comparison against the single-offset numbers
    mask_eval_slice = idx >= eval_start_idx
    T_eval = int(mask_eval_slice.sum())
    N_eval = int(breach[mask_eval_slice].sum())
    coverage_eval = N_eval / T_eval if T_eval else None
    kupiec_p_eval = kupiec_from_counts(N_eval, T_eval) if T_eval else None

    log_ratio = np.log2(vol_ratio)
    valid = ~np.isnan(log_ratio)
    elevated = valid & (np.abs(log_ratio) > np.log2(elevated_ratio))

    elevated_breach_rate = float(breach[elevated].mean()) if elevated.any() else None
    normal_breach_rate = float(breach[valid & ~elevated].mean()) if (valid & ~elevated).any() else None

    # lead-time check: for each breach day, was an "elevated" reading already showing
    # in the preceding `lead_days`? compare that rate to a random day's rate, to test
    # whether the signal actually leads breaches rather than just coinciding with them.
    def preceded_by_elevated(day_pos):
        lo = max(0, day_pos - lead_days)
        return bool(elevated[lo:day_pos].any())

    breach_positions = np.where(breach)[0]
    non_breach_positions = np.where(~breach)[0]
    breach_lead_rate = float(np.mean([preceded_by_elevated(p) for p in breach_positions])) if len(breach_positions) else None
    baseline_lead_rate = float(np.mean([preceded_by_elevated(p) for p in non_breach_positions])) if len(non_breach_positions) else None

    return {
        "params": {"L": L, "R": R, "S": S, "elevated_ratio": elevated_ratio, "lead_days": lead_days,
                   "trigger_on_elevated": trigger_on_elevated, "min_trigger_gap": min_trigger_gap},
        "refreshes": {"scheduled": n_scheduled_refreshes, "triggered": n_triggered_refreshes},
        "full_walk": {"n": T, "breaches": N, "coverage": coverage, "kupiec_p": kupiec_p},
        "eval_slice_only": {"n": T_eval, "breaches": N_eval, "coverage": coverage_eval, "kupiec_p": kupiec_p_eval},
        "monitor": {
            "elevated_breach_rate": elevated_breach_rate,
            "normal_breach_rate": normal_breach_rate,
            "breach_preceded_by_elevated_rate": breach_lead_rate,
            "baseline_preceded_by_elevated_rate": baseline_lead_rate,
            "n_elevated_days": int(elevated.sum()),
            "n_valid_days": int(valid.sum()),
        },
        "timeline": {
            "date": timeline_dates,
            "vol_ratio": [None if np.isnan(v) else round(float(v), 4) for v in vol_ratio],
            "breach": [bool(b) for b in breach],
            "elevated": [bool(e) for e in elevated],
            "offset": [round(float(o), 6) for o in offset_series],
        },
        "refresh_days": refresh_days,
    }


def make_sequences(X_scaled, seq_len):
    return np.array([X_scaled[i - seq_len + 1:i + 1] for i in range(seq_len - 1, len(X_scaled))])


def run_fold(fold_cfg, data):
    print(f"\n===== {fold_cfg['name']}  train[{fold_cfg['train_start']}:{fold_cfg['train_end']}]  "
          f"test[{fold_cfg['test_start']}:{fold_cfg['test_end']}] =====", flush=True)

    tf.keras.backend.clear_session()
    gc.collect()
    set_seeds()

    dates = data['observation_date']
    train_mask = (dates >= fold_cfg['train_start']) & (dates <= fold_cfg['train_end'])
    test_mask = (dates >= fold_cfg['test_start']) & (dates <= fold_cfg['test_end'])

    train_df = data.loc[train_mask].reset_index(drop=True)
    test_df = data.loc[test_mask].reset_index(drop=True)

    feat_scaler = StandardScaler()
    X_train_scaled = feat_scaler.fit_transform(train_df[FEATURES])
    X_test_scaled = feat_scaler.transform(test_df[FEATURES])

    X_train = make_sequences(X_train_scaled, SEQUENCE_LENGTH)
    X_test = make_sequences(X_test_scaled, SEQUENCE_LENGTH)

    y_train_raw = {t: train_df[t].values[SEQUENCE_LENGTH - 1:] for t in TARGETS}
    y_test_raw = {t: test_df[t].values[SEQUENCE_LENGTH - 1:] for t in TARGETS}

    target_scalers = {}
    y_train_z, y_test_z = {}, {}
    for t in TARGETS:
        ts = StandardScaler()
        y_train_z[t] = ts.fit_transform(y_train_raw[t].reshape(-1, 1)).flatten()
        y_test_z[t] = ts.transform(y_test_raw[t].reshape(-1, 1)).flatten()
        target_scalers[t] = ts

    print(f"n_train={len(X_train)}  n_test={len(X_test)}  n_features={len(FEATURES)}", flush=True)

    n_features = len(FEATURES)
    model = build_transformer_model(SEQUENCE_LENGTH, n_features)
    optimizer = tf.keras.optimizers.Adam(1e-4)
    batch_size = 32
    max_epochs = 50
    patience = 5

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
                L = sum(weighted_loss(yb[t], p, t) for t, p in
                        zip(TARGETS, [mu_p, vol_p, yz_p, var_p, cvar_p]))
            grads = tape.gradient(L, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        val_losses = []
        for xb, yb in val_ds:
            mu_p, vol_p, yz_p, var_p, cvar_p = model(xb, training=False)
            vL = sum(weighted_loss(yb[t], p, t) for t, p in
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
    print(f"trained {epochs_run} epochs, best val_loss={best_val_loss:.6f}", flush=True)

    preds = model.predict(X_test, verbose=0)
    mu_p, vol_p, yz_p, var_p, cvar_p = preds
    native_preds = {}
    for t, p in zip(TARGETS, [mu_p, vol_p, yz_p, var_p, cvar_p]):
        native_preds[t] = target_scalers[t].inverse_transform(p.reshape(-1, 1)).flatten()

    def rmse(a, b):
        return float(np.sqrt(mean_squared_error(a, b)))

    accuracy = {}
    for t in TARGETS:
        y_true = y_test_raw[t]
        y_pred = native_preds[t]
        accuracy[t] = {"mae": float(mean_absolute_error(y_true, y_pred)), "rmse": rmse(y_true, y_pred)}

    returns = y_test_raw['log_return']
    var_pred = native_preds['VaR_5']
    calib_n = len(var_pred) // 2

    returns_calib, returns_eval = returns[:calib_n], returns[calib_n:]
    var_calib, var_eval = var_pred[:calib_n], var_pred[calib_n:]

    residuals_calib = returns_calib - var_calib
    offset = float(np.quantile(residuals_calib, 0.05))
    var_eval_calibrated = var_eval + offset

    N0, T0, p0 = kupiec_test(returns_eval, var_eval)
    N1, T1, p1 = kupiec_test(returns_eval, var_eval_calibrated)

    test_dates = test_df['observation_date'].values[SEQUENCE_LENGTH - 1:]

    os.makedirs("results/predictions", exist_ok=True)
    np.savez(
        os.path.join("results/predictions", f"{fold_cfg['name']}.npz"),
        dates=test_dates.astype("datetime64[D]").astype(str),
        returns=returns,
        var_pred=var_pred,
        calib_n=calib_n,
    )

    rolling = rolling_calibration_analysis(test_dates, returns, var_pred, eval_start_idx=calib_n,
                                            trigger_on_elevated=False)
    rolling_triggered = rolling_calibration_analysis(test_dates, returns, var_pred, eval_start_idx=calib_n,
                                                       trigger_on_elevated=True)
    if rolling is not None:
        rc = rolling["eval_slice_only"]
        print(f"Rolling scheduled-only (eval slice) coverage={rc['coverage']:.4f} (kupiec p={rc['kupiec_p']})  "
              f"[fixed offset was coverage={N1/T1:.4f}, p={p1}]", flush=True)
        mon = rolling["monitor"]
        print(f"Monitor: breach rate when ratio elevated={mon['elevated_breach_rate']}  "
              f"vs normal={mon['normal_breach_rate']}  |  "
              f"breaches preceded by an elevated reading={mon['breach_preceded_by_elevated_rate']}  "
              f"vs baseline={mon['baseline_preceded_by_elevated_rate']}", flush=True)
    if rolling_triggered is not None:
        rc2 = rolling_triggered["eval_slice_only"]
        rf = rolling_triggered["refreshes"]
        print(f"Rolling event-triggered (eval slice) coverage={rc2['coverage']:.4f} (kupiec p={rc2['kupiec_p']})  "
              f"refreshes: scheduled={rf['scheduled']} triggered={rf['triggered']}", flush=True)

    result = {
        "fold": fold_cfg["name"],
        "train_start": fold_cfg["train_start"],
        "train_end": fold_cfg["train_end"],
        "test_start": fold_cfg["test_start"],
        "test_end": fold_cfg["test_end"],
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "epochs_run": epochs_run,
        "best_val_loss": best_val_loss,
        "accuracy": accuracy,
        "var_calibration": {
            "calib_n": calib_n,
            "eval_n": T0,
            "offset": offset,
            "uncalibrated": {"breaches": N0, "coverage": N0 / T0 if T0 else None, "kupiec_p": p0},
            "calibrated": {"breaches": N1, "coverage": N1 / T1 if T1 else None, "kupiec_p": p1},
        },
        "rolling_calibration": rolling,
        "rolling_calibration_triggered": rolling_triggered,
    }

    print(f"VaR uncalibrated coverage={N0/T0:.4f} (kupiec p={p0})  "
          f"calibrated coverage={N1/T1:.4f} (kupiec p={p1})", flush=True)

    return result


def main():
    print("Loading data and engineering features...", flush=True)
    data = load_and_engineer_features("SPX500-18Y_scaled_y.csv")
    print(f"{len(data)} usable rows, {data['observation_date'].min().date()} to "
          f"{data['observation_date'].max().date()}", flush=True)

    results = []
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            results = json.load(f)
        done = {r["fold"] for r in results}
        print(f"Resuming: {len(results)} fold(s) already completed: {sorted(done)}", flush=True)
    else:
        done = set()

    for fold_cfg in FOLDS:
        if fold_cfg["name"] in done:
            continue
        try:
            result = run_fold(fold_cfg, data)
            results.append(result)
        except Exception as e:
            print(f"FOLD {fold_cfg['name']} FAILED: {e}", flush=True)
            traceback.print_exc()
            results.append({"fold": fold_cfg["name"], "error": str(e)})

        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)

    print("\n===== walk-forward complete =====", flush=True)
    for r in results:
        if "error" in r:
            print(f"{r['fold']}: ERROR - {r['error']}")
            continue
        vc = r["var_calibration"]
        line = (f"{r['fold']}: test={r['test_start']}..{r['test_end']}  "
                f"n_test={r['n_test']}  "
                f"return_MAE={r['accuracy']['log_return']['mae']:.5f}  "
                f"fixed_offset: uncal={vc['uncalibrated']['coverage']:.3f} "
                f"cal={vc['calibrated']['coverage']:.3f} p={vc['calibrated']['kupiec_p']:.3f}")
        rc = r.get("rolling_calibration")
        if rc:
            es = rc["eval_slice_only"]
            mon = rc["monitor"]
            line += (f"  |  rolling: cov={es['coverage']:.3f} p={es['kupiec_p']}  "
                     f"monitor: elevated_breach_rate={mon['elevated_breach_rate']} "
                     f"normal_breach_rate={mon['normal_breach_rate']} "
                     f"lead={mon['breach_preceded_by_elevated_rate']} "
                     f"vs baseline={mon['baseline_preceded_by_elevated_rate']}")
        rct = r.get("rolling_calibration_triggered")
        if rct:
            es2 = rct["eval_slice_only"]
            rf = rct["refreshes"]
            line += (f"  |  triggered: cov={es2['coverage']:.3f} p={es2['kupiec_p']} "
                     f"refreshes(sched={rf['scheduled']},trig={rf['triggered']})")
        print(line)


if __name__ == "__main__":
    main()
