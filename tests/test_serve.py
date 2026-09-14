"""Tests for the FastAPI serving layer.

Two equivalences carry the weight here, and both were real bugs before they were tests:

  - Serving builds features from a short request window; training built them from 18 years of
    history. `drawdown` is measured against a running all-time high, so a 150-bar window
    measured it against the wrong peak and flipped the sign of the predicted return.
    `reseat_drawdown` fixes that.

  - The served model is a one-step-ahead forecaster (leakage-free features, RESULTS.md section
    13). A forecast for the day after the last bar is built from a placeholder bar, so the
    placeholder's prices must never reach the features. The forecast row is tested
    bit-identical to the row built from the real next bar, and the served forecast is tested
    against the offline pipeline's prediction for that day.

Tests needing the trained model skip unless a leakage-free version is at @production, so a fresh
clone or CI without mlruns/ still runs the validation and feature tests.
"""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from mtl_var import registry
from mtl_var.data import (
    LEAKAGE_FREE_FEATURES,
    SEQUENCE_LENGTH,
    TARGETS,
    build_leakage_free_frame,
    engineer_features,
    frame_for_feature_set,
    load_and_engineer_features,
)
from mtl_var.pipeline import FOLDS, build_fold_data
from mtl_var.serve import MIN_BARS, app, build_forecast_frame, check_servable, reseat_drawdown

CSV = "SPX500-18Y_scaled_y.csv"


def servable_model_ready():
    try:
        v = registry.get_version("production")
        return v is not None and (v.tags or {}).get("feature_set") == "leakage_free"
    except Exception:
        return False


needs_model = pytest.mark.skipif(
    not servable_model_ready(),
    reason="no leakage-free model at @production - run: mtl-var register-model --alias production",
)


@pytest.fixture(scope="module")
def raw_bars():
    raw = pd.read_csv(CSV, parse_dates=["observation_date"], dayfirst=True)
    raw = raw.dropna(axis=1, how="all").drop(index=0).dropna(subset=["observation_date"])
    return raw.reset_index(drop=True)


@pytest.fixture(scope="module")
def history():
    return load_and_engineer_features(CSV)


def to_payload(frame):
    return [{
        "date": str(pd.Timestamp(r["observation_date"]).date()),
        "open": float(r["Open"]), "high": float(r["High"]), "low": float(r["Low"]),
        "close": float(r["Close"]), "adj_close": float(r["Adj Close"]), "volume": float(r["Volume"]),
    } for _, r in frame.iterrows()]


def prior_high_before(history, first_date):
    return float(history.loc[history["observation_date"] < pd.Timestamp(first_date), "Adj Close"].max())


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------- validation (no model needed)

def test_rejects_too_few_bars(client, raw_bars):
    r = client.post("/predict", json={"bars": to_payload(raw_bars.tail(10))})
    assert r.status_code == 422
    assert f"at least {MIN_BARS} bars" in r.text


def test_rejects_out_of_order_bars(client, raw_bars):
    payload = to_payload(raw_bars.tail(MIN_BARS + 5))
    payload[0], payload[-1] = payload[-1], payload[0]
    r = client.post("/predict", json={"bars": payload})
    assert r.status_code == 422
    assert "chronological" in r.text


def test_rejects_duplicate_dates(client, raw_bars):
    payload = to_payload(raw_bars.tail(MIN_BARS + 5))
    payload[-1]["date"] = payload[-2]["date"]
    r = client.post("/predict", json={"bars": payload})
    assert r.status_code == 422
    assert "duplicate" in r.text


def test_rejects_inconsistent_ohlc(client, raw_bars):
    payload = to_payload(raw_bars.tail(MIN_BARS + 5))
    payload[-1]["high"] = payload[-1]["low"] / 2
    assert client.post("/predict", json={"bars": payload}).status_code == 422


def test_rejects_nonpositive_volume(client, raw_bars):
    payload = to_payload(raw_bars.tail(MIN_BARS + 5))
    payload[-1]["volume"] = 0.0  # log_volume would be -inf
    assert client.post("/predict", json={"bars": payload}).status_code == 422


# ---------------------------------------------------------------- pure feature logic

def test_refuses_to_serve_a_leaky_model():
    for tags in ({}, {"feature_set": "original"}):  # untagged versions predate feature sets
        with pytest.raises(RuntimeError, match="target leakage"):
            check_servable(tags, version="1")
    assert check_servable({"feature_set": "leakage_free"}) == "leakage_free"


def test_reseat_drawdown_uses_the_higher_of_window_and_prior_high():
    frame = pd.DataFrame({"Adj Close": [100.0, 110.0, 90.0]})
    assert reseat_drawdown(frame, 200.0)["drawdown"].tolist() == pytest.approx([-0.5, -0.45, -0.55])
    assert reseat_drawdown(frame, 1.0)["drawdown"].tolist() == pytest.approx([0.0, 0.0, 90 / 110 - 1])


def test_reseat_drawdown_reproduces_full_history_value(raw_bars, history):
    """The regression: a 150-bar window ending 2022-12-30 must report the real drawdown
    (-0.1995), not the window-local one (-0.0856)."""
    window = raw_bars.tail(150)
    windowed = engineer_features(window)
    assert windowed["drawdown"].iloc[-1] == pytest.approx(-0.0856, abs=1e-3)  # the bug

    fixed = reseat_drawdown(windowed, prior_high_before(history, window["observation_date"].iloc[0]))
    assert fixed["drawdown"].iloc[-1] == pytest.approx(history["drawdown"].iloc[-1], abs=1e-12)


def test_forecast_row_equals_the_real_next_days_row(raw_bars, history):
    """The placeholder bar must be invisible. Build the forecast row for 2022-12-30 from bars
    through 2022-12-29, then build that row again with the real 12-30 bar - they must match
    bit for bit, even though the placeholder's prices are 12-29's."""
    bars = raw_bars.iloc[:-1].tail(150).reset_index(drop=True)
    real_next = raw_bars.iloc[[-1]]
    assert real_next["Adj Close"].iloc[0] != bars["Adj Close"].iloc[-1]  # placeholder really differs

    prior_high = prior_high_before(history, bars["observation_date"].iloc[0])
    forecast = build_forecast_frame(bars, prior_high, "leakage_free")

    truth_frame = pd.concat([bars, real_next], ignore_index=True)
    truth = frame_for_feature_set(reseat_drawdown(engineer_features(truth_frame), prior_high), "leakage_free")

    np.testing.assert_array_equal(forecast[LEAKAGE_FREE_FEATURES].to_numpy()[-SEQUENCE_LENGTH:],
                                  truth[LEAKAGE_FREE_FEATURES].to_numpy()[-SEQUENCE_LENGTH:])


# ---------------------------------------------------------------- serving (needs the model)

@needs_model
def test_health_reports_registry_lineage(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_name"] == registry.REGISTERED_MODEL_NAME
    assert body["model_alias"] == "production"
    assert body["feature_set"] == "leakage_free"
    assert body["leakage_status"] == "corrected"
    assert body["data_sha256"]
    assert body["calibration_offset"] is not None
    assert "4 of 5" in body["caveat"]
    assert "no forecasting skill" in body["caveat"]


@needs_model
def test_predict_returns_a_calibrated_next_day_forecast(client, raw_bars):
    body = client.post("/predict", json={"bars": to_payload(raw_bars.tail(150))}).json()

    p = body["prediction"]
    assert set(p) == {"log_return", "log_volume", "yz_vol", "var_5_raw", "var_5", "cvar_5"}
    assert body["as_of_date"] == "2022-12-30"
    assert body["forecast_for"] == "next trading day after 2022-12-30"
    assert body["feature_set"] == "leakage_free"
    assert body["model_version"] == str(registry.get_version("production").version)
    assert body["latency_ms"] > 0

    assert p["var_5"] == pytest.approx(p["var_5_raw"] + body["calibration_offset"], abs=1e-12)
    assert p["var_5"] < 0
    assert p["yz_vol"] > 0


@needs_model
def test_served_forecast_matches_offline_pipeline(client, raw_bars, history):
    """A forecast for 2022-12-30 made from bars through 12-29 must equal the offline pipeline's
    prediction for that row, made with the full history."""
    bars = raw_bars.iloc[:-1].tail(150)
    served = client.post("/predict", json={"bars": to_payload(bars)}).json()
    assert served["as_of_date"] == "2022-12-29"

    clean = build_leakage_free_frame(history)
    fold_cfg = next(f for f in FOLDS if f["name"] == "fold5_2022_bear")
    fd = build_fold_data(fold_cfg, clean, LEAKAGE_FREE_FEATURES)

    window = clean.iloc[-SEQUENCE_LENGTH:]
    assert str(window["observation_date"].iloc[-1].date()) == "2022-12-30"
    X = fd["feat_scaler"].transform(window[LEAKAGE_FREE_FEATURES])[np.newaxis, :, :].astype("float32")
    preds = registry.load("production").predict(X, verbose=0)
    native = {t: float(fd["target_scalers"][t].inverse_transform(np.asarray(p).reshape(-1, 1))[0, 0])
              for t, p in zip(TARGETS, preds)}

    p = served["prediction"]
    assert p["log_return"] == pytest.approx(native["log_return"], abs=1e-9)
    assert p["log_volume"] == pytest.approx(native["log_volume"], abs=1e-9)
    assert p["yz_vol"] == pytest.approx(native["yz_vol"], abs=1e-12)
    assert p["var_5_raw"] == pytest.approx(native["VaR_5"], abs=1e-12)
    assert p["cvar_5"] == pytest.approx(native["CVaR_5"], abs=1e-12)


@needs_model
def test_prior_high_defaults_to_history_before_the_first_bar(client, raw_bars, history):
    late = client.post("/predict", json={"bars": to_payload(raw_bars.tail(150))}).json()
    assert late["prior_high_used"] == pytest.approx(4796.560059, abs=1e-3)
    assert late["prior_high_source"] == "service history before first bar"

    # A replay of 2020 must not be measured against a peak from 2022.
    covid = raw_bars[raw_bars["observation_date"] <= "2020-03-20"].tail(150)
    early = client.post("/predict", json={"bars": to_payload(covid)}).json()
    expected = prior_high_before(history, covid["observation_date"].iloc[0])
    assert early["prior_high_used"] == pytest.approx(expected, abs=1e-9)
    assert early["prior_high_used"] < 4000

    override = client.post("/predict", json={"bars": to_payload(raw_bars.tail(150)), "prior_high": 9999.0}).json()
    assert override["prior_high_source"] == "request"
    assert override["prediction"]["var_5"] != late["prediction"]["var_5"]


@needs_model
def test_outcome_scores_a_forecast_and_metrics_report_it(client, raw_bars):
    body = client.post("/predict", json={"bars": to_payload(raw_bars.iloc[:-1].tail(150))}).json()
    realised = float(np.log(raw_bars["Adj Close"].iloc[-1] / raw_bars["Adj Close"].iloc[-2]))

    scored = client.post("/outcome", json={"as_of_date": body["as_of_date"], "realised_return": realised}).json()
    assert scored["matched"] is True
    assert scored["breached"] == (realised < body["prediction"]["var_5"])

    unmatched = client.post("/outcome", json={"as_of_date": "1999-01-01", "realised_return": 0.0}).json()
    assert unmatched["matched"] is False

    m = client.get("/metrics").json()
    assert m["model"]["feature_set"] == "leakage_free"
    assert m["performance_drift"]["n_scored"] >= 1
    assert m["latency"]["n"] >= 1
    assert {"volatility_regime", "feature_drift", "alerts"} <= set(m)
