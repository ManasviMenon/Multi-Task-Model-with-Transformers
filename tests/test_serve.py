"""Tests for the FastAPI serving layer.

The load-bearing test here is `test_served_prediction_matches_offline_pipeline`: serving builds
features from a short request window, while training built them from 18 years of history, and
if those two paths disagree every prediction is quietly wrong. They did disagree at first -
`drawdown` is measured against a running all-time high, so a 150-bar window measured it against
the wrong peak and flipped the sign of the predicted return. That is what `reseat_drawdown`
fixes and what this test pins.

Tests needing the trained model skip when the registry isn't populated, so a fresh clone or CI
without mlruns/ still runs the validation tests.
"""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from mtl_var import registry
from mtl_var.data import FEATURES, SEQUENCE_LENGTH, TARGETS, load_and_engineer_features
from mtl_var.pipeline import FOLDS, build_fold_data
from mtl_var.serve import MIN_BARS, app, reseat_drawdown

CSV = "SPX500-18Y_scaled_y.csv"


def registry_ready():
    try:
        return registry.get_version("production") is not None
    except Exception:
        return False


needs_model = pytest.mark.skipif(not registry_ready(),
                                 reason="no model at @production - run: mtl-var register-model --alias production")


@pytest.fixture(scope="module")
def raw_bars():
    raw = pd.read_csv(CSV, parse_dates=["observation_date"], dayfirst=True)
    raw = raw.dropna(axis=1, how="all").drop(index=0).dropna(subset=["observation_date"])
    return raw


def to_payload(frame):
    return [{
        "date": str(pd.Timestamp(r["observation_date"]).date()),
        "open": float(r["Open"]), "high": float(r["High"]), "low": float(r["Low"]),
        "close": float(r["Close"]), "adj_close": float(r["Adj Close"]), "volume": float(r["Volume"]),
    } for _, r in frame.iterrows()]


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------- validation (no model needed)

def test_rejects_too_few_bars(client, raw_bars):
    body = {"bars": to_payload(raw_bars.tail(10))}
    r = client.post("/predict", json=body)
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
    payload[-1]["high"] = payload[-1]["low"] / 2  # high below low
    r = client.post("/predict", json={"bars": payload})
    assert r.status_code == 422


def test_rejects_nonpositive_volume(client, raw_bars):
    payload = to_payload(raw_bars.tail(MIN_BARS + 5))
    payload[-1]["volume"] = 0.0  # log_volume would be -inf
    r = client.post("/predict", json={"bars": payload})
    assert r.status_code == 422


# ---------------------------------------------------------------- drawdown reseating (pure)

def test_reseat_drawdown_uses_the_higher_of_window_and_prior_high():
    frame = pd.DataFrame({"Adj Close": [100.0, 110.0, 90.0]})

    seeded = reseat_drawdown(frame, prior_high=200.0)
    assert seeded["drawdown"].tolist() == pytest.approx([-0.5, -0.45, -0.55])

    # A prior high below the window's own peak must not lower the running max.
    unseeded = reseat_drawdown(frame, prior_high=1.0)
    assert unseeded["drawdown"].tolist() == pytest.approx([0.0, 0.0, 90 / 110 - 1])


def test_reseat_drawdown_reproduces_full_history_value():
    """The concrete regression: a 150-bar window ending 2022-12-30 must report the real
    drawdown (-0.1995), not the window-local one (-0.0856)."""
    from mtl_var.data import engineer_features

    full = load_and_engineer_features(CSV)
    raw = pd.read_csv(CSV, parse_dates=["observation_date"], dayfirst=True)
    raw = raw.dropna(axis=1, how="all").drop(index=0).dropna(subset=["observation_date"])

    windowed = engineer_features(raw.tail(150))
    assert windowed["drawdown"].iloc[-1] == pytest.approx(-0.0856, abs=1e-3)  # the bug

    prior_high = float(raw["Adj Close"].max())
    fixed = reseat_drawdown(windowed, prior_high)
    assert fixed["drawdown"].iloc[-1] == pytest.approx(full["drawdown"].iloc[-1], abs=1e-12)


# ---------------------------------------------------------------- serving (needs the model)

@needs_model
def test_health_reports_registry_lineage(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_name"] == registry.REGISTERED_MODEL_NAME
    assert body["model_alias"] == "production"
    assert body["trained_on_fold"]
    assert body["data_sha256"]
    assert body["calibration_offset"] is not None
    assert "4 of 5" in body["caveat"]  # the generalisation caveat must reach an operator


@needs_model
def test_predict_returns_all_five_heads_with_calibrated_var(client, raw_bars):
    body = client.post("/predict", json={"bars": to_payload(raw_bars.tail(150))}).json()

    p = body["prediction"]
    assert set(p) == {"log_return", "log_volume", "yz_vol", "var_5_raw", "var_5", "cvar_5"}
    assert body["as_of_date"] == "2022-12-30"
    assert body["model_version"] == "1"
    assert body["latency_ms"] > 0

    # the calibrated VaR is the raw head plus the registry-recorded offset
    assert p["var_5"] == pytest.approx(p["var_5_raw"] + body["calibration_offset"], abs=1e-12)
    assert p["var_5"] < 0  # a 5% VaR on daily returns must be a loss
    assert p["yz_vol"] > 0


@needs_model
def test_served_prediction_matches_offline_pipeline(client, raw_bars):
    """Serving from 150 bars must equal the full-history offline path, exactly."""
    from sklearn.preprocessing import StandardScaler

    served = client.post("/predict", json={"bars": to_payload(raw_bars.tail(150))}).json()

    data = load_and_engineer_features(CSV)
    fold_cfg = next(f for f in FOLDS if f["name"] == "fold5_2022_bear")
    fd = build_fold_data(fold_cfg, data)
    train_mask = ((data["observation_date"] >= fold_cfg["train_start"])
                  & (data["observation_date"] <= fold_cfg["train_end"]))
    scaler = StandardScaler().fit(data.loc[train_mask, FEATURES])

    window = data.iloc[-SEQUENCE_LENGTH:]
    X = scaler.transform(window[FEATURES])[np.newaxis, :, :].astype("float32")
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
def test_prior_high_override_changes_the_drawdown_basis(client, raw_bars):
    payload = to_payload(raw_bars.tail(150))
    default = client.post("/predict", json={"bars": payload}).json()
    override = client.post("/predict", json={"bars": payload, "prior_high": 9999.0}).json()

    assert default["prior_high_used"] == pytest.approx(4796.560059, abs=1e-3)
    assert override["prior_high_used"] == 9999.0
    assert override["prediction"]["var_5"] != default["prediction"]["var_5"]
