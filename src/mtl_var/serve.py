"""FastAPI serving layer: POST /predict, POST /outcome, GET /health, GET /metrics.

Loads the registry-pinned model once at startup (`models:/mtl-var-transformer@production`),
never a loose file, so what is served is always a version with recorded lineage.

What the raw model cannot do on its own, which this layer supplies:

  1. **Features.** The network takes 18 engineered, standardised features over a 5-day window,
     not raw OHLCV. Requests carry raw bars and go through `data.engineer_features` - the same
     function training used, not a reimplementation. The registered version's `feature_set` tag
     decides which features are built, and only the leakage-free set is servable: the original
     set contains the `log_return` target as an input (RESULTS.md section 13), and startup
     refuses a version trained on it.

  2. **Scalers.** The StandardScalers were fitted on the registered model's training fold and
     were never persisted. They are refitted at startup from the same fold window, which is
     deterministic and therefore reproduces the training-time scaling exactly. The cost is
     that serving needs the training csv on disk; startup verifies its SHA-256 against the
     hash recorded on the registered version and refuses to serve on a mismatch.

  3. **The calibration offset.** Read from the registered version's `var_offset` tag. The raw
     VaR head is uncalibrated and not servable without it.

  4. **An honest forecast horizon.** With leakage-free features every value in row t is known
     before day t, so the model is a one-step-ahead forecaster. A request carries bars through
     day D and gets a forecast for the next trading day. Row D+1's features depend only on bars
     up to D, so that row is built by appending a placeholder bar whose prices cannot reach the
     features - a test holds the forecast row bit-identical to the one built from the real next
     bar.

The caveats, repeated from the registry description and surfaced on every response: the VaR
calibration does not generalise across regimes (4 of 5 walk-forward folds fail Kupiec), and the
return head has no forecasting skill (it loses to a zero forecast on all 5 folds).
"""
import logging
import os
import time
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator, model_validator

from . import registry
from .data import (
    DEFAULT_CSV,
    FEATURE_SETS,
    SEQUENCE_LENGTH,
    TARGETS,
    engineer_features,
    frame_for_feature_set,
    load_and_engineer_features,
)
from .monitoring import PREDICTION_LOG, MonitoringState
from .pipeline import FOLDS, build_fold_data

logger = logging.getLogger("mtl_var.serve")

SERVING_FOLD = os.environ.get("MTL_VAR_SERVING_FOLD", "fold5_2022_bear")
SERVING_ALIAS = os.environ.get("MTL_VAR_SERVING_ALIAS", registry.PRODUCTION_ALIAS)
CSV_PATH = os.environ.get("MTL_VAR_CSV", DEFAULT_CSV)

# Only feature sets with no day-t information are servable. See RESULTS.md section 13.
SERVABLE_FEATURE_SETS = {"leakage_free"}

# Rolling(20) terms plus the lag chain consume ~78 leading rows before the first usable feature
# row; the leakage-free lag costs one more and the 5-day window needs 4 on top.
MIN_BARS = 90

STATE = {}

CAVEAT = ("Leakage-free one-step-ahead model (RESULTS.md 13). VaR calibration is fitted on this model's "
          "own fold and does not generalise across regimes: 4 of 5 walk-forward folds fail the Kupiec "
          "test. That fold-level verdict is unchanged across this model, the original-feature model and "
          "a model-free backtest of the raw VaR target. The return head has no forecasting skill - it "
          "loses to a constant zero forecast on all 5 folds.")


class OHLCVBar(BaseModel):
    """One daily bar. Field names mirror the training csv's columns."""

    date: str = Field(..., description="trading date, YYYY-MM-DD")
    open: float = Field(..., gt=0)
    high: float = Field(..., gt=0)
    low: float = Field(..., gt=0)
    close: float = Field(..., gt=0)
    adj_close: float = Field(..., gt=0)
    volume: float = Field(..., gt=0, description="must be > 0; log_volume is a model feature")

    @model_validator(mode="after")
    def check_ohlc_consistency(self):
        if self.high < self.low:
            raise ValueError(f"high ({self.high}) < low ({self.low}) on {self.date}")
        if not (self.low <= self.open <= self.high):
            raise ValueError(f"open ({self.open}) outside [low, high] on {self.date}")
        if not (self.low <= self.close <= self.high):
            raise ValueError(f"close ({self.close}) outside [low, high] on {self.date}")
        return self


class PredictRequest(BaseModel):
    bars: list[OHLCVBar] = Field(..., description=f"chronological daily bars, oldest first, >= {MIN_BARS}")
    prior_high: float | None = Field(
        None, gt=0,
        description=("Highest Adj Close BEFORE the first bar. The drawdown feature is measured against "
                     "a running all-time high, which a short request window may not contain. If omitted, "
                     "the service uses the highest close in its own history dated before the first bar, "
                     "which is exactly what training saw."),
    )

    @field_validator("bars")
    @classmethod
    def check_history(cls, bars):
        if len(bars) < MIN_BARS:
            raise ValueError(
                f"need at least {MIN_BARS} bars: the rolling-20 features and lag chain consume "
                f"~78 leading rows before the first usable row, plus a {SEQUENCE_LENGTH}-day window. "
                f"Got {len(bars)}."
            )
        dates = [b.date for b in bars]
        if dates != sorted(dates):
            raise ValueError("bars must be in chronological order, oldest first")
        if len(set(dates)) != len(dates):
            raise ValueError("duplicate dates in bars")
        return bars


class Prediction(BaseModel):
    log_return: float
    log_volume: float
    yz_vol: float
    var_5_raw: float = Field(..., description="uncalibrated VaR head output")
    var_5: float = Field(..., description="post-hoc calibrated 5% VaR - the number to use")
    cvar_5: float


class PredictResponse(BaseModel):
    as_of_date: str = Field(..., description="date of the last bar sent - the latest information used")
    forecast_for: str = Field(..., description="the day this forecast is for")
    prediction: Prediction
    calibration_offset: float
    prior_high_used: float = Field(..., description="running-high basis used for the drawdown feature")
    prior_high_source: str
    feature_set: str
    model_name: str
    model_version: str
    model_alias: str
    latency_ms: float
    caveat: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str | None = None
    model_version: str | None = None
    model_alias: str | None = None
    trained_on_fold: str | None = None
    seed: str | None = None
    feature_set: str | None = None
    leakage_status: str | None = None
    forecast_horizon: str | None = None
    git_commit: str | None = None
    data_sha256: str | None = None
    calibration_offset: float | None = None
    min_bars_required: int = MIN_BARS
    caveat: str | None = None


def check_servable(tags, version="?"):
    """Return the version's feature set, or refuse to serve it.

    Versions registered before feature sets existed carry no tag; they were trained on the
    original set, so a missing tag is treated as "original" and refused.
    """
    feature_set = (tags or {}).get("feature_set", "original")
    if feature_set not in SERVABLE_FEATURE_SETS:
        raise RuntimeError(
            f"refusing to serve {registry.REGISTERED_MODEL_NAME} version {version}: it was trained on the "
            f"{feature_set!r} feature set, which contains target leakage - log_close_ret is identical to "
            f"the log_return target (RESULTS.md section 13). Register a leakage-free model and point "
            f"@{SERVING_ALIAS} at it."
        )
    return feature_set


def _load_serving_state():
    """Load model, scalers, calibration offset and monitoring. Raises if anything is unservable."""
    version = registry.get_version(SERVING_ALIAS)
    if version is None:
        raise RuntimeError(
            f"no model registered at alias @{SERVING_ALIAS}. "
            f"Run: mtl-var register-model --alias {SERVING_ALIAS}"
        )

    tags = version.tags or {}
    feature_set = check_servable(tags, version.version)
    feature_names = FEATURE_SETS[feature_set]

    expected_hash = tags.get("data_sha256")
    actual_hash = registry.data_fingerprint(CSV_PATH)["data_sha256"]
    if expected_hash and expected_hash != actual_hash:
        raise RuntimeError(
            f"training data mismatch: registered version {version.version} was built from data "
            f"with sha256 {expected_hash}, but {CSV_PATH} hashes to {actual_hash}. Refusing to "
            f"serve - the scalers refitted from this file would not match the model."
        )

    if "var_offset" not in tags:
        raise RuntimeError(
            f"registered version {version.version} has no var_offset tag; the raw VaR head is "
            f"uncalibrated and cannot be served. Re-register with metrics attached."
        )
    offset = float(tags["var_offset"])

    fold_cfg = next((f for f in FOLDS if f["name"] == tags.get("fold", SERVING_FOLD)), None)
    if fold_cfg is None:
        raise RuntimeError(f"unknown fold {tags.get('fold')} on registered version")

    # Refit the fold's scalers through the same function training used. Deterministic - no RNG
    # involved and nothing is trained here.
    engineered = load_and_engineer_features(CSV_PATH)
    data = frame_for_feature_set(engineered, feature_set)
    fd = build_fold_data(fold_cfg, data, feature_names)
    feat_scaler = fd["feat_scaler"]

    model = registry.load(SERVING_ALIAS)

    # The volatility the active offset was fitted against - the denominator of the section 5
    # monitor signal. The fixed offset is fitted on the first half of the fold's test window.
    fold_returns = fd["y_test_raw"]["log_return"]
    calibration_window_vol = float(np.std(fold_returns[: len(fold_returns) // 2]))

    monitoring = MonitoringState(
        feature_names=feature_names,
        train_means=feat_scaler.mean_,
        train_stds=feat_scaler.scale_,
        calibration_window_vol=calibration_window_vol,
        prediction_log=os.environ.get("MTL_VAR_PREDICTION_LOG", PREDICTION_LOG),
    )

    logger.info("loaded %s version %s (@%s), fold=%s, feature_set=%s, offset=%.10g",
                registry.REGISTERED_MODEL_NAME, version.version, SERVING_ALIAS,
                fold_cfg["name"], feature_set, offset)

    return {
        "model": model,
        "version": version,
        "tags": tags,
        "offset": offset,
        "feature_set": feature_set,
        "feature_names": feature_names,
        "feat_scaler": feat_scaler,
        "target_scalers": fd["target_scalers"],
        "fold": fold_cfg["name"],
        # Closes the service has seen, for the default drawdown basis - see default_prior_high.
        # Taken from the engineered frame because that is where training's cummax started.
        "history_dates": engineered["observation_date"].to_numpy(dtype="datetime64[ns]"),
        "history_closes": engineered["Adj Close"].to_numpy(dtype=float),
        "monitoring": monitoring,
        "calibration_window_vol": calibration_window_vol,
    }


def reseat_drawdown(engineered, prior_high):
    """Recompute `drawdown` against a running high seeded with history before the window.

    Every other feature converges once the rolling-20 terms and lag chain warm up - verified by
    diffing a windowed frame against the full-history one. `drawdown` does not: it is
    close / cummax(close) - 1, so a request window that doesn't contain the true prior peak
    measures against a lower high. On a 150-bar window ending 2022-12-30 that gave -0.0856
    instead of -0.1995, enough to flip the sign of the predicted return.
    """
    engineered = engineered.copy()
    closes = engineered["Adj Close"].to_numpy(dtype=float)
    running = np.maximum(np.maximum.accumulate(closes), float(prior_high))
    engineered["drawdown"] = closes / running - 1.0
    return engineered


def default_prior_high(first_bar_date):
    """Highest close in the service's history strictly before the first bar, or 0 if none.

    An earlier default used the whole dataset's maximum, which is only right for requests that
    continue on from the end of the data. Replaying 2020 would have measured drawdown against a
    2022 peak that had not happened yet.
    """
    mask = STATE["history_dates"] < np.datetime64(pd.Timestamp(first_bar_date))
    return float(STATE["history_closes"][mask].max()) if mask.any() else 0.0


def build_forecast_frame(raw, prior_high, feature_set):
    """Feature frame whose last row is the forecast row for the day after the last bar.

    A leakage-free row depends only on earlier bars, so the next day's row can be built before
    that day exists: append a placeholder bar (a copy of the last one), engineer, and the
    placeholder's row holds the next day's features. The placeholder's own prices only reach
    columns the leakage-free set never reads.
    """
    placeholder = raw.iloc[[-1]].copy()
    placeholder["observation_date"] = placeholder["observation_date"] + pd.Timedelta(days=1)
    frame = pd.concat([raw, placeholder], ignore_index=True)
    engineered = reseat_drawdown(engineer_features(frame), prior_high)
    return frame_for_feature_set(engineered, feature_set)


def _attach_to_uvicorn_logging():
    """Route this module's logs through uvicorn's handlers.

    Without this the per-request latency lines go nowhere: uvicorn configures its own loggers
    and leaves other module loggers handler-less, so logger.info() is silently dropped.
    """
    uvicorn_logger = logging.getLogger("uvicorn")
    if uvicorn_logger.handlers:
        logger.handlers = uvicorn_logger.handlers
        logger.setLevel(uvicorn_logger.level or logging.INFO)
    elif not logger.handlers:
        logging.basicConfig(level=logging.INFO)
    logger.propagate = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once, at startup - not per request."""
    _attach_to_uvicorn_logging()
    try:
        STATE.update(_load_serving_state())
        STATE["ready"] = True
    except Exception as e:  # keep /health serving so the failure is diagnosable
        logger.error("startup failed: %s", e)
        STATE["ready"] = False
        STATE["error"] = str(e)
    yield
    STATE.clear()


app = FastAPI(
    title="mtl-var",
    description="Multi-task SPX transformer: next-day return, volume, volatility, VaR and CVaR.",
    version="0.1.0",
    lifespan=lifespan,
)


def bars_to_frame(bars):
    """Request payload -> the column layout engineer_features expects."""
    return pd.DataFrame({
        "observation_date": pd.to_datetime([b.date for b in bars]),
        "Open": [b.open for b in bars],
        "High": [b.high for b in bars],
        "Low": [b.low for b in bars],
        "Close": [b.close for b in bars],
        "Adj Close": [b.adj_close for b in bars],
        "Volume": [b.volume for b in bars],
    })


@app.get("/health", response_model=HealthResponse)
def health():
    if not STATE.get("ready"):
        return HealthResponse(status="degraded", model_loaded=False, caveat=STATE.get("error"))

    tags = STATE["tags"]
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_name=registry.REGISTERED_MODEL_NAME,
        model_version=str(STATE["version"].version),
        model_alias=SERVING_ALIAS,
        trained_on_fold=tags.get("fold"),
        seed=tags.get("seed"),
        feature_set=STATE["feature_set"],
        leakage_status=tags.get("leakage_status"),
        forecast_horizon="next trading day after the last bar sent",
        git_commit=tags.get("git_commit"),
        data_sha256=tags.get("data_sha256"),
        calibration_offset=STATE["offset"],
        caveat=CAVEAT,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    started = time.perf_counter()

    if not STATE.get("ready"):
        raise HTTPException(status_code=503, detail=f"model not loaded: {STATE.get('error')}")

    raw = bars_to_frame(request.bars)
    as_of = str(raw["observation_date"].iloc[-1].date())

    if request.prior_high is not None:
        prior_high, prior_high_source = float(request.prior_high), "request"
    else:
        prior_high = default_prior_high(raw["observation_date"].iloc[0])
        prior_high_source = "service history before first bar" if prior_high > 0 else "none - window only"

    features = build_forecast_frame(raw, prior_high, STATE["feature_set"])
    if len(features) < SEQUENCE_LENGTH:
        raise HTTPException(
            status_code=422,
            detail=(f"after feature engineering only {len(features)} usable rows remain, need "
                    f"{SEQUENCE_LENGTH}. Send more history: the rolling-20 terms and lag chain "
                    f"consume ~78 leading bars."),
        )

    names = STATE["feature_names"]
    window = features.iloc[-SEQUENCE_LENGTH:]
    X = STATE["feat_scaler"].transform(window[names])[np.newaxis, :, :].astype("float32")

    preds = STATE["model"].predict(X, verbose=0)
    native = {t: float(STATE["target_scalers"][t].inverse_transform(np.asarray(p).reshape(-1, 1))[0, 0])
              for t, p in zip(TARGETS, preds)}

    var_raw = native["VaR_5"]
    var_calibrated = var_raw + STATE["offset"]
    latency_ms = (time.perf_counter() - started) * 1000.0
    forecast_for = f"next trading day after {as_of}"

    logger.info("predict as_of=%s var_5=%.6f latency_ms=%.1f", as_of, var_calibrated, latency_ms)

    STATE["monitoring"].record_prediction(
        as_of_date=as_of,
        prediction={
            "log_return": native["log_return"],
            "log_volume": native["log_volume"],
            "yz_vol": native["yz_vol"],
            "var_5_raw": var_raw,
            "var_5": var_calibrated,
            "cvar_5": native["CVaR_5"],
        },
        latency_ms=latency_ms,
        feature_row=window[names].iloc[-1].to_numpy(dtype=float),
        # the placeholder row's return is fabricated; the volatility monitor must not see it
        recent_returns=features["log_return"].to_numpy(dtype=float)[:-1],
    )

    return PredictResponse(
        as_of_date=as_of,
        forecast_for=forecast_for,
        prediction=Prediction(
            log_return=native["log_return"],
            log_volume=native["log_volume"],
            yz_vol=native["yz_vol"],
            var_5_raw=var_raw,
            var_5=var_calibrated,
            cvar_5=native["CVaR_5"],
        ),
        calibration_offset=STATE["offset"],
        prior_high_used=prior_high,
        prior_high_source=prior_high_source,
        feature_set=STATE["feature_set"],
        model_name=registry.REGISTERED_MODEL_NAME,
        model_version=str(STATE["version"].version),
        model_alias=SERVING_ALIAS,
        latency_ms=round(latency_ms, 2),
        caveat=CAVEAT,
    )


class OutcomeRequest(BaseModel):
    """The realised return for a forecast already made.

    Performance drift cannot be measured without this: a VaR forecast is only scoreable once
    the day it was for has happened.
    """

    as_of_date: str = Field(..., description="the as_of_date of the prediction being scored")
    realised_return: float = Field(..., description="the log return realised on the day the forecast was for")


class OutcomeResponse(BaseModel):
    matched: bool = Field(..., description="false if no pending prediction had that as_of_date")
    as_of_date: str
    breached: bool | None = None
    var_5: float | None = None
    realised_return: float | None = None
    performance: dict


@app.post("/outcome", response_model=OutcomeResponse)
def outcome(request: OutcomeRequest):
    """Score a past forecast against what actually happened."""
    if not STATE.get("ready"):
        raise HTTPException(status_code=503, detail=f"model not loaded: {STATE.get('error')}")

    row = STATE["monitoring"].record_outcome(request.as_of_date, request.realised_return)
    perf = STATE["monitoring"].breaches.snapshot()

    if row is None:
        logger.info("outcome as_of=%s unmatched (no pending prediction)", request.as_of_date)
        return OutcomeResponse(matched=False, as_of_date=request.as_of_date, performance=perf)

    logger.info("outcome as_of=%s realised=%.6f breached=%s", request.as_of_date,
                request.realised_return, row["breached"])
    return OutcomeResponse(
        matched=True,
        as_of_date=request.as_of_date,
        breached=row["breached"],
        var_5=row["var_5"],
        realised_return=row["realised_return"],
        performance=perf,
    )


@app.get("/metrics")
def metrics():
    """Current breach rate, volatility ratio, feature-drift status and latency percentiles."""
    if not STATE.get("ready"):
        return {"status": "degraded", "error": STATE.get("error")}

    snap = STATE["monitoring"].snapshot()
    snap["model"] = {
        "name": registry.REGISTERED_MODEL_NAME,
        "version": str(STATE["version"].version),
        "alias": SERVING_ALIAS,
        "trained_on_fold": STATE["tags"].get("fold"),
        "feature_set": STATE["feature_set"],
        "calibration_offset": STATE["offset"],
        "calibration_window_vol": STATE["calibration_window_vol"],
    }
    return snap
