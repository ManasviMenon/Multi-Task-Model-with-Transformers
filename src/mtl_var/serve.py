"""FastAPI serving layer: POST /predict, GET /health.

Loads the registry-pinned model once at startup (`models:/mtl-var-transformer@production`),
never a loose file, so what is served is always a version with recorded lineage.

Three things the raw model cannot do on its own, which this layer supplies:

  1. **Features.** The network takes 18 engineered, standardised features over a 5-day window,
     not raw OHLCV. Requests carry raw bars and go through `data.engineer_features` - the same
     function training used, not a reimplementation.

  2. **Scalers.** The StandardScalers were fitted on the registered model's training fold and
     were never persisted. They are refitted at startup from the same fold window, which is
     deterministic and therefore reproduces the training-time scaling exactly. The cost is
     that serving needs the training csv on disk; startup verifies its SHA-256 against the
     hash recorded on the registered version and refuses to serve on a mismatch, so a swapped
     data file fails loudly instead of silently shifting every prediction.

  3. **The calibration offset.** The post-hoc additive offset is read from the registered
     version's `var_offset` tag. The model alone is not servable without it - the raw VaR head
     is uncalibrated.

The honest caveat, repeated from the registry description: this VaR calibration does not
generalise across regimes (4 of 5 walk-forward folds fail Kupiec). /health reports it so an
operator sees it without reading the paper.
"""
import logging
import os
import time
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator, model_validator
from sklearn.preprocessing import StandardScaler

from . import registry
from .data import DEFAULT_CSV, FEATURES, SEQUENCE_LENGTH, TARGETS, engineer_features, load_and_engineer_features
from .pipeline import FOLDS, build_fold_data

logger = logging.getLogger("mtl_var.serve")

SERVING_FOLD = os.environ.get("MTL_VAR_SERVING_FOLD", "fold5_2022_bear")
SERVING_ALIAS = os.environ.get("MTL_VAR_SERVING_ALIAS", registry.PRODUCTION_ALIAS)
CSV_PATH = os.environ.get("MTL_VAR_CSV", DEFAULT_CSV)

# Rolling(20) terms plus the lag chain consume ~78 leading rows before the first usable
# feature row; a 5-day input window needs 4 more on top.
MIN_BARS = 90

STATE = {}


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
        description=("Highest Adj Close BEFORE the first bar. The `drawdown` feature is measured "
                     "against a running all-time high, so a request window that doesn't contain "
                     "the true prior peak would understate it. Defaults to the training data's "
                     "maximum, which is correct when the request continues on from that data."),
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
    as_of_date: str = Field(..., description="date of the last bar the prediction conditions on")
    prediction: Prediction
    calibration_offset: float
    prior_high_used: float = Field(..., description="running-high basis used for the drawdown feature")
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
    git_commit: str | None = None
    data_sha256: str | None = None
    calibration_offset: float | None = None
    min_bars_required: int = MIN_BARS
    caveat: str | None = None


CAVEAT = ("VaR calibration is fitted on this model's own fold and does not generalise across "
          "regimes: 4 of 5 walk-forward folds fail the Kupiec test, robustly across 10 seeds.")


def _load_serving_state():
    """Load model, scalers, and calibration offset. Raises if the registry isn't populated."""
    version = registry.get_version(SERVING_ALIAS)
    if version is None:
        raise RuntimeError(
            f"no model registered at alias @{SERVING_ALIAS}. "
            f"Run: mtl-var register-model --alias {SERVING_ALIAS}"
        )

    tags = version.tags or {}
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

    # Refit the fold's scalers. build_fold_data is deterministic, so these are the same
    # objects training used - no RNG involved, nothing trained here.
    data = load_and_engineer_features(CSV_PATH)
    fd = build_fold_data(fold_cfg, data)

    feat_scaler = StandardScaler()
    train_mask = ((data['observation_date'] >= fold_cfg['train_start'])
                  & (data['observation_date'] <= fold_cfg['train_end']))
    feat_scaler.fit(data.loc[train_mask, FEATURES])

    model = registry.load(SERVING_ALIAS)

    logger.info("loaded %s version %s (@%s), fold=%s, offset=%.10g",
                registry.REGISTERED_MODEL_NAME, version.version, SERVING_ALIAS,
                fold_cfg["name"], offset)

    return {
        "model": model,
        "version": version,
        "tags": tags,
        "offset": offset,
        "feat_scaler": feat_scaler,
        "target_scalers": fd["target_scalers"],
        "fold": fold_cfg["name"],
        # Default running-high basis for the drawdown feature - see reseat_drawdown.
        "training_max_close": float(data["Adj Close"].max()),
    }


def reseat_drawdown(engineered, prior_high):
    """Recompute `drawdown` against a running high seeded with history before the window.

    Of the 18 features, 17 converge once the rolling-20 terms and lag chain have warmed up -
    verified by diffing a windowed frame against the full-history frame. `drawdown` is the
    exception: it is `close / cummax(close) - 1`, so a request window that doesn't happen to
    contain the true prior peak measures the drawdown against a lower high and understates it.
    On a 150-bar window ending 2022-12-30 that produced -0.0856 instead of -0.1995, which was
    enough to flip the sign of the predicted return.

    Seeding the accumulation with the prior high reproduces the training-time value exactly.
    """
    engineered = engineered.copy()
    closes = engineered["Adj Close"].to_numpy(dtype=float)
    running = np.maximum(np.maximum.accumulate(closes), float(prior_high))
    engineered["drawdown"] = closes / running - 1.0
    return engineered


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
    engineered = engineer_features(raw)

    prior_high = request.prior_high if request.prior_high is not None else STATE["training_max_close"]
    engineered = reseat_drawdown(engineered, prior_high)

    if len(engineered) < SEQUENCE_LENGTH:
        raise HTTPException(
            status_code=422,
            detail=(f"after feature engineering only {len(engineered)} usable rows remain, need "
                    f"{SEQUENCE_LENGTH}. Send more history: the rolling-20 terms and lag chain "
                    f"consume ~78 leading bars."),
        )

    window = engineered.iloc[-SEQUENCE_LENGTH:]
    X = STATE["feat_scaler"].transform(window[FEATURES])[np.newaxis, :, :].astype("float32")

    preds = STATE["model"].predict(X, verbose=0)
    native = {t: float(STATE["target_scalers"][t].inverse_transform(np.asarray(p).reshape(-1, 1))[0, 0])
              for t, p in zip(TARGETS, preds)}

    var_raw = native["VaR_5"]
    var_calibrated = var_raw + STATE["offset"]
    latency_ms = (time.perf_counter() - started) * 1000.0

    as_of = str(pd.Timestamp(window["observation_date"].iloc[-1]).date())
    logger.info("predict as_of=%s var_5=%.6f latency_ms=%.1f", as_of, var_calibrated, latency_ms)

    return PredictResponse(
        as_of_date=as_of,
        prediction=Prediction(
            log_return=native["log_return"],
            log_volume=native["log_volume"],
            yz_vol=native["yz_vol"],
            var_5_raw=var_raw,
            var_5=var_calibrated,
            cvar_5=native["CVaR_5"],
        ),
        calibration_offset=STATE["offset"],
        prior_high_used=float(prior_high),
        model_name=registry.REGISTERED_MODEL_NAME,
        model_version=str(STATE["version"].version),
        model_alias=SERVING_ALIAS,
        latency_ms=round(latency_ms, 2),
        caveat=CAVEAT,
    )
