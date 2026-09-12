"""Tests for the MLflow tracking layer.

The important property is that tracking is optional and inert: with MTL_VAR_TRACKING=0 every
call must be a silent no-op, so the frozen research scripts behave exactly as they did before
tracking existed. The metric-flattening logic is tested directly because a mis-mapped key
would silently populate the UI with wrong numbers.

Nothing here writes to the repo's mlruns/ - the one test that actually logs points the
tracking URI at a tmp_path.
"""
import math

import pytest

from mtl_var import tracking


@pytest.fixture
def tracking_off(monkeypatch):
    monkeypatch.setenv("MTL_VAR_TRACKING", "0")


@pytest.fixture
def isolated_tracking(monkeypatch, tmp_path):
    """Real tracking, but writing into a throwaway directory."""
    monkeypatch.setenv("MTL_VAR_TRACKING", "1")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path.as_posix()}/mlflow.db")
    monkeypatch.chdir(tmp_path)


SAMPLE_RESULT = {
    "epochs_run": 50,
    "best_val_loss": 0.05673676589503884,
    "n_train": 4450,
    "n_test": 247,
    "accuracy": {
        "log_return": {"mae": 0.0038592754646273263, "rmse": 0.005528055532567594},
        "VaR_5": {"mae": 0.001314734641089321, "rmse": 0.0016194568916692532},
    },
    "var_calibration": {
        "calib_n": 123,
        "eval_n": 124,
        "offset": -0.02125245817449967,
        "uncalibrated": {"breaches": 69, "coverage": 0.5564516129032258, "kupiec_p": 0.0},
        "calibrated": {"breaches": 3, "coverage": 0.024193548387096774, "kupiec_p": 0.1443909755485061},
    },
    "rolling_calibration": {"eval_slice_only": {"coverage": 0.02459016393442623, "kupiec_p": 0.1548177802519155}},
    "rolling_calibration_triggered": {"eval_slice_only": {"coverage": 0.0328, "kupiec_p": 0.3532}},
}


def test_disabled_calls_are_noops(tracking_off):
    assert tracking.tracking_enabled() is False
    assert tracking.configure() is None
    assert tracking.log_params({"fold": "x"}) is None
    assert tracking.log_metrics({"mae": 1.0}) is None
    assert tracking.log_model(object()) is None
    assert tracking.log_standalone_run("x", params={"a": 1}) is None
    with tracking.start_run("x") as run:
        assert run is None


def test_fold_metrics_flattens_the_result_dict():
    m = tracking.fold_metrics(SAMPLE_RESULT)

    assert m["epochs_run"] == 50
    assert m["best_val_loss"] == pytest.approx(0.05673676589503884)
    assert m["mae_log_return"] == pytest.approx(0.0038592754646273263)
    assert m["rmse_VaR_5"] == pytest.approx(0.0016194568916692532)
    assert m["var_cal_coverage"] == pytest.approx(0.024193548387096774)
    assert m["var_cal_kupiec_p"] == pytest.approx(0.1443909755485061)
    assert m["var_cal_breaches"] == 3
    assert m["var_uncal_coverage"] == pytest.approx(0.5564516129032258)
    assert m["var_offset"] == pytest.approx(-0.02125245817449967)
    assert m["rolling_coverage"] == pytest.approx(0.02459016393442623)
    assert m["rolling_triggered_kupiec_p"] == pytest.approx(0.3532)


def test_fold_metrics_tolerates_missing_blocks():
    """A fold that errored, or a backfill row without rolling results, must not raise."""
    m = tracking.fold_metrics({"epochs_run": 3})
    assert m["epochs_run"] == 3
    assert m["rolling_coverage"] is None


def test_fold_params_carries_architecture_and_seed():
    cfg = {"name": "fold1_gfc", "train_start": "2004-01-01", "train_end": "2007-12-31",
           "test_start": "2008-01-01", "test_end": "2009-12-31"}
    p = tracking.fold_params(cfg, seed=43, n_train=924, n_test=501, n_features=18,
                             seq_len=5, targets=["log_return", "VaR_5"], weights={"log_return": 0.2})

    assert p["fold"] == "fold1_gfc"
    assert p["seed"] == 43
    assert p["d_model"] == 64 and p["n_blocks"] == 4 and p["n_heads"] == 4
    assert p["learning_rate"] == 1e-4
    assert p["targets"] == "log_return,VaR_5"
    assert p["rolling_L"] == 125


def test_log_metrics_skips_none_but_keeps_nan(isolated_tracking):
    """A NaN Kupiec p means zero-or-all breaches, which is a real outcome worth seeing in the
    UI. None means the value was never computed, and should not appear at all."""
    import mlflow

    run_id = tracking.log_standalone_run(
        "nan-handling",
        params={"fold": "test", "missing": None},
        metrics={"good": 1.5, "nan_p": float("nan"), "absent": None, "unparseable": "abc"},
        experiment="pytest-tracking",
    )
    assert run_id is not None

    run = mlflow.MlflowClient().get_run(run_id)
    assert run.data.metrics["good"] == 1.5
    assert math.isnan(run.data.metrics["nan_p"])
    assert "absent" not in run.data.metrics
    assert "unparseable" not in run.data.metrics
    assert "missing" not in run.data.params
    assert run.data.tags["provenance"] == "live"
