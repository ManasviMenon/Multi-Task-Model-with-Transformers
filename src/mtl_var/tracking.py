"""MLflow experiment tracking, local file backend.

Wraps the training and walk-forward runs so every (fold, seed) combination becomes a
comparable MLflow run: architecture config and seed as params, per-task MAE/RMSE plus VaR
coverage and Kupiec p as metrics, and the trained model as an artifact.

Two design constraints from the research side:

  - Tracking must never touch the RNG. Every call here is either before `set_seeds()` or
    after training has finished; nothing runs between seeding and model construction, which
    is the boundary that silently broke an earlier seed study (RESULTS.md section 11).
  - Tracking must be switchable off. `MTL_VAR_TRACKING=0` degrades every call to a no-op, so
    the frozen research scripts can run exactly as they did before. A missing mlflow install
    degrades the same way rather than raising.
"""
import contextlib
import os
import subprocess

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only in a minimal install
    mlflow = None
    MLFLOW_AVAILABLE = False

# Local backend, no server. MLflow 3.x puts the plain './mlruns' file store in maintenance
# mode and raises unless you opt out, so this uses the recommended local alternative - a
# SQLite file - while keeping the database and the artifacts together under mlruns/.
MLRUNS_DIR = "mlruns"
DEFAULT_TRACKING_URI = f"sqlite:///{MLRUNS_DIR}/mlflow.db"
DEFAULT_ARTIFACT_ROOT = f"./{MLRUNS_DIR}/artifacts"
DEFAULT_EXPERIMENT = "mtl-var-walk-forward"

# Logged as params on every run so architecture changes are visible in run comparison.
ARCHITECTURE = {
    "d_model": 64,
    "n_blocks": 4,
    "n_heads": 4,
    "key_dim": 128,
    "ff_dim": 128,
    "pooling": "global_average",
    "n_heads_out": 5,
}

TRAINING_DEFAULTS = {
    "learning_rate": 1e-4,
    "batch_size": 32,
    "max_epochs": 50,
    "patience": 5,
    "early_stopping": "val_loss, best weights restored",
}

CALIBRATION_DEFAULTS = {
    "calibration": "post-hoc additive offset, 5th pct of residuals",
    "calib_split": "test window halved chronologically",
    "rolling_L": 125,
    "rolling_R": 21,
    "monitor_S": 10,
    "elevated_ratio": 1.5,
}


def tracking_enabled():
    """Off when MTL_VAR_TRACKING is 0/false/no, or when mlflow isn't installed."""
    if not MLFLOW_AVAILABLE:
        return False
    return os.environ.get("MTL_VAR_TRACKING", "1").lower() not in ("0", "false", "no")


def git_commit():
    """Short commit hash for run lineage, or None outside a git checkout."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def configure(experiment=DEFAULT_EXPERIMENT, tracking_uri=None):
    """Point mlflow at the local SQLite store and select the experiment, creating it with an
    artifact location under mlruns/ so the database and artifacts stay together."""
    if not tracking_enabled():
        return None

    os.makedirs(MLRUNS_DIR, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))

    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=DEFAULT_ARTIFACT_ROOT)
    mlflow.set_experiment(experiment)
    return mlflow.get_tracking_uri()


@contextlib.contextmanager
def start_run(run_name=None, tags=None, experiment=DEFAULT_EXPERIMENT, nested=False):
    """Context manager yielding the active run, or None when tracking is disabled."""
    if not tracking_enabled():
        yield None
        return

    configure(experiment)
    base_tags = {"provenance": "live"}
    commit = git_commit()
    if commit:
        base_tags["git_commit"] = commit
    if tags:
        base_tags.update({k: str(v) for k, v in tags.items()})

    with mlflow.start_run(run_name=run_name, tags=base_tags, nested=nested) as run:
        yield run


def log_params(params):
    if not tracking_enabled() or not mlflow.active_run():
        return
    mlflow.log_params({k: v for k, v in params.items() if v is not None})


def log_metrics(metrics):
    """Logs float-able metrics, skipping None. NaN is kept - a NaN Kupiec p is meaningful
    (zero or all breaches), not missing data, and flattening it to 0 would read as a
    catastrophic rejection."""
    if not tracking_enabled() or not mlflow.active_run():
        return
    clean = {}
    for k, v in metrics.items():
        if v is None:
            continue
        try:
            clean[k] = float(v)
        except (TypeError, ValueError):
            continue
    if clean:
        mlflow.log_metrics(clean)


def log_model(model, input_example=None, name="model"):
    """Log a trained Keras model as a run artifact. Returns the model URI, or None."""
    if not tracking_enabled() or not mlflow.active_run():
        return None
    info = mlflow.tensorflow.log_model(model, name=name, input_example=input_example)
    return info.model_uri


def log_standalone_run(run_name, params=None, metrics=None, tags=None,
                       experiment=DEFAULT_EXPERIMENT, provenance="live"):
    """Create one complete run in a single call - start, log, end.

    Exists so a script can emit a run without wrapping its loop body in a context manager.
    That matters for the frozen research scripts, where a single added call is a far safer
    edit than re-indenting code around the seeding/training boundary. Also what the backfill
    uses to materialise historical results as runs.
    """
    if not tracking_enabled():
        return None

    configure(experiment)
    all_tags = {"provenance": provenance}
    commit = git_commit()
    if commit:
        all_tags["git_commit"] = commit
    if tags:
        all_tags.update({k: str(v) for k, v in tags.items()})

    with mlflow.start_run(run_name=run_name, tags=all_tags) as run:
        if params:
            log_params(params)
        if metrics:
            log_metrics(metrics)
        return run.info.run_id


def fold_params(fold_cfg, seed, n_train, n_test, n_features, seq_len, targets, weights):
    """The full param set for one (fold, seed) run."""
    params = {
        "fold": fold_cfg["name"],
        "train_start": fold_cfg["train_start"],
        "train_end": fold_cfg["train_end"],
        "test_start": fold_cfg["test_start"],
        "test_end": fold_cfg["test_end"],
        "seed": seed,
        "n_train": n_train,
        "n_test": n_test,
        "n_features": n_features,
        "seq_len": seq_len,
        "targets": ",".join(targets),
        "task_weights": ",".join(f"{k}={v}" for k, v in weights.items()),
    }
    params.update(ARCHITECTURE)
    params.update(TRAINING_DEFAULTS)
    params.update(CALIBRATION_DEFAULTS)
    return params


def fold_metrics(result):
    """Flatten a run_fold result dict into MLflow metric keys."""
    metrics = {
        "epochs_run": result.get("epochs_run"),
        "best_val_loss": result.get("best_val_loss"),
        "n_train": result.get("n_train"),
        "n_test": result.get("n_test"),
    }

    for task, acc in (result.get("accuracy") or {}).items():
        metrics[f"mae_{task}"] = acc.get("mae")
        metrics[f"rmse_{task}"] = acc.get("rmse")

    vc = result.get("var_calibration") or {}
    metrics["var_offset"] = vc.get("offset")
    metrics["eval_n"] = vc.get("eval_n")
    for label, key in (("uncal", "uncalibrated"), ("cal", "calibrated")):
        block = vc.get(key) or {}
        metrics[f"var_{label}_coverage"] = block.get("coverage")
        metrics[f"var_{label}_kupiec_p"] = block.get("kupiec_p")
        metrics[f"var_{label}_breaches"] = block.get("breaches")

    for label, key in (("rolling", "rolling_calibration"),
                       ("rolling_triggered", "rolling_calibration_triggered")):
        block = (result.get(key) or {}).get("eval_slice_only") or {}
        metrics[f"{label}_coverage"] = block.get("coverage")
        metrics[f"{label}_kupiec_p"] = block.get("kupiec_p")

    return metrics
