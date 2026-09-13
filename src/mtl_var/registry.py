"""MLflow model registry: versioned models with recorded lineage, addressed by alias.

The serving layer must load a specific registered version, never a loose .keras file lying in
the repo. Everything here is local - the same SQLite store the tracking phase writes to.

Aliases, not stages. MLflow deprecated the Staging/Production *stage* API in 2.9 in favour of
named aliases, so "production" and "staging" here are aliases: `models:/mtl-var-transformer
@production` resolves to one pinned version, and re-pointing the alias is how a promotion
happens. Same operational meaning, on the API that isn't scheduled for removal.

Lineage is recorded on every version because "which model is this?" has to be answerable
later: git commit, the script that trained it, the data file with its SHA-256, and the fold
and seed. The data hash matters most - the whole research finding is that a calibration fitted
on one window fails on the next, so a model version is only meaningful against the exact data
that produced it.
"""
import hashlib
import os

from . import tracking

REGISTERED_MODEL_NAME = "mtl-var-transformer"
PRODUCTION_ALIAS = "production"
STAGING_ALIAS = "staging"


def _client():
    import mlflow

    tracking.configure()
    return mlflow.MlflowClient()


def data_fingerprint(csv_path):
    """SHA-256 of the training data, so a version pins the data that produced it."""
    if not os.path.exists(csv_path):
        return {"data_file": str(csv_path), "data_sha256": "missing", "data_bytes": 0}

    h = hashlib.sha256()
    size = 0
    with open(csv_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
            size += len(chunk)
    return {"data_file": os.path.basename(csv_path), "data_sha256": h.hexdigest()[:16], "data_bytes": size}


def default_description(fold, seed, metrics):
    """Short lineage blurb, including the caveat a deployer needs to see.

    The honest limitation goes in the description on purpose: this model's VaR calibration
    passes its own fold and does not generalise across regimes (4 of 5 folds fail, robustly
    across 10 seeds). Anyone loading this from the registry should know that before trusting
    the VaR head.
    """
    cov = metrics.get("var_cal_coverage")
    kp = metrics.get("var_cal_kupiec_p")
    perf = ""
    if cov is not None and kp is not None:
        perf = f" Calibrated VaR coverage {cov * 100:.2f}% (Kupiec p={kp:.4f}) on its own test window."

    return (
        f"Multi-task SPX transformer (5 heads: log_return, log_volume, yz_vol, VaR_5, CVaR_5). "
        f"Trained by mtl_var.pipeline.run_fold on {fold}, seed={seed}.{perf} "
        f"CAVEAT: the post-hoc VaR calibration does not generalise across regimes - 4 of 5 "
        f"walk-forward folds fail Kupiec, robustly across 10 seeds each (RESULTS.md 3, 11). "
        f"Treat the VaR head as calibrated only for conditions resembling its own test window."
    )


def register(model_uri, fold, seed, csv_path="SPX500-18Y_scaled_y.csv", metrics=None,
             description=None, name=REGISTERED_MODEL_NAME, alias=None, run_id=None):
    """Register a logged model as a new version, with lineage tags and a description."""
    if not tracking.tracking_enabled():
        print("tracking disabled - cannot register")
        return None

    client = _client()
    metrics = metrics or {}

    try:
        client.create_registered_model(
            name,
            description="Multi-task transformer for SPX500 risk forecasting. "
                        "Versions are pinned to the data and commit that produced them; "
                        "the serving layer loads by alias, not by version number.",
        )
    except Exception:
        pass  # already exists

    version = client.create_model_version(name=name, source=model_uri, run_id=run_id)

    tags = {
        "fold": str(fold),
        "seed": str(seed),
        "source": "mtl_var.pipeline.run_fold",
        "git_commit": tracking.git_commit() or "unknown",
        "mlflow_model_uri": model_uri,
        **data_fingerprint(csv_path),
    }
    # var_offset is load-bearing for serving, not just documentation: it is the post-hoc
    # calibration constant added to the raw VaR head, so the model alone is not servable
    # without it.
    for key in ("var_offset", "var_cal_coverage", "var_cal_kupiec_p", "best_val_loss", "mae_log_return"):
        if metrics.get(key) is not None:
            # repr round-trips exactly; a shortened format would quietly perturb the offset
            # that serving adds to every VaR prediction.
            tags[key] = repr(float(metrics[key]))

    for k, v in tags.items():
        client.set_model_version_tag(name, version.version, k, v)

    client.update_model_version(name, version.version,
                                description=description or default_description(fold, seed, metrics))

    if alias:
        set_alias(version.version, alias, name=name)

    print(f"registered {name} version {version.version} from {model_uri}")
    return version


def set_alias(version, alias, name=REGISTERED_MODEL_NAME):
    """Point an alias at a version. This is the promotion step."""
    client = _client()
    client.set_registered_model_alias(name, alias, str(version))
    print(f"alias @{alias} -> {name} version {version}")


def resolve_uri(alias=PRODUCTION_ALIAS, name=REGISTERED_MODEL_NAME):
    """The URI the serving layer loads. Pinned by alias, resolved to one version."""
    return f"models:/{name}@{alias}"


def get_version(alias=PRODUCTION_ALIAS, name=REGISTERED_MODEL_NAME):
    """The ModelVersion an alias currently points at, or None."""
    try:
        return _client().get_model_version_by_alias(name, alias)
    except Exception:
        return None


def load(alias=PRODUCTION_ALIAS, name=REGISTERED_MODEL_NAME):
    """Load the aliased model. This is what serving calls at startup."""
    import mlflow

    tracking.configure()
    return mlflow.tensorflow.load_model(resolve_uri(alias, name))


def describe(name=REGISTERED_MODEL_NAME):
    """Print every version with its aliases and lineage. Returns the rows."""
    client = _client()
    try:
        versions = client.search_model_versions(f"name='{name}'")
    except Exception as e:
        print(f"no registered model '{name}': {e}")
        return []

    # search_model_versions returns versions with an empty aliases field; the mapping only
    # comes back on the registered model itself, so invert it here.
    try:
        alias_map = {}
        for alias, ver in (client.get_registered_model(name).aliases or {}).items():
            alias_map.setdefault(str(ver), []).append(alias)
    except Exception:
        alias_map = {}

    rows = []
    for v in sorted(versions, key=lambda v: int(v.version)):
        row = {
            "version": v.version,
            "aliases": sorted(alias_map.get(str(v.version), [])),
            "fold": v.tags.get("fold"),
            "seed": v.tags.get("seed"),
            "git_commit": v.tags.get("git_commit"),
            "data_file": v.tags.get("data_file"),
            "data_sha256": v.tags.get("data_sha256"),
            "var_cal_coverage": v.tags.get("var_cal_coverage"),
            "var_cal_kupiec_p": v.tags.get("var_cal_kupiec_p"),
            "source": v.source,
        }
        rows.append(row)

        alias_str = ", ".join(f"@{a}" for a in row["aliases"]) or "(no alias)"
        print(f"\nversion {row['version']}  {alias_str}")
        print(f"  trained on : {row['fold']}  seed={row['seed']}")
        print(f"  commit     : {row['git_commit']}")
        print(f"  data       : {row['data_file']}  sha256={row['data_sha256']}")
        print(f"  VaR cal    : coverage={row['var_cal_coverage']}  kupiec_p={row['var_cal_kupiec_p']}")
        print(f"  source     : {row['source']}")
        if v.description:
            print(f"  description: {v.description[:110]}...")
    return rows


# Columns in the MLflow SQLite schema that store absolute artifact paths. Found by scanning
# every text column for the host root rather than guessing from the schema docs.
_ARTIFACT_PATH_COLUMNS = [
    ("runs", "artifact_uri"),
    ("logged_models", "artifact_location"),
    ("experiments", "artifact_location"),
    ("model_versions", "storage_location"),
]


def rebase_artifact_paths(db_path, new_root, dry_run=False):
    """Rewrite absolute artifact paths in a tracking DB to sit under `new_root`.

    MLflow resolves a relative artifact location to an absolute one when the experiment is
    created, so the store carries the host's paths. Copy mlruns/ into a container and every
    artifact lookup fails with `No such artifact: ''` - the model is physically present but
    the recorded path does not exist there. This rewrites the four columns that hold such
    paths so the registry is portable.

    Matches any `file:...` URI up to and including a `/mlruns` segment, so it works regardless
    of where the store was originally created.
    """
    import re
    import sqlite3

    pattern = re.compile(r"^file:.*?/mlruns")
    replacement = f"file:{str(new_root).replace(os.sep, '/').rstrip('/')}"

    con = sqlite3.connect(str(db_path))
    changed = 0
    try:
        cur = con.cursor()
        for table, column in _ARTIFACT_PATH_COLUMNS:
            try:
                rows = cur.execute(f'SELECT rowid, "{column}" FROM "{table}"').fetchall()
            except sqlite3.OperationalError:
                continue  # table absent in this MLflow version
            for rowid, value in rows:
                if not value or not isinstance(value, str):
                    continue
                new_value = pattern.sub(replacement, value)
                if new_value != value:
                    changed += 1
                    if not dry_run:
                        cur.execute(f'UPDATE "{table}" SET "{column}" = ? WHERE rowid = ?',
                                    (new_value, rowid))
        if not dry_run:
            con.commit()
    finally:
        con.close()

    print(f"{'would rewrite' if dry_run else 'rewrote'} {changed} artifact paths -> {replacement}")
    return changed


def find_logged_model(fold="fold5_2022_bear", provenance="live", experiment=tracking.DEFAULT_EXPERIMENT):
    """Most recent logged model whose source run matches this fold and provenance."""
    import mlflow

    tracking.configure(experiment)
    client = mlflow.MlflowClient()
    exp = client.get_experiment_by_name(experiment)
    if exp is None:
        return None, None

    runs = client.search_runs(
        [exp.experiment_id],
        filter_string=f"tags.fold = '{fold}' and tags.provenance = '{provenance}'",
        order_by=["attributes.start_time DESC"],
    )
    for run in runs:
        models = client.search_logged_models([exp.experiment_id])
        for m in models:
            if m.source_run_id == run.info.run_id:
                return f"models:/{m.model_id}", run
    return None, None
