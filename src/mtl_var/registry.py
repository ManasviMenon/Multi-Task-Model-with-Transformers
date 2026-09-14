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


def default_description(fold, seed, metrics, feature_set="original", supersedes=None):
    """Short lineage blurb, including the caveats a deployer needs to see.

    The honest limitations go in the description on purpose. The VaR calibration passes its own
    fold and does not generalise across regimes (4 of 5 folds fail, robustly across 10 seeds),
    and the feature set decides whether the return head means anything at all. Anyone loading
    this from the registry should know both before trusting the output.
    """
    cov = metrics.get("var_cal_coverage")
    kp = metrics.get("var_cal_kupiec_p")
    perf = ""
    if cov is not None and kp is not None:
        perf = f" Calibrated VaR coverage {cov * 100:.2f}% (Kupiec p={kp:.4f}) on its own test window."

    if feature_set == "leakage_free":
        features = (" Features: leakage-free - every day-t feature lagged one day, so this is a genuine "
                    "one-step-ahead forecast (RESULTS.md 13). The return head has no forecasting skill: "
                    "it loses to a constant zero forecast on all 5 walk-forward folds.")
    else:
        features = (" Features: ORIGINAL set, which contains target leakage - log_close_ret is identical "
                    "to the log_return target (RESULTS.md 13). Not servable.")
    lineage = f" Supersedes version {supersedes}." if supersedes else ""

    return (
        f"Multi-task SPX transformer (5 heads: log_return, log_volume, yz_vol, VaR_5, CVaR_5). "
        f"Trained by mtl_var.pipeline.run_fold on {fold}, seed={seed}.{perf}{features}{lineage} "
        f"CAVEAT: the post-hoc VaR calibration does not generalise across regimes - 4 of 5 "
        f"walk-forward folds fail Kupiec, robustly across 10 seeds each (RESULTS.md 3, 11). "
        f"Treat the VaR head as calibrated only for conditions resembling its own test window."
    )


def register(model_uri, fold, seed, csv_path="SPX500-18Y_scaled_y.csv", metrics=None,
             description=None, name=REGISTERED_MODEL_NAME, alias=None, run_id=None,
             feature_set="original", supersedes=None):
    """Register a logged model as a new version, with lineage tags and a description.

    `feature_set` is part of the model's contract, not documentation: serving reads it to decide
    which features to build, and refuses a version trained on the leaky original set.
    """
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
        "feature_set": feature_set,
        # Machine-readable, so "is this the corrected model?" is answerable from the registry
        # alone rather than by knowing which version number happened to come after the audit.
        "leakage_status": "corrected" if feature_set == "leakage_free" else "contains_target_leakage",
        "leakage_audit": "RESULTS.md section 13",
        **data_fingerprint(csv_path),
    }
    if supersedes:
        tags["supersedes_version"] = str(supersedes)
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

    client.update_model_version(
        name, version.version,
        description=description or default_description(fold, seed, metrics, feature_set, supersedes))

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
            "feature_set": v.tags.get("feature_set", "original"),
            "leakage_status": v.tags.get("leakage_status", "contains_target_leakage"),
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
        print(f"  features   : {row['feature_set']}  ({row['leakage_status']})")
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


def find_logged_model(fold="fold5_2022_bear", provenance="live", experiment=tracking.DEFAULT_EXPERIMENT,
                      feature_set="leakage_free"):
    """Most recent logged model whose source run matches this fold, provenance and feature set.

    Runs from before feature sets existed carry no feature_set tag and count as "original".
    """
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
    runs = [r for r in runs if (r.data.tags.get("feature_set") or "original") == feature_set]
    for run in runs:
        models = client.search_logged_models([exp.experiment_id])
        for m in models:
            if m.source_run_id == run.info.run_id:
                return f"models:/{m.model_id}", run
    return None, None


def retire(version, reason, name=REGISTERED_MODEL_NAME, protected_aliases=(PRODUCTION_ALIAS,)):
    """Delete a superseded version from the registry while keeping the evidence that it existed.

    Deleting a registry version only removes it from what can be loaded by name or alias. The
    run that trained it and its logged model stay in the tracking store, so it can still be
    inspected or re-registered. Before deleting, the retirement - which version, its run and
    source, and why - is appended to the registered model's own description, so the registry
    carries the record itself rather than relying on anyone remembering it.

    Refuses to retire a version a protected alias still points at: retiring the version in
    production would take serving down. Promote the replacement first.
    """
    import datetime

    client = _client()
    mv = client.get_model_version(name, str(version))
    alias_map = client.get_registered_model(name).aliases or {}
    pointing = sorted(a for a, v in alias_map.items() if str(v) == str(version))

    blocked = [a for a in pointing if a in protected_aliases]
    if blocked:
        raise RuntimeError(f"version {version} is still @{', @'.join(blocked)} - promote a replacement first")

    for alias in pointing:
        client.delete_registered_model_alias(name, alias)

    note = (f"\n\nRetired version {version} on {datetime.date.today().isoformat()}: {reason} "
            f"Its run ({mv.run_id}) and logged model ({mv.source}) are retained in the tracking store.")
    current = client.get_registered_model(name).description or ""
    client.update_registered_model(name, description=current + note)
    client.delete_model_version(name, str(version))

    print(f"retired {name} version {version} (aliases removed: {pointing or 'none'}); "
          f"source {mv.source} retained")
    return {"version": str(version), "source": mv.source, "run_id": mv.run_id,
            "aliases_removed": pointing, "reason": reason}
