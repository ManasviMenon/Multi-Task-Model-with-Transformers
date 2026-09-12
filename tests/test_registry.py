"""Tests for the model registry layer.

The property that matters for serving: an alias resolves to exactly one pinned version, and
re-pointing the alias is what promotion means. Also checks that lineage tags actually land -
a version without its data hash and commit is not traceable, which defeats the point.

Everything runs against a throwaway SQLite store in tmp_path; nothing touches the repo's
mlruns/ or its registered model.
"""
import numpy as np
import pytest

from mtl_var import registry, tracking


@pytest.fixture
def isolated_registry(monkeypatch, tmp_path):
    monkeypatch.setenv("MTL_VAR_TRACKING", "1")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path.as_posix()}/mlflow.db")
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def tiny_model_uri(isolated_registry):
    """A trivial logged model - the registry doesn't care what the weights are."""
    import mlflow
    import tensorflow as tf

    model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(3,))])
    with tracking.start_run(run_name="tiny"):
        tracking.log_metrics({"var_cal_coverage": 0.0242, "var_cal_kupiec_p": 0.1444})
        info = mlflow.tensorflow.log_model(model, name="model",
                                          input_example=np.zeros((1, 3), dtype="float32"))
    return info.model_uri


def test_data_fingerprint_is_content_addressed(tmp_path):
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    a.write_text("date,close\n2024-01-01,100\n")
    b.write_text("date,close\n2024-01-01,100\n")
    c = tmp_path / "c.csv"
    c.write_text("date,close\n2024-01-01,999\n")

    fa, fb, fc = (registry.data_fingerprint(str(p)) for p in (a, b, c))
    assert fa["data_sha256"] == fb["data_sha256"]
    assert fa["data_sha256"] != fc["data_sha256"]
    assert fa["data_file"] == "a.csv"
    assert fa["data_bytes"] > 0


def test_data_fingerprint_handles_missing_file():
    f = registry.data_fingerprint("does-not-exist.csv")
    assert f["data_sha256"] == "missing"
    assert f["data_bytes"] == 0


def test_description_carries_the_generalisation_caveat():
    d = registry.default_description("fold5_2022_bear", 42,
                                     {"var_cal_coverage": 0.0242, "var_cal_kupiec_p": 0.1444})
    assert "fold5_2022_bear" in d
    assert "seed=42" in d
    assert "CAVEAT" in d
    assert "4 of 5" in d  # the finding a deployer must not miss


def test_resolve_uri_is_alias_addressed():
    assert registry.resolve_uri("production") == "models:/mtl-var-transformer@production"
    assert registry.resolve_uri("staging") == "models:/mtl-var-transformer@staging"


def test_register_records_lineage_tags(tiny_model_uri):
    v = registry.register(tiny_model_uri, fold="fold5_2022_bear", seed=42,
                          csv_path="does-not-exist.csv",
                          metrics={"var_cal_coverage": 0.0242, "var_cal_kupiec_p": 0.1444})
    assert int(v.version) == 1  # mlflow returns version as an int, not a string

    mv = registry._client().get_model_version(registry.REGISTERED_MODEL_NAME, str(v.version))
    assert mv.tags["fold"] == "fold5_2022_bear"
    assert mv.tags["seed"] == "42"
    assert mv.tags["source"] == "mtl_var.pipeline.run_fold"
    assert "git_commit" in mv.tags
    assert "data_sha256" in mv.tags
    assert "CAVEAT" in mv.description


def test_alias_pins_one_version_and_promotion_repoints_it(tiny_model_uri):
    v1 = registry.register(tiny_model_uri, fold="fold5_2022_bear", seed=42, csv_path="x.csv")
    v2 = registry.register(tiny_model_uri, fold="fold5_2022_bear", seed=43, csv_path="x.csv")
    assert (int(v1.version), int(v2.version)) == (1, 2)

    registry.set_alias(v1.version, "production")
    assert int(registry.get_version("production").version) == 1

    registry.set_alias(v2.version, "production")  # the promotion
    assert int(registry.get_version("production").version) == 2
    assert registry.get_version("production").tags["seed"] == "43"


def test_get_version_returns_none_for_unset_alias(tiny_model_uri):
    registry.register(tiny_model_uri, fold="f", seed=1, csv_path="x.csv")
    assert registry.get_version("nonexistent-alias") is None


def test_describe_reports_aliases_against_versions(tiny_model_uri, capsys):
    v = registry.register(tiny_model_uri, fold="fold5_2022_bear", seed=42, csv_path="x.csv",
                          alias="production")
    rows = registry.describe()
    assert len(rows) == 1
    assert rows[0]["version"] == v.version
    assert rows[0]["aliases"] == ["production"]  # regression: search_model_versions returns []
    assert "@production" in capsys.readouterr().out


def test_registry_is_noop_when_tracking_disabled(monkeypatch):
    monkeypatch.setenv("MTL_VAR_TRACKING", "0")
    assert registry.register("models:/whatever", fold="f", seed=1) is None
