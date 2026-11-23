"""The test window must never reach training, early stopping or best-weight restoration.

Until RESULTS.md section 15, every training call passed the fold's test window as the
early-stopping validation set. These tests hold the fix in place three ways:
  1. the training functions have no parameter that could receive test data, and their bodies
     reference no identifier containing "test";
  2. the validation split is the chronological tail of the training sequences;
  3. end to end, the pipeline's fold run hands the trainer exactly the training sequences (fit
     part + tail) and nothing else. The trainer is stubbed, so this runs in seconds.
"""
import ast
import inspect
import os
import textwrap

import numpy as np
import pytest

from mtl_var import model as model_mod
from mtl_var import pipeline
from mtl_var.data import load_and_engineer_features

TRAINING_FUNCTIONS = [model_mod.train_model, model_mod.train_single_task,
                      model_mod.fit_with_train_tail_validation, model_mod.chronological_val_split]


@pytest.mark.parametrize("fn", TRAINING_FUNCTIONS, ids=lambda f: f.__name__)
def test_training_functions_take_no_test_argument(fn):
    params = inspect.signature(fn).parameters
    assert not [p for p in params if "test" in p.lower()], params


@pytest.mark.parametrize("fn", TRAINING_FUNCTIONS, ids=lambda f: f.__name__)
def test_training_function_bodies_reference_no_test_identifier(fn):
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.arg for n in ast.walk(tree) if isinstance(n, ast.arg)}
    names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not [n for n in names if "test" in n.lower()], names


def test_val_split_is_the_chronological_tail():
    X = np.arange(100).reshape(100, 1, 1)
    y = {"a": np.arange(100), "b": -np.arange(100)}
    X_fit, y_fit, X_val, y_val = model_mod.chronological_val_split(X, y, 0.15)
    assert len(X_val) == 15 and len(X_fit) == 85
    assert X_fit.ravel().tolist() == list(range(85))
    assert X_val.ravel().tolist() == list(range(85, 100))       # last 15%, order kept
    assert y_val["a"].tolist() == list(range(85, 100))
    assert y_fit["b"].tolist() == [-i for i in range(85)]


def test_val_split_rejects_degenerate_fraction():
    with pytest.raises(ValueError):
        model_mod.chronological_val_split(np.zeros((3, 1, 1)), np.zeros(3), 0.01)


class _StubModel:
    def predict(self, X, verbose=0):
        return [np.zeros((len(X), 1)) for _ in range(5)]


@pytest.mark.skipif(not os.path.exists("SPX500-18Y_scaled_y.csv"), reason="needs the project csv")
def test_pipeline_fold_run_gives_trainer_only_training_sequences(monkeypatch):
    monkeypatch.setenv("MTL_VAR_TRACKING", "0")
    seen = {}

    def recording_trainer(X_fit, y_fit, X_val, y_val, n_features):
        seen.update(X_fit=X_fit, y_fit=y_fit, X_val=X_val, y_val=y_val)
        return _StubModel(), 1, 0.0

    monkeypatch.setattr(model_mod, "train_model", recording_trainer)

    fold = next(f for f in pipeline.FOLDS if f["name"] == "fold5_2022_bear")
    data = load_and_engineer_features("SPX500-18Y_scaled_y.csv")
    result = pipeline.run_fold(fold, data, feature_set="leakage_free", save_predictions=False)

    framed = pipeline.frame_for_feature_set(data, "leakage_free")
    fd = pipeline.build_fold_data(fold, framed, pipeline.FEATURE_SETS["leakage_free"])
    X_train, X_test = fd["X_train"], fd["X_test"]

    # exactly the training sequences, split fit | tail, in time order
    assert np.array_equal(np.concatenate([seen["X_fit"], seen["X_val"]]), X_train)
    assert len(seen["X_val"]) == round(len(X_train) * model_mod.VAL_FRACTION)
    for t, v in fd["y_train_z"].items():
        assert np.array_equal(np.concatenate([seen["y_fit"][t], seen["y_val"][t]]), v)

    # the training rows end before the test window starts
    assert fd["train_df"]["observation_date"].max() < fd["test_df"]["observation_date"].min()
    assert len(seen["X_val"]) != len(X_test) or not np.array_equal(seen["X_val"], X_test)

    assert result["early_stopping"] == pipeline.EARLY_STOPPING
    assert (result["n_fit"], result["n_val"]) == (len(seen["X_fit"]), len(seen["X_val"]))


def test_new_runs_cannot_overwrite_recorded_results():
    for fs in pipeline.FEATURE_SETS:
        assert pipeline.results_path(fs).endswith("_cleanval.json")
        assert pipeline.predictions_dir(fs).endswith("_cleanval")
        assert pipeline.results_path(fs) != pipeline.RESULTS_PATH
        assert pipeline.predictions_dir(fs) != pipeline.PREDICTIONS_DIR
