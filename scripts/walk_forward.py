"""Walk-forward validation for the SPX500 multi-task transformer.

The implementation moved into the installable `mtl_var` package (src/mtl_var/) during the
Phase 1 packaging refactor - the code was relocated, not rewritten. This module is now a
thin re-export so that:

  - `python scripts/walk_forward.py` still runs the full 5-fold campaign, and
  - the nine sibling scripts that do `from walk_forward import ...` keep working unchanged,
    which matters because they are the provenance of the numbers in RESULTS.md.

Where each piece now lives:
  mtl_var.data         FEATURES, TARGETS, SEQUENCE_LENGTH, load_and_engineer_features, make_sequences
  mtl_var.model        loss functions, WEIGHTS, transformer_block, build_transformer_model, train_model
  mtl_var.backtest     kupiec_from_counts, kupiec_test
  mtl_var.calibration  fixed_offset_calibration, rolling_calibration_analysis
  mtl_var.monitor      the volatility-ratio monitor primitives
  mtl_var.pipeline     FOLDS, run_fold, main
  mtl_var.seeding      set_seeds
"""
# Imported first so the package sets PYTHONHASHSEED / CUDA_VISIBLE_DEVICES before TensorFlow
# is loaded, exactly as this module's own header used to.
from mtl_var.backtest import kupiec_from_counts, kupiec_test
from mtl_var.calibration import fixed_offset_calibration, rolling_calibration_analysis
from mtl_var.data import (
    FEATURES,
    SEQUENCE_LENGTH,
    TARGETS,
    load_and_engineer_features,
    make_sequences,
)
from mtl_var.model import (
    WEIGHTS,
    _assert_same_shape,
    build_transformer_model,
    huber_loss,
    log_cosh,
    quantile_loss_05,
    train_model,
    transformer_block,
    weighted_loss,
)
from mtl_var.pipeline import FOLDS, PREDICTIONS_DIR, RESULTS_PATH, build_fold_data, main, run_fold
from mtl_var.seeding import SEED, set_seeds

__all__ = [
    "FEATURES", "TARGETS", "SEQUENCE_LENGTH", "FOLDS", "WEIGHTS", "SEED",
    "RESULTS_PATH", "PREDICTIONS_DIR",
    "load_and_engineer_features", "make_sequences", "build_fold_data",
    "huber_loss", "log_cosh", "quantile_loss_05", "weighted_loss", "_assert_same_shape",
    "transformer_block", "build_transformer_model", "train_model",
    "kupiec_from_counts", "kupiec_test",
    "fixed_offset_calibration", "rolling_calibration_analysis",
    "set_seeds", "run_fold", "main",
]


if __name__ == "__main__":
    main()
