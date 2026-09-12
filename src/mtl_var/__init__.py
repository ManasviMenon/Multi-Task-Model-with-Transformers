"""Multi-task transformer for SPX500 risk forecasting.

The research pipeline this package wraps lives in `scripts/` and produced the results in
RESULTS.md. Packaging moved the reusable logic here without changing any of it: the loss
functions, model architecture, calibration procedure, and Kupiec test are the same code,
relocated. `scripts/walk_forward.py` now re-exports from this package, so every existing
script keeps working unchanged.
"""
import os

# Set before TensorFlow is imported anywhere in the process - these were module-level side
# effects of scripts/walk_forward.py, and the published results were produced under them.
# CUDA_VISIBLE_DEVICES=-1 in particular forces CPU execution, which is load-bearing for
# reproducing the recorded numbers.
os.environ["PYTHONHASHSEED"] = "42"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

__version__ = "0.1.0"

__all__ = ["__version__"]
