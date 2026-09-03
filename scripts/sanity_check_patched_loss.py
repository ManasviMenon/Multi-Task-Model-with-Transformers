"""
Sanity check before trusting the fold campaign: does the ACTUAL patched weighted_loss in
walk_forward.py (not the standalone duplicate in verify_loss_bug_impact.py) reproduce the
~48.2% uncal -> 2.76% cal, p=0.015 result found there? Same single 80/20 split, same seed.

If this doesn't land near that, something differs between the patch and the verification
script, and that needs resolving before the 5-fold campaign is trusted.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from walk_forward import (
    FEATURES, SEQUENCE_LENGTH, TARGETS,
    weighted_loss,  # the actual patched function, imported live from walk_forward.py
    build_transformer_model, load_and_engineer_features,
    kupiec_from_counts, make_sequences,
)
from verify_loss_bug_impact import build_single_split, train_once, evaluate

CSV_PATH = "SPX500-18Y_scaled_y.csv"


def main():
    print("Building single 80/20 split...")
    X_train, X_test, y_train_raw, y_test_raw, y_train_z, y_test_z, target_scalers = build_single_split()
    n_features = len(FEATURES)
    print(f"n_train={len(X_train)}  n_test={len(X_test)}\n")

    print("Training with walk_forward.py's ACTUAL patched weighted_loss...")
    model = train_once(weighted_loss, X_train, y_train_z, X_test, y_test_z, n_features)
    acc, var = evaluate(model, X_test, y_test_raw, target_scalers)

    print("\nExpected (from verify_loss_bug_impact.py's standalone 'fixed' run): "
          "uncal ~48.2%, cal ~2.76%, p~0.015")
    print(f"Actual (patched walk_forward.weighted_loss): "
          f"uncal {var['uncalibrated']['coverage']*100:.2f}%, "
          f"cal {var['calibrated']['coverage']*100:.2f}%, "
          f"p={var['calibrated']['kupiec_p']:.5g}")
    for t in TARGETS:
        print(f"  {t}: MAE={acc[t]['mae']:.5f}")


if __name__ == "__main__":
    main()
