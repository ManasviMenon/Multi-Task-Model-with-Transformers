# Leakage-free results trained with test-window early stopping

Byte-for-byte copies, taken 2026-11-23, of every leakage-free result whose neural model used
the fold's TEST window as its early-stopping and best-weight validation set (RESULTS.md section 15).
Features and targets were leakage-free, but model selection looked at test-period targets.

The originals also remain at their old paths under `results/`, unchanged. The corrected runs,
with early stopping on the last 15% of each fold's training sequences, are written to `*_cleanval*`
paths instead.
