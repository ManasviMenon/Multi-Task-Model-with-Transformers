"""Materialise already-completed runs as MLflow runs, from the recorded result JSONs.

The 10-seed x 5-fold campaign (RESULTS.md section 11) cost 50 trainings and is already done;
its results live in results/seed_distribution_results.json. Re-running it purely to populate
MLflow would burn hours of compute to recompute numbers we already have - and the campaign
script's resume logic would skip everything anyway. This reads the recorded results and
creates one run per (fold, seed) so the campaign is comparable in the UI now.

Every backfilled run is tagged `provenance=backfill` with the source file and the row it came
from. That distinction matters: these runs carry no model artifact and no live timing, because
the training that produced them happened before tracking existed. Runs produced by the
pipeline from here on are tagged `provenance=live`.
"""
import json
import os

from . import tracking

SEED_DISTRIBUTION_PATH = "results/seed_distribution_results.json"
WALK_FORWARD_PATH = "results/walk_forward_results.json"


def backfill_seed_distribution(path=SEED_DISTRIBUTION_PATH, experiment=tracking.DEFAULT_EXPERIMENT):
    """One run per (fold, seed) from the 50-training multi-seed campaign."""
    if not os.path.exists(path):
        print(f"skipped: {path} not found")
        return 0

    with open(path) as f:
        rows = json.load(f)

    n = 0
    for row in rows:
        tracking.log_standalone_run(
            run_name=f"{row['fold']}_seed{row['seed']}",
            experiment=experiment,
            provenance="backfill",
            tags={"fold": row["fold"], "seed": row["seed"], "campaign": "seed-distribution",
                  "stage": "fixed-offset-baseline", "source_file": path},
            params={"fold": row["fold"], "seed": row["seed"], **tracking.ARCHITECTURE,
                    **tracking.TRAINING_DEFAULTS},
            metrics={"var_uncal_coverage": row.get("uncalibrated_coverage"),
                     "var_cal_coverage": row.get("calibrated_coverage"),
                     "var_cal_kupiec_p": row.get("calibrated_kupiec_p")},
        )
        n += 1
    print(f"backfilled {n} runs from {path}")
    return n


def backfill_walk_forward(path=WALK_FORWARD_PATH, experiment=tracking.DEFAULT_EXPERIMENT):
    """One run per fold from the headline seed=42 walk-forward campaign, with full metrics."""
    if not os.path.exists(path):
        print(f"skipped: {path} not found")
        return 0

    with open(path) as f:
        results = json.load(f)

    n = 0
    for result in results:
        if "error" in result:
            continue
        fold_cfg = {k: result[k] for k in ("train_start", "train_end", "test_start", "test_end")}
        fold_cfg["name"] = result["fold"]
        params = tracking.fold_params(
            fold_cfg, seed=42, n_train=result.get("n_train"), n_test=result.get("n_test"),
            n_features=18, seq_len=5,
            targets=list((result.get("accuracy") or {}).keys()),
            weights={"all_tasks": 0.2},
        )
        tracking.log_standalone_run(
            run_name=f"{result['fold']}_seed42",
            experiment=experiment,
            provenance="backfill",
            tags={"fold": result["fold"], "seed": 42, "campaign": "walk-forward",
                  "stage": "walk-forward", "source_file": path},
            params=params,
            metrics=tracking.fold_metrics(result),
        )
        n += 1
    print(f"backfilled {n} runs from {path}")
    return n


def main(experiment=tracking.DEFAULT_EXPERIMENT):
    if not tracking.tracking_enabled():
        print("tracking disabled (MTL_VAR_TRACKING=0 or mlflow not installed) - nothing to do")
        return 0
    uri = tracking.configure(experiment)
    print(f"tracking uri: {uri}  experiment: {experiment}")
    total = backfill_walk_forward(experiment=experiment) + backfill_seed_distribution(experiment=experiment)
    print(f"total backfilled runs: {total}")
    return total


if __name__ == "__main__":
    main()
