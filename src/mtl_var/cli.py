"""Console entry point: `mtl-var <command>`.

Relative paths (the CSV, results/) resolve against the current working directory, matching
the project's established convention of running everything from the repo root.
"""
import argparse

from . import __version__
from .data import DEFAULT_CSV


def main(argv=None):
    parser = argparse.ArgumentParser(prog="mtl-var", description="Multi-task transformer VaR pipeline")
    parser.add_argument("--version", action="version", version=f"mtl-var {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    wf = sub.add_parser("walk-forward", help="run the 5-fold regime-conditioned walk-forward campaign")
    wf.add_argument("--csv", default=DEFAULT_CSV, help=f"path to the OHLCV csv (default: {DEFAULT_CSV})")

    bf = sub.add_parser("backfill-mlflow",
                        help="create MLflow runs from already-recorded results (no retraining)")
    bf.add_argument("--experiment", default=None, help="MLflow experiment name")

    sub.add_parser("tracking-info", help="show the MLflow tracking backend this install will write to")

    args = parser.parse_args(argv)

    if args.command == "walk-forward":
        from .pipeline import main as run_walk_forward
        run_walk_forward(csv_path=args.csv)
        return 0

    if args.command == "backfill-mlflow":
        from . import tracking
        from .backfill import main as run_backfill
        run_backfill(experiment=args.experiment or tracking.DEFAULT_EXPERIMENT)
        return 0

    if args.command == "tracking-info":
        from . import tracking
        print(f"mlflow installed: {tracking.MLFLOW_AVAILABLE}")
        print(f"tracking enabled: {tracking.tracking_enabled()}")
        print(f"tracking uri:     {tracking.DEFAULT_TRACKING_URI}")
        print(f"artifact root:    {tracking.DEFAULT_ARTIFACT_ROOT}")
        print(f"experiment:       {tracking.DEFAULT_EXPERIMENT}")
        print(f"git commit:       {tracking.git_commit()}")
        print(f"\nview the ui with:  mlflow ui --backend-store-uri {tracking.DEFAULT_TRACKING_URI}")
        return 0

    parser.error(f"unknown command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
