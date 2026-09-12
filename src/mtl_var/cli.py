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

    reg = sub.add_parser("register-model", help="register a logged model as a new registry version")
    reg.add_argument("--model-uri", default=None,
                     help="models:/<id> to register (default: newest live model for --fold)")
    reg.add_argument("--fold", default="fold5_2022_bear", help="fold the model was trained on")
    reg.add_argument("--seed", type=int, default=42, help="seed the model was trained with")
    reg.add_argument("--csv", default=DEFAULT_CSV, help="training data, hashed into the lineage tags")
    reg.add_argument("--alias", default=None, help="alias to point at this version, e.g. production")

    promote = sub.add_parser("promote-model", help="point an alias at an existing registry version")
    promote.add_argument("version", help="registry version number")
    promote.add_argument("--alias", default="production", help="alias to set (default: production)")

    sub.add_parser("registry-info", help="list registry versions with their aliases and lineage")

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

    if args.command == "register-model":
        from . import registry

        model_uri, run = args.model_uri, None
        if model_uri is None:
            model_uri, run = registry.find_logged_model(fold=args.fold)
            if model_uri is None:
                parser.error(f"no live logged model found for fold {args.fold} - pass --model-uri")
            print(f"found logged model {model_uri} from run {run.info.run_name}")

        metrics = dict(run.data.metrics) if run is not None else {}
        registry.register(model_uri, fold=args.fold, seed=args.seed, csv_path=args.csv,
                          metrics=metrics, alias=args.alias,
                          run_id=run.info.run_id if run is not None else None)
        return 0

    if args.command == "promote-model":
        from . import registry
        registry.set_alias(args.version, args.alias)
        return 0

    if args.command == "registry-info":
        from . import registry
        rows = registry.describe()
        if not rows:
            print("no versions registered yet - run: mtl-var register-model --alias production")
        else:
            print(f"\nserving loads: {registry.resolve_uri()}")
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
