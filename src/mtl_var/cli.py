"""Console entry point: `mtl-var <command>`.

Relative paths (the CSV, results/) resolve against the current working directory, matching
the project's established convention of running everything from the repo root.
"""
import argparse

from . import __version__
from .data import DEFAULT_CSV, FEATURE_SETS


def main(argv=None):
    parser = argparse.ArgumentParser(prog="mtl-var", description="Multi-task transformer VaR pipeline")
    parser.add_argument("--version", action="version", version=f"mtl-var {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    wf = sub.add_parser("walk-forward", help="run the 5-fold regime-conditioned walk-forward campaign")
    wf.add_argument("--csv", default=DEFAULT_CSV, help=f"path to the OHLCV csv (default: {DEFAULT_CSV})")
    wf.add_argument("--feature-set", choices=sorted(FEATURE_SETS), default="original",
                    help="original reproduces the recorded results but contains target leakage "
                         "(RESULTS.md section 13); leakage_free lags every day-t feature one day")

    bf = sub.add_parser("backfill-mlflow",
                        help="create MLflow runs from already-recorded results (no retraining)")
    bf.add_argument("--experiment", default=None, help="MLflow experiment name")

    sub.add_parser("tracking-info", help="show the MLflow tracking backend this install will write to")

    reg = sub.add_parser("register-model", help="register a logged model as a new registry version")
    reg.add_argument("--model-uri", default=None,
                     help="models:/<id> to register (default: newest live model for --fold and --feature-set)")
    reg.add_argument("--fold", default="fold5_2022_bear", help="fold the model was trained on")
    reg.add_argument("--seed", type=int, default=42, help="seed the model was trained with")
    reg.add_argument("--csv", default=DEFAULT_CSV, help="training data, hashed into the lineage tags")
    reg.add_argument("--feature-set", choices=sorted(FEATURE_SETS), default="leakage_free",
                     help="feature set the model was trained on (default: leakage_free, the only servable set)")
    reg.add_argument("--supersedes", default=None, help="registry version this one replaces")
    reg.add_argument("--alias", default=None, help="alias to point at this version, e.g. production")

    promote = sub.add_parser("promote-model", help="point an alias at an existing registry version")
    promote.add_argument("version", help="registry version number")
    promote.add_argument("--alias", default="production", help="alias to set (default: production)")

    retire = sub.add_parser("retire-model",
                            help="delete a superseded version from the registry, keeping its run and artifact")
    retire.add_argument("version", help="registry version number")
    retire.add_argument("--reason", required=True, help="why it was retired - recorded in the registry")

    sub.add_parser("registry-info", help="list registry versions with their aliases and lineage")

    reb = sub.add_parser("rebase-registry",
                         help="rewrite absolute artifact paths in the tracking db (for containers)")
    reb.add_argument("--db", default="mlruns/mlflow.db", help="path to the tracking sqlite db")
    reb.add_argument("--to", required=True, help="new mlruns root, e.g. /app/mlruns")
    reb.add_argument("--dry-run", action="store_true")

    srv = sub.add_parser("serve", help="run the FastAPI prediction service")
    srv.add_argument("--host", default="127.0.0.1")
    srv.add_argument("--port", type=int, default=8000)
    srv.add_argument("--reload", action="store_true", help="auto-reload on code changes (dev only)")

    args = parser.parse_args(argv)

    if args.command == "walk-forward":
        from .pipeline import main as run_walk_forward
        run_walk_forward(csv_path=args.csv, feature_set=args.feature_set)
        return 0

    if args.command == "backfill-mlflow":
        from . import tracking
        from .backfill import main as run_backfill
        run_backfill(experiment=args.experiment or tracking.DEFAULT_EXPERIMENT)
        return 0

    if args.command == "serve":
        import uvicorn
        uvicorn.run("mtl_var.serve:app", host=args.host, port=args.port, reload=args.reload)
        return 0

    if args.command == "register-model":
        from . import registry

        model_uri, run = args.model_uri, None
        if model_uri is None:
            model_uri, run = registry.find_logged_model(fold=args.fold, feature_set=args.feature_set)
            if model_uri is None:
                parser.error(f"no live logged model found for fold {args.fold} with feature set "
                             f"{args.feature_set} - pass --model-uri")
            print(f"found logged model {model_uri} from run {run.info.run_name}")

        metrics = dict(run.data.metrics) if run is not None else {}
        registry.register(model_uri, fold=args.fold, seed=args.seed, csv_path=args.csv,
                          metrics=metrics, alias=args.alias,
                          run_id=run.info.run_id if run is not None else None,
                          feature_set=args.feature_set, supersedes=args.supersedes)
        return 0

    if args.command == "promote-model":
        from . import registry
        registry.set_alias(args.version, args.alias)
        return 0

    if args.command == "retire-model":
        from . import registry
        registry.retire(args.version, args.reason)
        return 0

    if args.command == "rebase-registry":
        from . import registry
        registry.rebase_artifact_paths(args.db, args.to, dry_run=args.dry_run)
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
