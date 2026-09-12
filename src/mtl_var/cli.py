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

    args = parser.parse_args(argv)

    if args.command == "walk-forward":
        from .pipeline import main as run_walk_forward
        run_walk_forward(csv_path=args.csv)
        return 0

    parser.error(f"unknown command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
