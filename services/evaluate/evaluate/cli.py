"""evaluate run | report | compare (python -m evaluate ...)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

from evaluate import bench
from evaluate.registry import RegistryError
from evaluate.rundir import DEFAULT_RUNS


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="evaluate")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="generate, recommend, benchmark, score")
    r.add_argument(
        "datasets",
        nargs="+",
        help="registry names, groups, name:rows, --all or .arrow paths",
    )
    r.add_argument(
        "--workers",
        type=int,
        default=1,
        help="suites timed in parallel; above 1 the query-cost scores share the server's CPUs",
    )
    r.add_argument("--samples", type=int, default=10)
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--resume", type=Path)
    r.add_argument("--recommender", choices=("oneshot", "streaming"), default="oneshot")
    r.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS)
    rep = sub.add_parser(
        "report", help="re-score a run and print its report (no server)"
    )
    rep.add_argument("run_dir", type=Path)
    c = sub.add_parser("compare", help="summary metric changes from run A to run B")
    c.add_argument("run_a", type=Path)
    c.add_argument("run_b", type=Path)
    return p


def _missing(args) -> str | None:
    """The first run directory argument that is not one (or lacks what the command reads)."""
    if args.cmd == "run":
        need = [(args.resume, "run.json")] if args.resume is not None else []
    elif args.cmd == "report":
        need = [(args.run_dir, "run.json")]
    else:
        need = [(args.run_a, "summary.arrow"), (args.run_b, "summary.arrow")]
    for path, file in need:
        if not (path / file).is_file():
            return f"{path} is not a run directory (no {file})"
    return None


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    argv = list(sys.argv[1:] if argv is None else argv)
    # `--all` is a dataset argument, not an option
    argv = ["__all__" if a == "--all" else a for a in argv]
    args = _parser().parse_args(argv)
    from evaluate.run import report_run, run_loop
    from evaluate.summary import compare

    missing = _missing(args)
    if missing is not None:
        print(f"evaluate: {missing}", file=sys.stderr)
        return 2
    try:
        if args.cmd == "run":
            names = ["--all" if a == "__all__" else a for a in args.datasets]
            return run_loop(
                names,
                workers=args.workers,
                samples=args.samples,
                dry_run=args.dry_run,
                resume=args.resume,
                recommender=args.recommender,
                root=args.runs_dir,
            )
        if args.cmd == "report":
            return report_run(args.run_dir)
        with pl.Config(tbl_rows=200, tbl_cols=30):
            print(compare(args.run_a, args.run_b))
        return 0
    except RegistryError as e:
        print(f"evaluate: {e}", file=sys.stderr)
        return 2
    except bench.ServerUnavailable as e:
        print(f"evaluate: {e}", file=sys.stderr)
        return 3
