"""The run flow (spec, Data flow): registry -> frame -> recommend -> suites -> bench -> score ->
summary. Each step skips what is on disk, so `--resume` continues an interrupted run."""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import polars as pl
import yaml

from evaluate import bench
from evaluate.recommend import recommend
from evaluate.registry import RegistryError, resolve
from evaluate.rundir import DEFAULT_RUNS, RunDir, versions
from evaluate.score import score_run
from evaluate.suites import build
from evaluate.summary import report, summarise


def _loads(suite_path: Path) -> int:
    ch = yaml.safe_load(suite_path.read_text(encoding="utf-8"))["clickhouse"]
    return (
        len(ch["codec"])
        * len(ch["compression"])
        * len(ch["skip_index"])
        * len(ch["type"])
    )


def _prepare(
    run: RunDir, name: str, rows: int, load, samples: int
) -> list[dict] | None:
    """Steps 2-5 for one dataset; the suites index, or None when the recommender failed."""
    # a dataset added on --resume has no entry yet
    info = run.info()["datasets"].get(name, {"status": "pending", "rows": rows})
    frame_path = run.frame(name)
    if not frame_path.is_file():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            # via a temp file: a crash leaves no truncated frame that resume would take as done
            tmp = frame_path.with_name(frame_path.name + ".tmp")
            load().write_ipc(tmp)
            os.replace(tmp, frame_path)
        info["warnings"] = [str(w.message) for w in caught]
        for w in info["warnings"]:
            print(f"  {name}: warning: {w}")
    if not (run.result(name).is_file() and run.cost(name).is_file()):
        cost = recommend(frame_path, run.dataset(name))
        if cost["status"] != "ok":
            run.update_dataset(
                name, **{**info, "status": "failed", "error": cost["error"]}
            )
            print(f"  {name}: recommender failed (exit {cost['exit_code']})")
            return None
    index = run.suites(name) / "suites.json"
    if index.is_file():
        entries = json.loads(index.read_text(encoding="utf-8"))
    else:
        frame = pl.read_ipc(frame_path)
        result = pl.read_ipc(run.result(name))
        run.suites(name).mkdir(exist_ok=True)
        entries = build(name, result, frame, run.suites(name), samples=samples)
    info.pop("error", None)  # a stale failure from an earlier attempt
    run.update_dataset(name, **{**info, "status": "prepared"})
    return entries


def run_loop(
    args: list[str],
    *,
    workers: int = 1,
    samples: int = 3,
    dry_run: bool = False,
    resume: Path | None = None,
    root: Path = DEFAULT_RUNS,
) -> int:
    datasets = resolve(args)  # RegistryError before anything is written
    if resume is not None:
        run = RunDir.open(resume)
        held = run.info()["datasets"]
        for d in datasets:
            if d.name in held and held[d.name]["rows"] != d.rows:
                raise RegistryError(
                    f"{d.name}: the run holds {held[d.name]['rows']} rows, "
                    f"the argument asks {d.rows}; start a new run"
                )
    else:
        run = RunDir.new(root)
        run.write_info(
            {
                "versions": versions(),
                "workers": workers,
                "samples": samples,
                "recommender": {},
                "datasets": {
                    d.name: {"status": "pending", "rows": d.rows} for d in datasets
                },
            }
        )
    print(f"run {run.path}")
    prepared: dict[str, list[dict]] = {}
    for d in datasets:
        try:
            entries = _prepare(run, d.name, d.rows, d.load, samples)
        except (
            Exception
        ) as e:  # noqa: BLE001 - one dataset's failure must not stop the others
            info = run.info()["datasets"].get(d.name, {"rows": d.rows})
            error = f"{type(e).__name__}: {e}"
            run.update_dataset(d.name, **{**info, "status": "failed", "error": error})
            print(f"  {d.name}: failed ({error})")
            continue
        if entries is not None:
            prepared[d.name] = entries
    if dry_run:
        for name, entries in prepared.items():
            loads = sum(_loads(run.suites(name) / e["yaml"]) for e in entries)
            columns = len({e["column"] for e in entries})
            print(
                f"{name}: {columns} columns, {len(entries)} suites, {loads} table loads"
            )
        return 0
    version = bench.check_server()
    info = run.info()
    info["clickhouse_version"] = version
    run.write_info(info)
    jobs = [
        (run.suites(name) / e["yaml"], run.stores(name))
        for name, entries in prepared.items()
        for e in entries
    ]
    errors = bench.run_all(jobs, run.path / "cache", workers=workers)
    info = run.info()
    info["suite_errors"] = {k: v for k, v in errors.items() if v}
    for name in prepared:
        info["datasets"][name]["status"] = "ok"
        info["datasets"][name].pop("error", None)
    run.write_info(info)
    for suite, err in info["suite_errors"].items():
        print(f"  ERROR {suite}: {err}")
    notes = score_run(run)
    summarise(run, notes)
    report(run)
    return 0


def report_run(path: Path) -> int:
    run = RunDir.open(path)
    summarise(run, score_run(run))
    report(run)
    return 0
