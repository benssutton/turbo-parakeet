"""Dataset names -> frames. An argument is a registry name, a group, `name:rows` (a scalable entry
at another length), `--all`, or a path to an Arrow IPC frame (named by its sanitised stem; its row
order is the stored order). Defaults for factories with required arguments are chosen here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import polars as pl
import pyarrow as pa
from synthdata import synth_frame, synth_small, synth_windows
from synthdata.scenarios import (
    delta_frame,
    describe_mixed,
    describe_narrow,
    gcd_frame,
    integer_multiples,
    integer_random,
    low_cardinality,
    mixed_dtypes,
    near_constant,
    near_unique,
    runs_int,
    runs_str,
    skip_frame,
)
from synthdata.synth import SMALL_ROWS, WINDOW_ROWS


class RegistryError(ValueError):
    """An argument that names no dataset (the message lists the registry)."""


@dataclass(frozen=True)
class Entry:
    factory: Callable[[int], pl.DataFrame]
    rows: int
    scalable: bool


ENTRIES: dict[str, Entry] = {
    "synth_small": Entry(lambda n: synth_small(), SMALL_ROWS, False),
    "synth_windows": Entry(lambda n: synth_windows(), WINDOW_ROWS, False),
    "synth_frame": Entry(synth_frame, 50_000, True),
    "gcd": Entry(gcd_frame, 1_000_000, True),
    "delta": Entry(delta_frame, 1_000_000, True),
    "skip": Entry(skip_frame, 1_000_000, True),
    "near_unique": Entry(near_unique, 200_000, True),
    "near_constant": Entry(lambda n: near_constant(n, 3), 200_000, True),
    "integer_multiples": Entry(lambda n: integer_multiples(n, 4, 1000), 200_000, True),
    "integer_random": Entry(lambda n: integer_random(n, 4), 200_000, True),
    "runs_int": Entry(lambda n: runs_int(n, 3, 100, 1000), 200_000, True),
    "runs_str": Entry(lambda n: runs_str(n, 3, 100, 1000), 200_000, True),
    "low_cardinality": Entry(lambda n: low_cardinality(n, 4), 200_000, True),
    "mixed_dtypes": Entry(mixed_dtypes, 200_000, True),
    "describe_mixed": Entry(describe_mixed, 200_000, True),
    "describe_narrow": Entry(describe_narrow, 200_000, True),
}
GROUPS: dict[str, tuple[str, ...]] = {
    "calibration": ("synth_frame", "gcd", "delta", "skip"),
    # query time resolves at >= 1M rows (docs/results/2026-10-29-query-regret.md)
    "query_time": ("synth_frame:1000000", "gcd", "delta", "skip"),
}


@dataclass(frozen=True)
class Dataset:
    name: str
    rows: int
    load: Callable[[], pl.DataFrame]


def sanitise(name: str) -> str:
    s = re.sub(r"[^a-z0-9_]", "_", name.lower()) or "d"
    return "d_" + s if s[0].isdigit() else s


def _listing() -> str:
    return "registry: " + ", ".join([*ENTRIES, *GROUPS, "--all"])


def _ipc(arg: str) -> Dataset:
    path = Path(arg)
    try:
        with pa.ipc.open_file(path) as r:
            rows = r.read_all().num_rows
    except (OSError, pa.ArrowInvalid) as e:
        raise RegistryError(f"cannot read {arg} as Arrow IPC: {e}") from e
    return Dataset(sanitise(path.stem), rows, lambda: pl.read_ipc(path))


def _named(arg: str) -> list[Dataset]:
    name, _, rows_text = arg.partition(":")
    if name in GROUPS and not rows_text:
        return [d for n in GROUPS[name] for d in _named(n)]
    entry = ENTRIES.get(name)
    if entry is None:
        raise RegistryError(f"unknown dataset {arg!r}; {_listing()}")
    rows = entry.rows
    if rows_text:
        if not entry.scalable:
            raise RegistryError(f"{name} has a fixed size ({entry.rows} rows)")
        try:
            rows = int(rows_text)
        except ValueError:
            rows = 0
        if rows <= 0:
            raise RegistryError(f"{arg}: rows must be a positive integer")
    return [Dataset(name, rows, lambda f=entry.factory, n=rows: f(n))]


def resolve(args: list[str]) -> list[Dataset]:
    """The datasets of `args`, in order; RegistryError before anything runs."""
    out: list[Dataset] = []
    for arg in args:
        if arg == "--all":
            out += [d for n in ENTRIES for d in _named(n)]
        elif arg.endswith(".arrow") or Path(arg).is_file():
            out.append(_ipc(arg))
        else:
            out += _named(arg)
    seen: set[str] = set()
    for d in out:
        if d.name in seen:
            raise RegistryError(f"dataset {d.name} is named twice")
        seen.add(d.name)
    return out
