"""Uniform contract shared by every analytical technique.

Every technique is used the same way:

    result = Impl(**params).add({"name": frame, ...}).result()

and returns one flat table, in canonical combination order:

    df_a, col_a[, df_b, col_b[, df_c, col_c]] | status | DESCRIPTORS | METRICS | CONCLUSIONS

The base enumerates column combinations and decides eligibility, so every
implementation of a technique computes metrics for exactly the same rows. An
implementation supplies only `_compute`; the technique base supplies
`eligible` / `compatible` / `describe` / `_conclude`.

status: "computed"   metrics filled in
        "ineligible" a column's dtype/content (or the pair's value families) do not
                     qualify - metrics and conclusions null
        "pruned"     a probabilistic implementation chose not to evaluate the
                     combination - metrics and conclusions null
Null means "not computed"; NaN means "computed but mathematically undefined".
"""

from __future__ import annotations

import importlib
import io
import math
from abc import ABC, abstractmethod
from itertools import combinations
from typing import Any, Callable, ClassVar, Literal, Self, Sequence

import polars as pl

Scope = Literal["per_column", "multi_set", "ordered"]
Column = tuple[str, str]  # (frame name, column name)
Combo = tuple[Column, ...]  # ARITY columns
STATUS = pl.Enum(["computed", "ineligible", "pruned"])
_SUFFIXES = ("a", "b", "c")


class Technique(ABC):
    SCOPE: ClassVar[Scope]
    ARITY: ClassVar[int]
    METRICS: ClassVar[dict[str, pl.DataType]]
    DESCRIPTORS: ClassVar[dict[str, pl.DataType]] = {}
    CONCLUSIONS: ClassVar[dict[str, pl.DataType]] = {}
    # Private per-row values an implementation returns for the base's conclusions;
    # never in the result (e.g. Describe's argmin / f1 / capture_history).
    INPUTS: ClassVar[dict[str, pl.DataType]] = {}
    EXACT: ClassVar[bool] = True
    RTOL: ClassVar[float] = 0.0
    ATOL: ClassVar[float] = 0.0

    def __init__(self) -> None:
        self._frames: dict[str, pl.DataFrame | pl.LazyFrame] = {}
        self._collected: dict[str, pl.DataFrame] = {}

    # ── public API ────────────────────────────────────────────────────────────

    def add(self, frames: dict[str, Any]) -> Self:
        """Register named frames: Polars DataFrames / LazyFrames, or any Arrow tabular
        object implementing the Arrow PyCapsule interface (pyarrow Table, RecordBatch,
        RecordBatchReader, …), read once into a Polars DataFrame (after the Rust
        boundary's checks: ValueError on malformed values or a Decimal256). Every
        column holding an Array or Struct (at any depth) is rebuilt first
        (`_normalise`). Lazy-safe. Names are unique for the life of the instance."""
        accepted = {}
        for name, frame in frames.items():
            if not isinstance(name, str) or not name:
                raise ValueError(f"frame names must be non-empty strings, got {name!r}")
            if not isinstance(frame, (pl.DataFrame, pl.LazyFrame)):
                if not (
                    hasattr(frame, "__arrow_c_stream__")
                    or hasattr(frame, "__arrow_c_array__")
                ):
                    raise TypeError(
                        f"frame {name!r} must be a polars DataFrame or LazyFrame, or Arrow tabular data "
                        f"(an object with __arrow_c_stream__), got {type(frame).__name__}"
                    )
                frame = _checked_arrow(frame)
                try:
                    frame = pl.DataFrame(frame)
                except Exception as exc:
                    raise TypeError(
                        f"frame {name!r} is not Arrow tabular data: {exc}"
                    ) from exc
            if name in self._frames:
                raise ValueError(f"frame {name!r} already added")
            accepted[name] = _normalise(frame)
        self._frames.update(accepted)
        self._on_add()
        return self

    def result(self) -> pl.DataFrame:
        if not self._frames:
            raise ValueError(f"{type(self).__name__}.result() called before add()")
        frames = {
            n: f.collect() if isinstance(f, pl.LazyFrame) else f
            for n, f in self._frames.items()
        }
        self._collected = frames
        try:
            ok = {
                (n, c): self.eligible(f[c])
                for n, f in frames.items()
                for c in f.columns
            }
            dtypes = {
                (n, c): dt for n, f in frames.items() for c, dt in f.schema.items()
            }
            combos = self.enumerate(frames)
            usable = [
                all(ok[col] for col in k)
                and self.compatible([dtypes[col] for col in k])
                for k in combos
            ]
            good = [k for k, u in zip(combos, usable) if u]
            bad = [k for k, u in zip(combos, usable) if not u]
            rows = (
                self._compute(frames, good) if good else self.null_frame([], "computed")
            )
            self._check(rows, len(good))
            keys = self.key_columns()
            columns = [*keys, "status", *self.METRICS, *self.INPUTS]
            out = self.keys_frame(combos).join(
                pl.concat([rows.select(columns), self.null_frame(bad, "ineligible")]),
                on=keys,
                how="left",
                maintain_order="left",
            )
            if out.height != len(combos) or out["status"].null_count():
                raise ValueError(
                    f"{type(self).__name__}._compute must return exactly one row per eligible combination"
                )
            described = self.describe(frames, combos)
            out = out.with_columns(
                [
                    pl.Series(k, v, dtype=self.DESCRIPTORS[k])
                    for k, v in described.items()
                ]
            )
            out = self._conclude(out)
            return out.select(
                *keys, "status", *self.DESCRIPTORS, *self.METRICS, *self.CONCLUSIONS
            )
        finally:
            self._collected = {}
            self._on_result_end()

    # ── row builders (used by implementations) ────────────────────────────────

    @classmethod
    def key_columns(cls) -> list[str]:
        return [f"{p}_{s}" for s in _SUFFIXES[: cls.ARITY] for p in ("df", "col")]

    @classmethod
    def enumerate(cls, frames: dict[str, pl.DataFrame]) -> list[Combo]:
        """Canonical combinations: frame insertion order, then column order."""
        if cls.SCOPE == "per_column":
            return [((n, c),) for n, f in frames.items() for c in f.columns]
        if cls.SCOPE == "ordered":
            return [
                k
                for n, f in frames.items()
                for k in combinations([(n, c) for c in f.columns], cls.ARITY)
            ]
        if cls.SCOPE == "multi_set":
            return list(
                combinations(
                    [(n, c) for n, f in frames.items() for c in f.columns], cls.ARITY
                )
            )
        raise ValueError(f"unknown SCOPE {cls.SCOPE!r}")

    @classmethod
    def keys_frame(cls, combos: Sequence[Combo]) -> pl.DataFrame:
        data: dict[str, list[str]] = {}
        for i, s in enumerate(_SUFFIXES[: cls.ARITY]):
            data[f"df_{s}"] = [k[i][0] for k in combos]
            data[f"col_{s}"] = [k[i][1] for k in combos]
        return pl.DataFrame(data, schema={k: pl.String for k in cls.key_columns()})

    @classmethod
    def metrics_frame(
        cls,
        combos: Sequence[Combo],
        metrics: dict[str, Sequence],
        status: str | Sequence[str] = "computed",
    ) -> pl.DataFrame:
        """keys + status + METRICS + INPUTS (null where not given), one row per combo,
        values in combo order."""
        statuses = [status] * len(combos) if isinstance(status, str) else list(status)
        return cls.keys_frame(combos).with_columns(
            pl.Series("status", statuses, dtype=STATUS),
            *(
                _metric_series(name, metrics[name], dtype)
                for name, dtype in cls.METRICS.items()
            ),
            *(
                _metric_series(name, metrics.get(name, [None] * len(combos)), dtype)
                for name, dtype in cls.INPUTS.items()
            ),
        )

    @classmethod
    def null_frame(cls, combos: Sequence[Combo], status: str) -> pl.DataFrame:
        return cls.metrics_frame(
            combos, {m: [None] * len(combos) for m in cls.METRICS}, status
        )

    @classmethod
    def rows_from_plugin(cls, frame: str, plugin_rows: pl.DataFrame) -> pl.DataFrame:
        """Plugin output keyed by col_a[, col_b[, col_c]] -> keys + computed status + METRICS.
        Carries no INPUTS: a technique with INPUTS selects them itself."""
        return plugin_rows.with_columns(
            *(pl.lit(frame).alias(f"df_{s}") for s in _SUFFIXES[: cls.ARITY]),
            pl.lit("computed", dtype=STATUS).alias("status"),
        ).select(
            *cls.key_columns(),
            "status",
            *(pl.col(m).cast(dt) for m, dt in cls.METRICS.items()),
        )

    # ── hooks ─────────────────────────────────────────────────────────────────

    def eligible(self, series: pl.Series) -> bool:
        """Technique base: may this column take part at all?"""
        return True

    def compatible(self, dtypes: Sequence[pl.DataType]) -> bool:
        """Technique base: may these columns be compared with each other?"""
        return True

    def describe(
        self, frames: dict[str, pl.DataFrame], combos: list[Combo]
    ) -> dict[str, list]:
        """Technique base: DESCRIPTORS values for every combo (eligible or not)."""
        return {}

    @abstractmethod
    def _compute(
        self, frames: dict[str, pl.DataFrame], combos: list[Combo]
    ) -> pl.DataFrame:
        """Implementation: keys + status ("computed"/"pruned") + METRICS + INPUTS
        (optional) for exactly `combos`."""

    def _conclude(self, out: pl.DataFrame) -> pl.DataFrame:
        """Technique base: add CONCLUSIONS (null unless status == computed)."""
        return out

    def _on_add(self) -> None:
        """Called after every add(); e.g. clears per-instance caches."""

    def _on_result_end(self) -> None:
        """Called when result() returns or raises; e.g. clears per-instance caches
        populated during _compute so nothing stays warm/stale between calls."""

    def agreement(self, result: pl.DataFrame, reference: pl.DataFrame) -> list[str]:
        """Problems found comparing `result` with the reference implementation's result."""
        return metric_mismatches(
            result,
            reference,
            self.key_columns(),
            list(self.METRICS),
            self.RTOL,
            self.ATOL,
        )

    # ── internal ──────────────────────────────────────────────────────────────

    def _check(self, rows: pl.DataFrame, expected_rows: int) -> None:
        name = type(self).__name__
        expected = {
            **{k: pl.String for k in self.key_columns()},
            "status": STATUS,
            **self.METRICS,
            **self.INPUTS,
        }
        if dict(rows.schema) != expected:
            raise TypeError(
                f"{name}._compute returned schema {dict(rows.schema)}, expected {expected}"
            )
        if rows.height != expected_rows:
            raise ValueError(
                f"{name}._compute returned {rows.height} rows for {expected_rows} combinations"
            )
        if (rows["status"] == "ineligible").any():
            raise ValueError(
                f"{name}._compute may only return status 'computed' or 'pruned'"
            )


# ── helpers for technique bases and implementations ─────────────────────────────


def _checked_arrow(frame: Any) -> Any:
    """Arrow tabular `frame` after the Rust boundary's checks (ValueError naming the
    column): py-polars trusts Arrow input, and malformed values (dictionary keys out of
    range) or a Decimal256 make it panic. An object exposing only `__arrow_c_array__`
    (one struct array) is read as a one-batch pyarrow Table first; without pyarrow (or
    when it is not a struct array) it goes to Polars as it is. The extension is
    imported here, not at module import, so pure-Python techniques do not need it."""
    if not hasattr(frame, "__arrow_c_stream__"):
        try:
            import pyarrow as pa

            frame = pa.table(frame)
        except Exception:  # no pyarrow, or not a struct array: Polars decides
            return frame
    from analytics import _plugin

    return _arrow_for_polars(_plugin.checked_table(frame))


def _arrow_for_polars(table: Any) -> Any:
    """`table` (an Arrow C stream) as py-polars reads Arrow best: a pyarrow Table when
    pyarrow is installed. py-polars' own C-stream import builds broken frames from
    some types (Decimal32 / Decimal64 abort on export) and panics, rather than raising,
    on types it cannot read (unions, intervals)."""
    try:
        import pyarrow as pa
    except ImportError:
        return table
    return pa.table(table)


def _holds_array_or_struct(dtype: pl.DataType) -> bool:
    if isinstance(dtype, (pl.Array, pl.Struct)):
        return True
    if isinstance(dtype, pl.List):
        return _holds_array_or_struct(dtype.inner)
    return False


def _normalise(frame: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    """`frame` with every column holding an Array or Struct (at any depth) rebuilt.

    py-polars 1.41 exports a *sliced* Array / Struct with nulls, at any depth, as
    inconsistent Arrow: polars-arrow slices their children eagerly but exports
    `offset = validity offset` (Array: slice offset applied twice; Struct: short
    child). The Rust extension refuses it (ValueError) and pyarrow / DataFusion raise.
    An in-memory IPC round trip of those columns rewrites every level from offset 0
    (a gather does not: a gathered List keeps a sliced inner Array / Struct). IPC is
    chosen for that correctness, not speed: it costs more than a gather (~83 ms vs
    ~16 ms for a 1M-row Struct{x: int, y: str}) and only runs for frames holding an
    Array / Struct column; flat frames are returned unchanged.
    """
    schema = frame.collect_schema() if isinstance(frame, pl.LazyFrame) else frame.schema
    cols = [c for c, dt in schema.items() if _holds_array_or_struct(dt)]
    if not cols:
        return frame
    if isinstance(frame, pl.LazyFrame):
        return frame.map_batches(_normalise, schema=schema, streamable=False)
    buf = io.BytesIO()
    frame.select(cols).write_ipc(buf, compression="uncompressed")
    buf.seek(0)
    return frame.with_columns(pl.read_ipc(buf, memory_map=False).get_columns())


def computed(expr: pl.Expr) -> pl.Expr:
    """`expr` where status == computed, null elsewhere."""
    return pl.when(pl.col("status") == "computed").then(expr)


def at_least(column: str, threshold: float) -> pl.Expr:
    """column >= threshold, with NaN -> False (Polars orders NaN above every number)."""
    return pl.col(column).is_not_nan() & (pl.col(column) >= threshold)


def check_unit(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")


def columns_of(combos: Sequence[Combo]) -> list[Column]:
    """Distinct columns used by `combos`, in first-seen order."""
    return list(dict.fromkeys(col for k in combos for col in k))


def group_by_frame(combos: Sequence[Combo]) -> dict[str, list[Combo]]:
    """Combos grouped by the frame of their first column (ordered/per-column scopes)."""
    groups: dict[str, list[Combo]] = {}
    for k in combos:
        groups.setdefault(k[0][0], []).append(k)
    return groups


def _metric_series(name: str, values: Sequence, dtype: pl.DataType) -> pl.Series:
    if isinstance(dtype, pl.Decimal):
        # polars cannot build a Decimal Series from Python ints; the cast is strict,
        # so a value too wide for the precision raises rather than being dropped.
        return pl.Series(
            name, [None if v is None else int(v) for v in values], dtype=pl.Int128
        ).cast(dtype)
    return pl.Series(name, list(values), dtype=dtype, strict=True)


def same_value(got, want, rtol: float, atol: float) -> bool:
    if got is None or want is None:
        return got is None and want is None
    if isinstance(got, float) or isinstance(want, float):
        if math.isnan(got) or math.isnan(want):
            return math.isnan(got) and math.isnan(want)
        return math.isclose(got, want, rel_tol=rtol, abs_tol=atol)
    return got == want


def metric_mismatches(
    result: pl.DataFrame,
    reference: pl.DataFrame,
    keys: list[str],
    metrics: list[str],
    rtol: float,
    atol: float,
) -> list[str]:
    """Row-by-row differences in status and `metrics`; both frames in canonical order."""
    key_rows = result.select(keys).rows()
    if key_rows != reference.select(keys).rows():
        return ["key columns differ from the reference"]
    labels = [
        " ~ ".join(f"{r[i]}.{r[i + 1]}" for i in range(0, len(r), 2)) for r in key_rows
    ]
    problems = [
        f"{label}: status {got} != {want}"
        for label, got, want in zip(labels, result["status"], reference["status"])
        if got != want
    ]
    for m in metrics:
        for label, got, want in zip(
            labels, result[m].to_list(), reference[m].to_list()
        ):
            if not same_value(got, want, rtol, atol):
                problems.append(f"{label}: {m} {got!r} != {want!r}")
    return problems


def lazy_attributes(package: str, modules: dict[str, str]) -> Callable[[str], type]:
    """Module-level __getattr__ that imports optional implementations on first use,
    so `import analytics.<technique>` never needs third-party reference libraries."""

    def __getattr__(name: str) -> type:
        if name in modules:
            # `modules` is a hard-coded name -> module map in each package's __init__; the
            # name is only looked up in it, never imported directly.
            module = importlib.import_module(modules[name], package)  # nosemgrep
            return getattr(module, name)
        raise AttributeError(f"module {package!r} has no attribute {name!r}")

    return __getattr__
