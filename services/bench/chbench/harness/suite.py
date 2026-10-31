"""Parse a YAML suite file into typed, validated dataclasses.

This is the harness's entry point: everything a run needs (the dataset spec,
the ClickHouse table axes, the in-process engines' blocks (`polars:` / `datafusion:`, at
least one engine block per suite), the query axes, the results config) is parsed and
validated up front, so a malformed suite fails before anything touches the
generator or the database. `suites/` holds example suite files.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import pyarrow as pa
import yaml

from synthdata import DatasetSpec, Distribution, SpecError
from synthdata.column.realdata import path_spec
from chbench.harness.inproc_types import (
    DEFAULT_IPC_BATCH_ROWS,
    STORAGES,
    feed_error,
    normalise_ipc_batch_rows,
    normalise_ipc_compression,
    type_label,
)
from chbench.harness.validate import (
    BLOOM_FPRS,
    KNOWN_CODECS,
    KNOWN_TYPES,
    SKIP_GRANULARITIES,
    is_known_compression,
    is_known_skip_index,
)

_REQUIRED_BLOCKS = ("name", "dataset", "queries", "results")
_ENGINE_BLOCKS = ("clickhouse", "polars", "datafusion")
_TOP_LEVEL_BLOCKS = _REQUIRED_BLOCKS + _ENGINE_BLOCKS
_INPROC_AXES = ("type", "storage", "compression")
_INPROC_KEYS = set(_INPROC_AXES) | {"threads", "ipc_batch_rows"}
_QUERY_KINDS = ("full_scan", "filter")
_PREDICATES = ("range", "in")
_CLICKHOUSE_AXES = (
    "type",
    "nullable",
    "primary_key",
    "projection",
    "codec",
    "skip_index",
    "compression",
)
_DATASET_FIELDS = {f.name for f in fields(DatasetSpec)}
_PATH_DATASET_KEYS = {"path", "column", "seed", "decimal_places"}
_DISTRIBUTION_FIELDS = {f.name for f in fields(Distribution)}
_QUERIES_KEYS = {"kinds", "samples"}
# Retired by ToDo 3: every sample now runs warm with the query condition cache off.
_RETIRED_QUERY_KEYS = ("query_condition_cache", "cache_mode")
_RESULTS_KEYS = {"percentiles"}


class SuiteError(ValueError):
    """A suite file that cannot be parsed or is invalid. Names the field or
    block at fault."""


@dataclass(frozen=True)
class QueryKind:
    kind: str
    target_fraction: float | None
    predicate: str | None
    values: int | None


@dataclass(frozen=True)
class ClickHouseAxes:
    type: list[str]
    nullable: list[bool]
    primary_key: list[bool]
    projection: list[bool]
    codec: list[str]
    skip_index: list[str]
    compression: list[str]


@dataclass(frozen=True)
class InprocAxes:
    """A `polars:` or `datafusion:` block. `type` keeps the axis values as written (their labels:
    `inproc_types.type_label`), `compression` the stored spelling (`none`, `LZ4_FRAME`, `ZSTD(N)`),
    `threads` None for the default (the Docker VM's CPU count, else the host's)."""

    type: list[str]
    storage: list[str]
    compression: list[str]
    threads: list[int] | None
    ipc_batch_rows: list[str] = field(
        default_factory=lambda: [str(DEFAULT_IPC_BATCH_ROWS)]
    )


@dataclass(frozen=True)
class QueryAxes:
    kinds: list[QueryKind]
    samples: int


@dataclass(frozen=True)
class ResultsConfig:
    percentiles: list[int]


@dataclass(frozen=True)
class Suite:
    name: str
    dataset: DatasetSpec
    clickhouse: ClickHouseAxes | None
    queries: QueryAxes
    results: ResultsConfig
    source_yaml: str
    polars: InprocAxes | None = None
    datafusion: InprocAxes | None = None


def load_suite(path: str | Path) -> Suite:
    """Read and parse a suite YAML file from disk. A dataset `path` resolves
    relative to the suite file's directory."""
    text = Path(path).read_text(encoding="utf-8")
    return parse_suite(text, base_dir=Path(path).parent)


def parse_suite(text: str, base_dir: str | Path | None = None) -> Suite:
    """Parse suite YAML text into a validated `Suite`. `base_dir` (default the
    working directory) is what a dataset `path` resolves against."""
    doc = yaml.safe_load(text)
    if not isinstance(doc, dict):
        raise SuiteError(
            "suite must be a YAML mapping with top-level blocks " f"{_TOP_LEVEL_BLOCKS}"
        )

    for block in _REQUIRED_BLOCKS:
        if block not in doc:
            raise SuiteError(f"suite is missing the required top-level block {block!r}")
    _reject_unknown_keys(doc, set(_TOP_LEVEL_BLOCKS), "suite")
    if not any(block in doc for block in _ENGINE_BLOCKS):
        raise SuiteError(
            f"suite names no engine: give at least one of the blocks {list(_ENGINE_BLOCKS)}"
        )

    name = doc["name"]

    dataset = _parse_dataset(doc["dataset"], base_dir)
    clickhouse = None
    if "clickhouse" in doc:
        clickhouse = _parse_clickhouse(doc["clickhouse"])
        _check_dataset_against_type_axis(dataset, clickhouse)
    inproc = {}
    for engine in ("polars", "datafusion"):
        if engine in doc:
            inproc[engine] = _parse_inproc(doc[engine], engine)
            _check_dataset_against_inproc_axis(dataset, engine, inproc[engine])
    queries = _parse_queries(doc["queries"])
    results = _parse_results(doc["results"])

    return Suite(
        name=name,
        dataset=dataset,
        clickhouse=clickhouse,
        queries=queries,
        results=results,
        source_yaml=text,
        polars=inproc.get("polars"),
        datafusion=inproc.get("datafusion"),
    )


def _parse_path_dataset(block: dict, base_dir: str | Path | None) -> DatasetSpec:
    """`dataset: {path, column[, seed, decimal_places]}`: every other field is
    measured from the file, so any other key is an error naming it."""
    if "path" not in block or "column" not in block:
        raise SuiteError(
            "dataset needs path and column together; got "
            f"path={block.get('path')!r}, column={block.get('column')!r}"
        )
    extra = set(block) - _PATH_DATASET_KEYS
    if extra:
        raise SuiteError(
            f"dataset with a path reads its data from the file; key(s) "
            f"{sorted(extra)!r} do not apply (allowed: {sorted(_PATH_DATASET_KEYS)!r})"
        )
    path = os.path.abspath(
        Path(base_dir if base_dir is not None else os.getcwd()) / str(block["path"])
    )
    try:
        return path_spec(
            path,
            block["column"],
            seed=block.get("seed", 0),
            decimal_places=block.get("decimal_places"),
        )
    except SpecError as e:
        raise SuiteError(str(e)) from e


def _parse_dataset(block: dict, base_dir: str | Path | None = None) -> DatasetSpec:
    if not isinstance(block, dict):
        raise SuiteError(f"dataset block must be a mapping, got {block!r}")
    _reject_unknown_keys(block, _DATASET_FIELDS - {"content_hash"}, "dataset")
    if "path" in block or "column" in block:
        return _parse_path_dataset(block, base_dir)

    d = dict(block)

    if d.get("value_range") is not None:
        try:
            d["value_range"] = tuple(d["value_range"])
        except TypeError as e:
            raise SuiteError(
                f"dataset.value_range must be a list of two numbers, got "
                f"{d['value_range']!r}"
            ) from e

    if d.get("array_len_range") is not None:
        try:
            d["array_len_range"] = tuple(d["array_len_range"])
        except TypeError as e:
            raise SuiteError(
                f"dataset.array_len_range must be a list of two numbers, got "
                f"{d['array_len_range']!r}"
            ) from e

    if d.get("distribution") is not None:
        dist = d["distribution"]
        if not isinstance(dist, dict):
            raise SuiteError(f"dataset.distribution must be a mapping, got {dist!r}")
        unknown = set(dist) - _DISTRIBUTION_FIELDS
        if unknown:
            raise SuiteError(
                f"dataset.distribution has unknown key(s) {sorted(unknown)!r}"
            )
        try:
            d["distribution"] = Distribution(**dist)
        except (TypeError, ValueError) as e:
            raise SuiteError(f"dataset.distribution: {e}") from e

    try:
        spec = DatasetSpec(**d)
    except (TypeError, ValueError) as e:
        raise SuiteError(f"dataset block: {e}") from e

    try:
        spec.validate()
    except SpecError as e:
        raise SuiteError(str(e)) from e

    return spec


def _parse_clickhouse(block: dict) -> ClickHouseAxes:
    _reject_unknown_keys(block, set(_CLICKHOUSE_AXES), "clickhouse")
    for axis in _CLICKHOUSE_AXES:
        if axis not in block:
            raise SuiteError(f"clickhouse block is missing the {axis!r} axis")
        _require_non_empty_list(block[axis], axis)

    for value in block["type"]:
        if value not in KNOWN_TYPES:
            raise SuiteError(
                f"clickhouse.type has unmeasured value {value!r}; the type axis "
                f"vocabulary is exactly the base types measured against a live "
                f"server, {sorted(KNOWN_TYPES)!r}. Measure a new type in "
                f"spikes/spike_ch_vocabulary.py before adding it."
            )

    for value in block["codec"]:
        if value not in KNOWN_CODECS:
            raise SuiteError(
                f"clickhouse.codec has unknown value {value!r}; must be one of "
                f"{sorted(KNOWN_CODECS)!r}"
            )

    for value in block["skip_index"]:
        if not is_known_skip_index(value):
            raise SuiteError(
                f"clickhouse.skip_index has unknown value {value!r}; must be "
                f"'none', or 'minmax', 'bloom_filter', 'bloom_filter(F)' (F in "
                f"{list(BLOOM_FPRS)}) or 'set(N)' (N positive), each optionally "
                f"followed by ':G' with G in {list(SKIP_GRANULARITIES)}"
            )

    for value in block["compression"]:
        if not is_known_compression(value):
            raise SuiteError(
                f"clickhouse.compression has unknown value {value!r}; must be "
                f"'LZ4', 'ZSTD', or 'ZSTD(N)' for N from 1 to 22"
            )

    return ClickHouseAxes(
        type=block["type"],
        nullable=block["nullable"],
        primary_key=block["primary_key"],
        projection=block["projection"],
        codec=block["codec"],
        skip_index=block["skip_index"],
        compression=block["compression"],
    )


# The `type` axis values that `validate._build_variant` compares as numbers:
# `UInt*` against `value_min < 0`, the integers and `Decimal*` through the
# range-fit rule, and the rest through the same `cast_column` a numeric
# dataset takes. A string or a temporal dataset would reach those as a
# `str`/`datetime.date` and raise `TypeError` mid-run.
_NUMERIC_AXIS_TYPES = frozenset(
    {
        "Float64",
        "Float32",
        "Decimal32",
        "Decimal64",
        "Decimal128",
        "Int8",
        "Int16",
        "Int32",
        "Int64",
        "UInt8",
        "UInt16",
        "UInt32",
        "UInt64",
    }
)


def _is_integer_date_or_timestamp(dtype) -> bool:
    return (
        pa.types.is_integer(dtype)
        or pa.types.is_date32(dtype)
        or pa.types.is_timestamp(dtype)
    )


# Axis types (outside `_NUMERIC_AXIS_TYPES`) that only some dataset types can
# feed: (predicate on the Arrow type, what the dataset must be). The
# loader's casts (`loader.cast_column`) and `validate.resolve_type` rely on
# these.
_TYPED_AXIS_TYPES = {
    "Bool": (pa.types.is_boolean, "a boolean dataset"),
    "Enum8": (pa.types.is_string, "a string dataset"),
    "Enum16": (pa.types.is_string, "a string dataset"),
    "FixedString": (
        lambda t: pa.types.is_string(t) or pa.types.is_fixed_size_binary(t),
        "a string or fixed_size_binary dataset",
    ),
    "Time64": (
        lambda t: pa.types.is_time32(t) or pa.types.is_time64(t),
        "a time32 or time64 dataset",
    ),
    "Date32": (
        _is_integer_date_or_timestamp,
        "an integer, date32 or timestamp dataset",
    ),
    "DateTime": (
        _is_integer_date_or_timestamp,
        "an integer, date32 or timestamp dataset",
    ),
}


def _check_dataset_against_type_axis(
    dataset: DatasetSpec, clickhouse: ClickHouseAxes
) -> None:
    """Reject at parse time the dataset/`type`-axis pairs that no grid cell
    can run, rather than crash in `validate` or the query renderer after
    the dataset has been generated.

    Each block validates on its own; these are the checks that need both.
    `DatasetSpec.validate` already requires `decimal_places` for a float or
    decimal *dataset*, so the first check only bites an integer (or other)
    dataset routed to a `Decimal*` column. A numeric dataset on a `String`
    axis stays legal: that is the int suite, cast to `String` by the loader.
    The axis types whose parameters or cast need a particular dataset type
    (`Bool`, `Enum*`, `FixedString`, `Time64`, `Date32`, `DateTime`) are
    checked against `_TYPED_AXIS_TYPES`.
    """
    dtype = dataset.arrow_dtype
    if pa.types.is_list(dtype):
        raise SuiteError(
            f"dataset.arrow_type {dataset.arrow_type!r} is a list type, and no "
            f"clickhouse.type axis value can hold a list column; the harness "
            f"benchmarks one scalar column per table"
        )
    numeric_dataset = (
        pa.types.is_integer(dtype)
        or pa.types.is_floating(dtype)
        or pa.types.is_decimal(dtype)
    )
    for value in clickhouse.type:
        if value in _NUMERIC_AXIS_TYPES and not numeric_dataset:
            raise SuiteError(
                f"clickhouse.type value {value!r} is numeric and needs a numeric "
                f"dataset (an integer, float or decimal arrow_type), but "
                f"dataset.arrow_type is {dataset.arrow_type!r}"
            )
        if (
            value in ("Decimal32", "Decimal64", "Decimal128")
            and dataset.decimal_places is None
        ):
            raise SuiteError(
                f"clickhouse.type value {value!r} needs dataset.decimal_places "
                f"(its scale), which is not set for arrow_type "
                f"{dataset.arrow_type!r}"
            )
        if value in _TYPED_AXIS_TYPES:
            accepts, need = _TYPED_AXIS_TYPES[value]
            if not accepts(dtype):
                raise SuiteError(
                    f"clickhouse.type value {value!r} needs {need}, but "
                    f"dataset.arrow_type is {dataset.arrow_type!r}"
                )


def _parse_inproc(block, engine: str) -> InprocAxes:
    """A `polars:` / `datafusion:` block: `type`, `storage`, `compression` (non-empty lists) and an
    optional `threads` (positive integers) and `ipc_batch_rows` (positive integers, by default
    122 880, or `whole`; it applies to `storage: ipc` cells only). Every error names
    `<engine>.<axis>`."""
    if not isinstance(block, dict):
        raise SuiteError(f"{engine} block must be a mapping, got {block!r}")
    _reject_unknown_keys(block, _INPROC_KEYS, engine)
    for axis in _INPROC_AXES:
        if axis not in block:
            raise SuiteError(f"{engine} block is missing the {axis!r} axis")
        _require_non_empty_list(block[axis], f"{engine}.{axis}")

    labels = []
    for value in block["type"]:
        try:
            labels.append(type_label(engine, value))
        except ValueError as e:
            raise SuiteError(f"{engine}.type: {e}") from e
    if len(set(labels)) != len(labels):
        raise SuiteError(
            f"{engine}.type names one type twice (as labelled: {labels!r}); give each once"
        )

    for value in block["storage"]:
        if value not in STORAGES:
            raise SuiteError(
                f"{engine}.storage has unknown value {value!r}; must be one of {list(STORAGES)}"
            )

    try:
        compression = list(
            dict.fromkeys(normalise_ipc_compression(v) for v in block["compression"])
        )
    except ValueError as e:
        raise SuiteError(f"{engine}.compression: {e}") from e

    threads = block.get("threads")
    if threads is not None:
        _require_non_empty_list(threads, f"{engine}.threads")
        for t in threads:
            if not isinstance(t, int) or isinstance(t, bool) or t <= 0:
                raise SuiteError(
                    f"{engine}.threads must hold positive integers, got {t!r}"
                )

    batch_rows = block.get("ipc_batch_rows", [DEFAULT_IPC_BATCH_ROWS])
    _require_non_empty_list(batch_rows, f"{engine}.ipc_batch_rows")
    try:
        batch_rows = list(
            dict.fromkeys(normalise_ipc_batch_rows(v) for v in batch_rows)
        )
    except ValueError as e:
        raise SuiteError(f"{engine}.ipc_batch_rows: {e}") from e

    return InprocAxes(
        type=[str(v) for v in block["type"]],
        storage=list(dict.fromkeys(block["storage"])),
        compression=compression,
        threads=list(dict.fromkeys(threads)) if threads is not None else None,
        ipc_batch_rows=batch_rows,
    )


def _check_dataset_against_inproc_axis(
    dataset: DatasetSpec, engine: str, axes: InprocAxes
) -> None:
    """Reject at parse time an in-process `type` value the dataset cannot feed
    (`inproc_types.FEEDS`: a list dataset, a Categorical over numbers, a number over strings). A cast
    that is allowed but fails, or does not survive the round trip, is a skipped cell at run time.
    """
    for value in axes.type:
        error = feed_error(dataset.arrow_dtype, engine, value)
        if error is not None:
            raise SuiteError(error)


def _parse_queries(block: dict) -> QueryAxes:
    for key in _RETIRED_QUERY_KEYS:
        if key in block:
            raise SuiteError(
                f"queries.{key} was retired (ToDo 3): every sample runs warm with the "
                f"query condition cache off; remove the key"
            )
    _reject_unknown_keys(block, _QUERIES_KEYS, "queries")

    if "kinds" not in block:
        raise SuiteError("queries block is missing the 'kinds' axis")
    _require_non_empty_list(block["kinds"], "kinds")

    if "samples" not in block:
        raise SuiteError("queries block is missing 'samples'")

    samples = block["samples"]
    if not isinstance(samples, int) or isinstance(samples, bool) or samples <= 0:
        raise SuiteError(f"samples must be a positive integer, got {samples!r}")

    return QueryAxes(
        kinds=[_parse_query_kind(k) for k in block["kinds"]], samples=samples
    )


def _parse_query_kind(k: dict) -> QueryKind:
    if not isinstance(k, dict) or "kind" not in k:
        raise SuiteError("each query kind must be a mapping with a 'kind' field")

    kind = k["kind"]
    if kind not in _QUERY_KINDS:
        raise SuiteError(f"query kind must be one of {_QUERY_KINDS}, got {kind!r}")

    target_fraction = k.get("target_fraction")
    predicate = k.get("predicate")
    values = k.get("values")

    if target_fraction is not None and not (0 < target_fraction < 1):
        raise SuiteError(
            f"target_fraction must be strictly between 0 and 1, got {target_fraction!r}"
        )

    if kind == "filter":
        if target_fraction is None:
            raise SuiteError("filter query kind requires 'target_fraction'")
        if predicate is None:
            raise SuiteError("filter query kind requires 'predicate'")
        if predicate not in _PREDICATES:
            raise SuiteError(
                f"predicate must be one of {_PREDICATES}, got {predicate!r}"
            )
        if predicate == "in" and values is None:
            raise SuiteError("predicate 'in' requires 'values'")

    return QueryKind(
        kind=kind, target_fraction=target_fraction, predicate=predicate, values=values
    )


def _parse_results(block: dict) -> ResultsConfig:
    _reject_unknown_keys(block, _RESULTS_KEYS, "results")

    if "percentiles" not in block:
        raise SuiteError("results block is missing 'percentiles'")
    percentiles = block["percentiles"]
    _require_non_empty_list(percentiles, "percentiles")
    for p in percentiles:
        if not (0 <= p <= 100):
            raise SuiteError(f"percentiles must be between 0 and 100, got {p!r}")
    return ResultsConfig(percentiles=percentiles)


def _require_non_empty_list(value: Any, name: str) -> None:
    if not isinstance(value, list) or len(value) == 0:
        raise SuiteError(f"{name!r} must be a non-empty list")


def _reject_unknown_keys(block: dict, allowed: set[str], name: str) -> None:
    unknown = set(block) - allowed
    if unknown:
        raise SuiteError(f"{name} has unknown key(s) {sorted(unknown)!r}")
