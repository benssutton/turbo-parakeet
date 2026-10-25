"""Parse a YAML suite file into typed, validated dataclasses.

This is the harness's entry point: everything a run needs (the dataset spec,
the ClickHouse table axes, the query axes, the results config) is parsed and
validated up front, so a malformed suite fails before anything touches the
generator or the database. See spec section 5.1 for the suite YAML shape and
`docs/superpowers/specs/2026-09-13-clickhouse-benchmark-framework-design.md`
for the design.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import pyarrow as pa
import yaml

from chbench.datagen import DatasetSpec, Distribution, SpecError
from chbench.datagen.realdata import path_spec
from chbench.harness.validate import (
    BLOOM_FPRS, KNOWN_CODECS, KNOWN_TYPES, SKIP_GRANULARITIES, is_known_compression,
    is_known_skip_index,
)

_TOP_LEVEL_BLOCKS = ("name", "dataset", "clickhouse", "queries", "results")
_QUERY_KINDS = ("full_scan", "filter")
_PREDICATES = ("range", "in")
_CACHE_MODES = {"warm", "cold"}
_CLICKHOUSE_AXES = ("type", "nullable", "primary_key", "projection", "codec",
                    "skip_index", "compression")
_DATASET_FIELDS = {f.name for f in fields(DatasetSpec)}
_PATH_DATASET_KEYS = {"path", "column", "seed", "decimal_places"}
_DISTRIBUTION_FIELDS = {f.name for f in fields(Distribution)}
_QUERIES_KEYS = {"kinds", "query_condition_cache", "cache_mode", "samples"}
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
class QueryAxes:
    kinds: list[QueryKind]
    query_condition_cache: list[bool]
    cache_mode: list[str]
    samples: int


@dataclass(frozen=True)
class ResultsConfig:
    percentiles: list[int]


@dataclass(frozen=True)
class Suite:
    name: str
    dataset: DatasetSpec
    clickhouse: ClickHouseAxes
    queries: QueryAxes
    results: ResultsConfig
    source_yaml: str


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
        raise SuiteError("suite must be a YAML mapping with top-level blocks "
                          f"{_TOP_LEVEL_BLOCKS}")

    for block in _TOP_LEVEL_BLOCKS:
        if block not in doc:
            raise SuiteError(f"suite is missing the required top-level block {block!r}")
    _reject_unknown_keys(doc, set(_TOP_LEVEL_BLOCKS), "suite")

    name = doc["name"]

    dataset = _parse_dataset(doc["dataset"], base_dir)
    clickhouse = _parse_clickhouse(doc["clickhouse"])
    _check_dataset_against_type_axis(dataset, clickhouse)
    queries = _parse_queries(doc["queries"])
    results = _parse_results(doc["results"])

    return Suite(name=name, dataset=dataset, clickhouse=clickhouse,
                 queries=queries, results=results, source_yaml=text)


def _parse_path_dataset(block: dict, base_dir: str | Path | None) -> DatasetSpec:
    """`dataset: {path, column[, seed, decimal_places]}` (spec section 5.1,
    ruling R35): every other field is measured from the file, so any other key
    is an error naming it."""
    if "path" not in block or "column" not in block:
        raise SuiteError("dataset needs path and column together; got "
                         f"path={block.get('path')!r}, column={block.get('column')!r}")
    extra = set(block) - _PATH_DATASET_KEYS
    if extra:
        raise SuiteError(
            f"dataset with a path reads its data from the file; key(s) "
            f"{sorted(extra)!r} do not apply (allowed: {sorted(_PATH_DATASET_KEYS)!r})")
    path = os.path.abspath(Path(base_dir if base_dir is not None else os.getcwd())
                           / str(block["path"]))
    try:
        return path_spec(path, block["column"], seed=block.get("seed", 0),
                         decimal_places=block.get("decimal_places"))
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
            raise SuiteError(
                f"dataset.distribution must be a mapping, got {dist!r}"
            )
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
_NUMERIC_AXIS_TYPES = frozenset({
    "Float64", "Float32", "Decimal32", "Decimal64", "Decimal128",
    "Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64",
})


def _is_integer_date_or_timestamp(dtype) -> bool:
    return (pa.types.is_integer(dtype) or pa.types.is_date32(dtype)
            or pa.types.is_timestamp(dtype))


# Axis types (outside `_NUMERIC_AXIS_TYPES`) that only some dataset types can
# feed: (predicate on the Arrow type, what the dataset must be). Ruling R36;
# the loader's casts (`loader.cast_column`) and `validate.resolve_type` rely on
# these.
_TYPED_AXIS_TYPES = {
    "Bool": (pa.types.is_boolean, "a boolean dataset"),
    "Enum8": (pa.types.is_string, "a string dataset"),
    "Enum16": (pa.types.is_string, "a string dataset"),
    "FixedString": (lambda t: pa.types.is_string(t) or pa.types.is_fixed_size_binary(t),
                    "a string or fixed_size_binary dataset"),
    "Time64": (lambda t: pa.types.is_time32(t) or pa.types.is_time64(t),
               "a time32 or time64 dataset"),
    "Date32": (_is_integer_date_or_timestamp, "an integer, date32 or timestamp dataset"),
    "DateTime": (_is_integer_date_or_timestamp, "an integer, date32 or timestamp dataset"),
}


def _check_dataset_against_type_axis(dataset: DatasetSpec,
                                     clickhouse: ClickHouseAxes) -> None:
    """Reject at parse time the dataset/`type`-axis pairs that no grid cell
    can run, rather than crash in `validate` or the query renderer after
    the dataset has been generated (ruling R31c).

    Each block validates on its own; these are the checks that need both.
    `DatasetSpec.validate` already requires `decimal_places` for a float or
    decimal *dataset*, so the first check only bites an integer (or other)
    dataset routed to a `Decimal*` column. A numeric dataset on a `String`
    axis stays legal: that is the int suite, cast to `String` by the loader.
    The axis types whose parameters or cast need a particular dataset type
    (`Bool`, `Enum*`, `FixedString`, `Time64`, `Date32`, `DateTime`) are
    checked against `_TYPED_AXIS_TYPES` (ruling R36).
    """
    dtype = dataset.arrow_dtype
    if pa.types.is_list(dtype):
        raise SuiteError(
            f"dataset.arrow_type {dataset.arrow_type!r} is a list type, and no "
            f"clickhouse.type axis value can hold a list column; the harness "
            f"benchmarks one scalar column per table")
    numeric_dataset = (pa.types.is_integer(dtype) or pa.types.is_floating(dtype)
                       or pa.types.is_decimal(dtype))
    for value in clickhouse.type:
        if value in _NUMERIC_AXIS_TYPES and not numeric_dataset:
            raise SuiteError(
                f"clickhouse.type value {value!r} is numeric and needs a numeric "
                f"dataset (an integer, float or decimal arrow_type), but "
                f"dataset.arrow_type is {dataset.arrow_type!r}")
        if value in ("Decimal32", "Decimal64", "Decimal128") and dataset.decimal_places is None:
            raise SuiteError(
                f"clickhouse.type value {value!r} needs dataset.decimal_places "
                f"(its scale), which is not set for arrow_type "
                f"{dataset.arrow_type!r}")
        if value in _TYPED_AXIS_TYPES:
            accepts, need = _TYPED_AXIS_TYPES[value]
            if not accepts(dtype):
                raise SuiteError(
                    f"clickhouse.type value {value!r} needs {need}, but "
                    f"dataset.arrow_type is {dataset.arrow_type!r}")


def _parse_queries(block: dict) -> QueryAxes:
    _reject_unknown_keys(block, _QUERIES_KEYS, "queries")

    for axis in ("kinds", "query_condition_cache", "cache_mode"):
        if axis not in block:
            raise SuiteError(f"queries block is missing the {axis!r} axis")
        _require_non_empty_list(block[axis], axis)

    if "samples" not in block:
        raise SuiteError("queries block is missing 'samples'")

    samples = block["samples"]
    if not isinstance(samples, int) or isinstance(samples, bool) or samples <= 0:
        raise SuiteError(f"samples must be a positive integer, got {samples!r}")

    kinds = [_parse_query_kind(k) for k in block["kinds"]]

    cache_mode = block["cache_mode"]
    for mode in cache_mode:
        if mode not in _CACHE_MODES:
            raise SuiteError(
                f"cache_mode must be 'warm' or 'cold', got {mode!r}"
            )

    return QueryAxes(
        kinds=kinds,
        query_condition_cache=block["query_condition_cache"],
        cache_mode=cache_mode,
        samples=samples,
    )


def _parse_query_kind(k: dict) -> QueryKind:
    if not isinstance(k, dict) or "kind" not in k:
        raise SuiteError("each query kind must be a mapping with a 'kind' field")

    kind = k["kind"]
    if kind not in _QUERY_KINDS:
        raise SuiteError(
            f"query kind must be one of {_QUERY_KINDS}, got {kind!r}"
        )

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

    return QueryKind(kind=kind, target_fraction=target_fraction,
                      predicate=predicate, values=values)


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
