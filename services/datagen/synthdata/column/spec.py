"""The generator's input: a validated, hashable description of one dataset.

This is the only module that raises on bad input. Everything downstream is
best-effort (a missed target warns, never fails), so every malformed
combination must be caught here, with a message naming the fields involved.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import re
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Sequence

import numpy as np
import pyarrow as pa


class SpecError(ValueError):
    """A DatasetSpec that cannot be generated. The only error the generator raises."""


_SCALARS = {
    "int8": pa.int8,
    "int16": pa.int16,
    "int32": pa.int32,
    "int64": pa.int64,
    "uint8": pa.uint8,
    "uint16": pa.uint16,
    "uint32": pa.uint32,
    "uint64": pa.uint64,
    "float32": pa.float32,
    "float64": pa.float64,
    "bool": pa.bool_,
    "string": pa.string,
    "date32": pa.date32,
}
_DECIMALS = {
    "decimal32": (pa.decimal32, 9),
    "decimal64": (pa.decimal64, 18),
    "decimal128": (pa.decimal128, 38),
}
_TIME_UNITS = {"s", "ms", "us", "ns"}
_TIME32_UNITS = {"s", "ms"}
_TIME64_UNITS = {"us", "ns"}

_DECIMAL_RE = re.compile(r"^(decimal32|decimal64|decimal128)\((\d+),(\d+)\)$")
_TIMESTAMP_RE = re.compile(r"^timestamp\[(\w+)(?:, tz=([^\]]+))?\]$")
_FIXED_BINARY_RE = re.compile(r"^fixed_size_binary\((\d+)\)$")
_TIME32_RE = re.compile(r"^time32\[(\w+)\]$")
_TIME64_RE = re.compile(r"^time64\[(\w+)\]$")
_LIST_RE = re.compile(r"^list<(.+)>$")


def parse_arrow_type(s: str) -> pa.DataType:
    """Parse one of the supported `arrow_type` strings. Raises SpecError
    otherwise.

    Decimal precision limits are per width: 9, 18 and 38. list<T> nests exactly
    one
    scalar; list of list is not supported.
    """
    return _parse(s, allow_list=True)


def _parse(s: str, allow_list: bool) -> pa.DataType:
    if s in _SCALARS:
        return _SCALARS[s]()

    m = _DECIMAL_RE.match(s)
    if m:
        name, precision_s, scale_s = m.group(1), m.group(2), m.group(3)
        ctor, max_precision = _DECIMALS[name]
        precision, scale = int(precision_s), int(scale_s)
        if not (1 <= precision <= max_precision):
            raise SpecError(
                f"{name} precision must be between 1 and {max_precision}, "
                f"got {precision}"
            )
        try:
            return ctor(precision, scale)
        except Exception as e:  # pragma: no cover - pyarrow validates scale etc.
            raise SpecError(f"invalid {name}({precision},{scale}): {e}") from e

    m = _TIMESTAMP_RE.match(s)
    if m:
        unit = m.group(1)
        if unit not in _TIME_UNITS:
            raise SpecError(f"unsupported timestamp unit {unit!r} in {s!r}")
        return pa.timestamp(unit, tz=m.group(2))

    m = _FIXED_BINARY_RE.match(s)
    if m:
        if int(m.group(1)) < 1:
            raise SpecError(f"fixed_size_binary width must be positive in {s!r}")
        return pa.binary(int(m.group(1)))

    m = _TIME32_RE.match(s)
    if m:
        unit = m.group(1)
        if unit not in _TIME32_UNITS:
            raise SpecError(f"unsupported time32 unit {unit!r} in {s!r}")
        return pa.time32(unit)

    m = _TIME64_RE.match(s)
    if m:
        unit = m.group(1)
        if unit not in _TIME64_UNITS:
            raise SpecError(f"unsupported time64 unit {unit!r} in {s!r}")
        return pa.time64(unit)

    if allow_list:
        m = _LIST_RE.match(s)
        if m:
            inner = _parse(m.group(1), allow_list=False)
            return pa.list_(inner)

    raise SpecError(f"unsupported arrow_type {s!r}")


_INT_BOUNDS: dict[str, tuple[int, int]] = {
    "int8": (-(2**7), 2**7 - 1),
    "int16": (-(2**15), 2**15 - 1),
    "int32": (-(2**31), 2**31 - 1),
    "int64": (-(2**63), 2**63 - 1),
    "uint8": (0, 2**8 - 1),
    "uint16": (0, 2**16 - 1),
    "uint32": (0, 2**32 - 1),
    "uint64": (0, 2**64 - 1),
}
_PER_SECOND = {"s": 1, "ms": 10**3, "us": 10**6, "ns": 10**9}


def representable_range(dtype: pa.DataType) -> tuple[float, float] | None:
    """The closed interval of values a scalar type can hold, in the units the
    generator draws in (integers in the unit's resolution for temporals), or
    None for types with no meaningful bound (string, float64).

    Decimals hold `10**(precision - scale) - 10**-scale` in magnitude; time32
    and time64 hold one day in their unit; date32 is days as int32; timestamps
    are int64 in their unit; bool is the lattice {0, 1}; float32 is bounded by
    its largest finite value (its precision limit is a separate rule).
    """
    if pa.types.is_integer(dtype):
        return _INT_BOUNDS[str(dtype)]
    if pa.types.is_decimal(dtype):
        bound = 10 ** (dtype.precision - dtype.scale) - 10 ** (-dtype.scale)
        return (-bound, bound)
    if pa.types.is_boolean(dtype):
        return (0, 1)
    if pa.types.is_date32(dtype):
        return _INT_BOUNDS["int32"]
    if pa.types.is_time(dtype):
        return (0, 86_400 * _PER_SECOND[dtype.unit] - 1)
    if pa.types.is_timestamp(dtype):
        return _INT_BOUNDS["int64"]
    if pa.types.is_float32(dtype):
        return (-3.4028234663852886e38, 3.4028234663852886e38)
    return None


def _is_number(c: Any, elem_dtype: pa.DataType) -> bool:
    """A numeric category for `elem_dtype`. bool counts as a number only for the
    boolean type, which draws its categories as 0 / 1."""
    if isinstance(c, bool):
        return pa.types.is_boolean(elem_dtype)
    return isinstance(c, numbers.Real)


def _jsonable(o):
    """json.dumps fallback: numpy scalars (a category list built from an array)
    become Python scalars, other sequences become lists."""
    if hasattr(o, "item") and callable(o.item):
        return o.item()
    return list(o)


@dataclass(frozen=True)
class Distribution:
    kind: str = "uniform"  # uniform | normal | lognormal
    mean: float | None = None  # lognormal: mean of the variate BEFORE shift
    sigma: float | None = None  # None means "solve for index_of_dispersion"
    shift: float = 0.0


@dataclass(frozen=True)
class Tolerances:
    """Defaults are measured values (FINDINGS §6); do not invent new ones."""

    entropy_bits: float = 0.01
    cardinality_ratio: float = 0.001
    index_of_dispersion: float = 0.01
    runs_z: float = 0.01
    interarrival_cv_relative: float = 0.10


# DatasetSpec fields that are generation recipes, not measurements.
_GENERATOR_FIELDS = (
    "categories",
    "string_length",
    "distribution",
    "cardinality_ratio",
    "entropy_bits",
    "index_of_dispersion",
    "runs_z",
    "interarrival_cv",
    "array_len_range",
    "array_null_count",
    "tolerances",
)


@dataclass(frozen=True)
class DatasetSpec:
    rows: int
    arrow_type: str
    null_count: int
    seed: int
    value_range: tuple[float, float] | None = None
    categories: Sequence[Any] | None = None
    string_length: int | None = None
    decimal_places: int | None = None
    distribution: Distribution = field(default_factory=Distribution)
    cardinality_ratio: float | None = None
    entropy_bits: float | None = None
    index_of_dispersion: float | None = None
    runs_z: float | None = None
    interarrival_cv: float | None = None
    array_len_range: tuple[int, int] | None = None
    array_null_count: int = 0
    tolerances: Tolerances = field(default_factory=Tolerances)
    # Real-data source: when `path` is set the other fields are measured from
    # the file (`realdata.path_spec`), not generation recipes; `content_hash`
    # pins the file's bytes and column.
    path: str | None = None
    column: str | None = None
    content_hash: str | None = None

    def validate(self) -> None:
        """Raise SpecError on any combination that cannot be generated."""
        if self.rows <= 0:
            raise SpecError(f"rows must be positive, got {self.rows}")
        if not (0 <= self.null_count <= self.rows):
            raise SpecError(
                f"null_count must be between 0 and rows ({self.rows}), "
                f"got {self.null_count}"
            )

        dtype = parse_arrow_type(self.arrow_type)
        if self.path is not None or self.column is not None:
            self._validate_path_dataset(dtype)
            return
        if pa.types.is_fixed_size_binary(dtype) or (
            pa.types.is_timestamp(dtype) and dtype.tz is not None
        ):
            raise SpecError(
                f"arrow_type {self.arrow_type!r} (a time zone or fixed_size_binary) "
                "is only available to a path dataset; the "
                "generator draws no such values"
            )
        if self.content_hash is not None:
            raise SpecError(
                "content_hash belongs to a path dataset; set path and column"
            )
        is_list = pa.types.is_list(dtype)
        elem_dtype = dtype.value_type if is_list else dtype

        sources = (self.value_range, self.categories, self.string_length)
        if sum(x is not None for x in sources) != 1:
            raise SpecError(
                "exactly one of value_range, categories or string_length must be set"
            )

        if self.categories is not None and len(self.categories) == 0:
            raise SpecError("categories must contain at least one value")
        if self.categories is not None and len(self.categories) > self.rows:
            # Every support value must appear at least once, so the support
            # cannot be larger than the row count. The cardinality_ratio path
            # is already safe because the ratio is bounded by 1.
            raise SpecError(
                f"categories has {len(self.categories)} values but rows is "
                f"{self.rows}; every category must appear at least once"
            )

        if self.categories is not None:
            is_num = [_is_number(c, elem_dtype) for c in self.categories]
            is_str = [isinstance(c, str) for c in self.categories]
            if pa.types.is_string(elem_dtype) and not all(is_str):
                raise SpecError(
                    f"categories for the string arrow_type {self.arrow_type!r} "
                    "must all be strings"
                )
            if not pa.types.is_string(elem_dtype) and not all(is_num):
                raise SpecError(
                    f"categories for the numeric arrow_type {self.arrow_type!r} "
                    "must all be numeric (bool is accepted only for the bool type)"
                )
            bounds = representable_range(elem_dtype)
            if bounds is not None and all(is_num):
                outside = [c for c in self.categories if c < bounds[0] or c > bounds[1]]
                if outside:
                    raise SpecError(
                        f"categories value {outside[0]!r} does not fit "
                        f"arrow_type {self.arrow_type!r}, whose representable "
                        f"range is [{bounds[0]}, {bounds[1]}]"
                    )
            fractional = pa.types.is_floating(elem_dtype) or pa.types.is_decimal(
                elem_dtype
            )
            if all(is_num) and not (fractional and self.decimal_places is None):
                # The metrics are measured on the categories as given and
                # `wrap` rounds afterwards, so a category off the grid would
                # make the manifest's cardinality overstate the array's.
                # Lattice types (integers, temporals, bool) have no
                # decimal_places and round to the integer grid. A fractional
                # type with no decimal_places is rejected further down.
                places = self.decimal_places if self.decimal_places is not None else 0
                off_grid = [c for c in self.categories if round(c, places) != c]
                if off_grid:
                    raise SpecError(
                        f"categories value {off_grid[0]!r} is not on the "
                        f"decimal_places grid ({places} places) that "
                        f"arrow_type {self.arrow_type!r} values are rounded "
                        f"to before casting; it would collapse onto "
                        f"{round(off_grid[0], places)!r} and the manifest "
                        "would overstate the cardinality"
                    )

        if self.value_range is not None:
            lo, hi = self.value_range
            if not (math.isfinite(lo) and math.isfinite(hi)):
                raise SpecError(f"value_range {self.value_range} bounds must be finite")
            if lo > hi:
                raise SpecError(f"value_range {self.value_range} must have low <= high")
            if pa.types.is_unsigned_integer(elem_dtype) and lo < 0:
                raise SpecError(
                    f"unsigned arrow_type {self.arrow_type!r} cannot hold a "
                    f"negative value_range low bound {lo}; it does not fit"
                )
            if pa.types.is_boolean(elem_dtype) and not (0 <= lo and hi <= 1):
                raise SpecError(
                    f"bool arrow_type draws on the lattice [0, 1]; value_range "
                    f"{self.value_range} lies outside it and does not fit"
                )
            bounds = representable_range(elem_dtype)
            if bounds is not None:
                # The generator rounds to the lattice (or to decimal_places)
                # before casting, so check the bounds as the cast will see them.
                places = self.decimal_places if self.decimal_places is not None else 0
                r_lo, r_hi = round(lo, places), round(hi, places)
                if r_lo < bounds[0] or r_hi > bounds[1]:
                    unit_note = (
                        " (integers in the unit's resolution)"
                        if pa.types.is_temporal(elem_dtype)
                        else ""
                    )
                    raise SpecError(
                        f"value_range {self.value_range} does not fit arrow_type "
                        f"{self.arrow_type!r}, whose representable range is "
                        f"[{bounds[0]}, {bounds[1]}]{unit_note}"
                    )

        if self.string_length is not None and self.string_length <= 0:
            raise SpecError(f"string_length must be positive, got {self.string_length}")

        if pa.types.is_floating(elem_dtype) or pa.types.is_decimal(elem_dtype):
            if self.decimal_places is None or self.decimal_places < 0:
                raise SpecError(
                    f"decimal_places must be set and non-negative for "
                    f"arrow_type {self.arrow_type!r}, got {self.decimal_places}"
                )
            if (
                pa.types.is_decimal(elem_dtype)
                and self.decimal_places != elem_dtype.scale
            ):
                raise SpecError(
                    f"decimal_places ({self.decimal_places}) must equal the "
                    f"scale of arrow_type {self.arrow_type!r} "
                    f"({elem_dtype.scale})"
                )
            if pa.types.is_float32(elem_dtype) and self.value_range is not None:
                lo, hi = self.value_range
                grid_points = max(abs(lo), abs(hi)) * 10**self.decimal_places
                if grid_points > 2**24:
                    raise SpecError(
                        f"float32 cannot represent {self.decimal_places} "
                        f"decimal places over value_range {self.value_range}: "
                        f"it has 24 significant bits, and max(|lo|, |hi|) * "
                        f"10**decimal_places = {grid_points:.4g} exceeds "
                        f"2**24 = {2**24}; values drawn at that precision "
                        "collapse when cast, so the manifest would overstate "
                        "the cardinality. Reduce decimal_places, narrow the "
                        "range, or use float64"
                    )

        for name in (
            "cardinality_ratio",
            "entropy_bits",
            "index_of_dispersion",
            "runs_z",
            "interarrival_cv",
        ):
            target = getattr(self, name)
            if target is not None and not math.isfinite(target):
                raise SpecError(f"{name} must be finite, got {target}")

        if self.cardinality_ratio is not None:
            if not (0 < self.cardinality_ratio <= 1):
                raise SpecError(
                    f"cardinality_ratio must be in (0, 1], got "
                    f"{self.cardinality_ratio}"
                )
            if round(self.cardinality_ratio * self.rows) < 1:
                raise SpecError(
                    f"cardinality_ratio {self.cardinality_ratio} yields fewer "
                    f"than one support value for rows={self.rows}"
                )

        if self.entropy_bits is not None:
            if self.cardinality_ratio is None and self.categories is None:
                raise SpecError(
                    "entropy_bits requires cardinality_ratio or categories to "
                    "define a finite support set"
                )

        if self.distribution.kind not in ("uniform", "normal", "lognormal"):
            raise SpecError(
                f"distribution.kind must be one of uniform, normal or "
                f"lognormal, got {self.distribution.kind!r}"
            )
        if self.distribution.sigma is not None and not (self.distribution.sigma > 0):
            raise SpecError(
                f"distribution.sigma must be strictly positive when set, got "
                f"{self.distribution.sigma}"
            )
        if self.distribution.kind == "lognormal":
            if self.distribution.mean is None or self.distribution.mean <= 0:
                raise SpecError(
                    f"lognormal distribution requires a positive mean, got "
                    f"{self.distribution.mean}"
                )
        numeric_support = self.string_length is None and (
            self.categories is None
            or all(_is_number(c, elem_dtype) for c in self.categories)
        )
        if self.index_of_dispersion is not None and not numeric_support:
            raise SpecError(
                "index_of_dispersion is var/mean of the values, which needs a "
                "numeric support; string_length and string categories cannot "
                "target it"
            )
        if self.index_of_dispersion is not None:
            if self.entropy_bits is not None:
                raise SpecError(
                    "entropy_bits together with index_of_dispersion is not "
                    "supported because the profile solve and the spread solve "
                    "interact: the entropy target shuffles the rank-to-value "
                    "assignment, which makes the dispersion objective "
                    "non-monotone in sigma"
                )
            if self.distribution.kind == "uniform":
                raise SpecError(
                    "index_of_dispersion is solved by bisection on the "
                    "distribution's spread parameter, and the uniform "
                    "distribution has none; use kind='normal' or 'lognormal'"
                )
            if self.distribution.sigma is not None:
                raise SpecError(
                    f"distribution.sigma ({self.distribution.sigma}) fixes the "
                    "spread while index_of_dispersion asks to solve for it; "
                    "set one or the other"
                )
        if self.distribution.kind in ("normal", "lognormal"):
            if self.distribution.sigma is None and self.index_of_dispersion is None:
                raise SpecError(
                    f"{self.distribution.kind} distribution has no spread: set "
                    "distribution.sigma or index_of_dispersion (the dispersion "
                    "solver then chooses sigma); one of the two is needed"
                )
            if not numeric_support:
                raise SpecError(
                    f"{self.distribution.kind} distribution profiles the support "
                    "by density, which needs a numeric support; string_length "
                    "and string categories take the uniform distribution (set "
                    "entropy_bits to shape their frequencies)"
                )
            if (
                self.distribution.kind == "normal"
                and self.distribution.mean is None
                and self.value_range is None
            ):
                raise SpecError(
                    "normal distribution with no mean is centred on value_range, "
                    "which is not set: set distribution.mean or value_range"
                )

        if (
            pa.types.is_uint64(elem_dtype)
            and self.value_range is not None
            and self.value_range[1] > 2**63 - 1
            and (self.cardinality_ratio is None or self.distribution.kind != "uniform")
        ):
            raise SpecError(
                f"uint64 values above 2**63 - 1 (value_range {self.value_range}) "
                "are drawn only on the uniform lattice path, which needs "
                "cardinality_ratio and the uniform distribution; the uncapped and "
                "the non-uniform draws pass through float64 and int64, which "
                "cannot hold them. Set cardinality_ratio with the uniform "
                "distribution, or lower value_range's high bound to 2**63 - 1"
            )

        if self.interarrival_cv is not None and self.interarrival_cv <= 0:
            raise SpecError(
                f"interarrival_cv must be positive, got {self.interarrival_cv}"
            )

        if is_list and self.array_len_range is None:
            raise SpecError(
                f"array_len_range is required for the list arrow_type "
                f"{self.arrow_type!r}: it sets the inner value count "
                "(rows * round(mean(array_len_range)))"
            )
        if (self.array_len_range is not None or self.array_null_count) and not is_list:
            raise SpecError(
                "array_len_range and array_null_count require a list<T> arrow_type"
            )

        if self.array_null_count < 0:
            raise SpecError(
                f"array_null_count must be non-negative, got "
                f"{self.array_null_count}"
            )
        if self.array_len_range is not None:
            lo_len, hi_len = self.array_len_range
            if lo_len > hi_len:
                raise SpecError(
                    f"array_len_range {self.array_len_range} must have low <= high"
                )
            if lo_len <= 0:
                raise SpecError(
                    f"array_len_range {self.array_len_range} bounds must be " "positive"
                )
            mean_len = int(round((lo_len + hi_len) / 2))
            if mean_len < 1:
                raise SpecError(
                    f"array_len_range {self.array_len_range} rounds to a mean "
                    "length below 1, so no inner values would be generated"
                )
            inner = self.rows * mean_len
            if self.array_null_count > inner:
                raise SpecError(
                    f"array_null_count {self.array_null_count} exceeds the "
                    f"inner value count rows ({self.rows}) * round(mean("
                    f"array_len_range)) ({mean_len}) = {inner}"
                )

        if (
            pa.types.is_string(dtype)
            and self.string_length is None
            and self.categories is None
        ):
            raise SpecError(
                "string arrow_type requires string_length or categories to be set"
            )

    def _validate_path_dataset(self, dtype: pa.DataType) -> None:
        """A path dataset carries only measured
        fields plus seed and decimal_places; any generation field is an error."""
        if self.path is None or self.column is None:
            raise SpecError(
                "path and column must be set together: "
                f"path={self.path!r}, column={self.column!r}"
            )
        if self.content_hash is None:
            raise SpecError(
                "a path dataset needs its content_hash (build it "
                "with realdata.path_spec)"
            )
        defaults = DatasetSpec(rows=1, arrow_type="int8", null_count=0, seed=0)
        generator = [
            name
            for name in _GENERATOR_FIELDS
            if getattr(self, name) != getattr(defaults, name)
        ]
        if generator:
            raise SpecError(
                f"generator field(s) {sorted(generator)!r} do not apply "
                f"to a dataset read from path {self.path!r}"
            )
        if pa.types.is_list(dtype):
            raise SpecError(
                f"list arrow_type {self.arrow_type!r} is not supported "
                f"for a dataset read from path {self.path!r}"
            )
        if self.value_range is not None and self.value_range[0] > self.value_range[1]:
            raise SpecError(f"value_range {self.value_range} must have low <= high")
        if pa.types.is_decimal(dtype) and self.decimal_places != dtype.scale:
            raise SpecError(
                f"decimal_places ({self.decimal_places}) must equal the "
                f"scale of arrow_type {self.arrow_type!r} ({dtype.scale})"
            )
        if self.decimal_places is not None and self.decimal_places < 0:
            raise SpecError(
                f"decimal_places must be non-negative, got {self.decimal_places}"
            )

    @property
    def arrow_dtype(self) -> pa.DataType:
        return parse_arrow_type(self.arrow_type)

    @property
    def support_size(self) -> int | None:
        """k, the number of distinct support values, or None if uncapped."""
        if self.categories is not None:
            return len(self.categories)
        if self.cardinality_ratio is None:
            return None
        return max(1, round(self.cardinality_ratio * self.rows))

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, default=_jsonable)

    @classmethod
    def from_dict(cls, d: dict) -> "DatasetSpec":
        # rebuild the nested dataclasses and re-tuple the sequence fields so that
        # from_dict(json.loads(to_json())) == the original
        d = dict(d)
        if d.get("value_range") is not None:
            d["value_range"] = tuple(d["value_range"])
        if d.get("array_len_range") is not None:
            d["array_len_range"] = tuple(d["array_len_range"])
        if d.get("distribution") is not None:
            d["distribution"] = Distribution(**d["distribution"])
        if d.get("tolerances") is not None:
            d["tolerances"] = Tolerances(**d["tolerances"])
        return cls(**d)

    def id_payload(self) -> str:
        """The JSON `dataset_id` hashes: every field but `tolerances` (they judge
        the result, they do not change it), plus the numpy version, whose
        generators define the draws. For a path dataset the file's location is
        left out and `content_hash` (the bytes and column name) stands in: a
        changed file is a new dataset, a moved one is the same."""
        d = asdict(replace(self, path=None) if self.path is not None else self)
        del d["tolerances"]
        d["numpy"] = np.__version__
        return json.dumps(d, sort_keys=True, default=_jsonable)

    @property
    def dataset_id(self) -> str:
        """Stable 16-hex-character hash of `id_payload()`."""
        return hashlib.sha256(self.id_payload().encode()).hexdigest()[:16]
