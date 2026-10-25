from decimal import Decimal

import pyarrow as pa
import pytest

from synthdata.frame.dtypes import to_polars, to_type

INT = pa.array([0, 3, None, 7], pa.int64())
STR = pa.array(["b", None, "a", "b"])


@pytest.mark.parametrize(
    "type_, base, arrow",
    [
        ("int64", INT, pa.int64()),
        ("uint32", INT, pa.uint32()),
        ("bool", pa.array([0, 1, None]), pa.bool_()),
        ("date32", INT, pa.date32()),
        ("timestamp", INT, pa.timestamp("us")),
        ("duration", INT, pa.duration("us")),
        ("time64", INT, pa.time64("us")),
        ("decimal", INT, pa.decimal128(12, 2)),
        ("binary", STR, pa.binary()),
        ("categorical", STR, pa.dictionary(pa.int32(), pa.string())),
        ("string", STR, pa.string()),
    ],
)
def test_types(type_, base, arrow):
    out = to_type(base, type_)
    assert out.type == arrow and out.null_count == base.null_count


def test_decimal_keeps_the_unscaled_value():
    assert to_type(INT, "decimal").to_pylist() == [
        Decimal("0.00"),
        Decimal("0.03"),
        None,
        Decimal("0.07"),
    ]


def test_enum_in_polars():
    df = to_polars(pa.table({"e": to_type(STR, "enum")}), {"e": "enum"})
    assert str(df.schema["e"]).startswith("Enum") and list(
        df.schema["e"].categories
    ) == ["a", "b"]


def test_decimal_of_a_slice():
    sliced = pa.array([9, None, 1, None, 5], pa.int64()).slice(2)
    assert to_type(sliced, "decimal").to_pylist() == [
        Decimal("0.01"),
        None,
        Decimal("0.05"),
    ]
