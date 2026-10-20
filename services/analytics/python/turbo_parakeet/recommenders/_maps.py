"""Arrow export of a recommender result (spec
docs/superpowers/specs/2026-10-07-top-k-frequencies-design.md §6)."""

from __future__ import annotations

import polars as pl

TOP_K_COLUMNS = ("top_k", "inner_top_k")
_ENTRIES = pl.List(pl.Struct({"key": pl.String, "value": pl.UInt64}))


def from_arrow(result) -> pl.DataFrame:
    """A recommender's Arrow result as a Polars frame with `top_k` / `inner_top_k` as
    lists of key / value structs on every Polars version (Polars 2 reads an Arrow map
    as its own `Map` type, Polars 1 as exactly this list)."""
    df = pl.DataFrame(result)
    maps = [c for c in TOP_K_COLUMNS if c in df.columns and df.schema[c] != _ENTRIES]
    return df.with_columns(pl.col(c).cast(_ENTRIES) for c in maps)


def to_arrow(result: pl.DataFrame):
    """`result` as a pyarrow Table with `top_k` / `inner_top_k` as
    `map<string, uint64>` (Polars, which has no map type, holds them as lists of
    key / value structs). Entry order is kept: most frequent first. Needs pyarrow
    (imported here: the rest of the package does not)."""
    import pyarrow as pa

    top_k_type = pa.map_(pa.string(), pa.uint64())
    table = result.to_arrow()
    for name in TOP_K_COLUMNS:
        if name not in table.column_names:
            continue
        lists = table.column(name).combine_chunks()
        maps = pa.MapArray.from_arrays(
            lists.offsets.cast(pa.int32()),
            lists.values.field("key").cast(pa.string()),
            lists.values.field("value"),
            mask=lists.is_null(),
        )
        i = table.schema.get_field_index(name)
        table = table.set_column(i, pa.field(name, top_k_type), maps)
    return table
