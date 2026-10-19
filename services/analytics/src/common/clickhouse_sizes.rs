//! ClickHouse on-disk size emulation (spec 2026-10-19 §3).

/// How a ClickHouse column is laid out on disk (MergeTree Wide parts).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ChLayout {
    /// Little-endian fixed-width values; `signed` for the GCD codec's magnitude;
    /// `gcd_applies` for integer, decimal, Date, Date32, DateTime and DateTime64 types.
    Fixed {
        width: usize,
        signed: bool,
        gcd_applies: bool,
    },
    /// Per row a varint length and the bytes.
    String,
    /// A dictionary stream (String layout, the distinct values) and a keys stream.
    LowCardinality,
    FixedString(usize),
    /// A UInt8 null-map stream plus the inner type with defaults in null slots.
    Nullable(Box<ChLayout>),
    /// A UInt64 offsets stream plus the inner streams.
    Array(Box<ChLayout>),
    Tuple(Vec<ChLayout>),
}
