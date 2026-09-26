// ─────────────────────────────────────────────────────────────────────────────
// describe — per-column profile (plugin entry points `describe_columns`, `column_sizes`)
// ─────────────────────────────────────────────────────────────────────────────
//
// Spec: docs/superpowers/specs/2026-09-26-describe-technique-design.md. One pass
// per column, columns in parallel (rayon); inside a column the frequency, float
// and string work runs over 64K-row chunks in parallel. Every value metric is
// computed on the column and again on its inner values (List/Array flattened one
// level, null lists skipped). Field names and order match
// analytics/describe/base.py (DescribeRust maps them by name).

pub(crate) mod frequency;
pub(crate) mod numeric;
pub(crate) mod patterns;
pub(crate) mod range;
pub(crate) mod sizes;

use crate::shared::encode_series;
use patterns::StringStats;
use polars::prelude::*;
use polars_arrow::array::Array;
use pyo3_polars::derive::polars_expr;
use rayon::prelude::*;
use serde::Deserialize;

type Row = Vec<AnyValue<'static>>;

/// Metrics computed on any value series, in output order (base.py VALUE_METRICS).
fn value_fields() -> Vec<(&'static str, DataType)> {
    use DataType::{Float64 as F64, UInt32 as U32, UInt64 as U64};
    let list = DataType::List(Box::new(U64));
    // Arrow has no plain 128-bit integer; parse_i128 caps values at 38 digits.
    let d38 = DataType::Decimal(Some(38), Some(0));
    vec![
        ("n_unique", U64), ("entropy", F64), ("f1", U64), ("f2", U64), ("argmin", U64), ("argmax", U64),
        ("min_len", U64), ("max_len", U64), ("top5_idx", list.clone()), ("top5_count", list.clone()), ("capture_history", list),
        ("n_nan", U64), ("n_inf", U64), ("n_fractional", U64), ("max_frac_digits", U32), ("n_f32_inexact", U64),
        ("n_numeric", U64), ("n_numeric_int", U64), ("n_leading_zero", U64), ("numeric_int_min", d38.clone()), ("numeric_int_max", d38),
        ("numeric_max_int_digits", U32), ("numeric_max_frac_digits", U32),
        ("n_iso_date", U64), ("n_iso_time", U64), ("n_iso_datetime", U64), ("n_iso_datetime_tz", U64),
        ("iso_max_frac_digits", U32), ("iso_n_offsets", U64), ("iso_n_midnight", U64),
    ]
}

fn fields() -> Vec<(String, DataType)> {
    let mut f = vec![("column".to_string(), DataType::String), ("n_rows".into(), DataType::UInt64), ("n_null".into(), DataType::UInt64)];
    f.extend(value_fields().into_iter().map(|(n, d)| (n.to_string(), d)));
    f.push(("n_midnight".into(), DataType::UInt64));
    f.push(("inner_n_values".into(), DataType::UInt64));
    f.push(("inner_n_null".into(), DataType::UInt64));
    f.extend(value_fields().into_iter().map(|(n, d)| (format!("inner_{n}"), d)));
    f
}

fn describe_output_type(_input_fields: &[Field]) -> PolarsResult<Field> {
    let fields = fields().into_iter().map(|(n, d)| Field::new(n.into(), d)).collect();
    Ok(Field::new("describe".into(), DataType::Struct(fields)))
}

fn u64v(v: Option<u64>) -> AnyValue<'static> { v.map_or(AnyValue::Null, AnyValue::UInt64) }
fn u32v(v: Option<u32>) -> AnyValue<'static> { v.map_or(AnyValue::Null, AnyValue::UInt32) }
fn d38v(v: Option<i128>) -> AnyValue<'static> { v.map_or(AnyValue::Null, |v| AnyValue::Decimal(v, 0)) }
fn listv(v: &[u64]) -> AnyValue<'static> { AnyValue::List(Series::new(PlSmallStr::EMPTY, v)) }
fn nulls(n: usize) -> Row { vec![AnyValue::Null; n] }

fn strings(s: &Series) -> PolarsResult<Option<StringStats>> {
    let st = match s.dtype() {
        DataType::String => s.clone(),
        DataType::Categorical(_, _) | DataType::Enum(_, _) => s.cast(&DataType::String)?,
        _ => return Ok(None),
    };
    Ok(Some(
        st.str()?
            .downcast_iter()
            .map(|arr| {
                (0..arr.len())
                    .into_par_iter()
                    .with_min_len(frequency::CHUNK)
                    .fold(StringStats::default, |mut acc, i| {
                        if arr.is_valid(i) {
                            acc.add(arr.value(i).as_bytes());
                        }
                        acc
                    })
                    .reduce(StringStats::default, StringStats::merge)
            })
            .fold(StringStats::default(), StringStats::merge),
    ))
}

/// Every value metric for one series, in `value_fields()` order.
fn profile(s: &Series, seed: u64) -> PolarsResult<Row> {
    let f = frequency::frequencies(&encode_series(s)?, seed);
    let r = range::range(s)?;
    let mut row: Row = vec![
        AnyValue::UInt64(f.n_unique), AnyValue::Float64(f.entropy), AnyValue::UInt64(f.f1), AnyValue::UInt64(f.f2),
        u64v(r.argmin), u64v(r.argmax), u64v(r.min_len), u64v(r.max_len),
        listv(&f.top5_idx), listv(&f.top5_count), listv(&f.capture_history),
    ];
    match numeric::float_stats(s)? {
        Some(fl) => row.extend([
            AnyValue::UInt64(fl.n_nan), AnyValue::UInt64(fl.n_inf), AnyValue::UInt64(fl.n_fractional), u32v(fl.max_frac_digits),
            if s.dtype() == &DataType::Float32 { AnyValue::Null } else { AnyValue::UInt64(fl.n_f32_inexact) },
        ]),
        None => row.extend(nulls(5)),
    }
    match strings(s)? {
        Some(st) => {
            let (lo, hi) = if st.int_overflow { (None, None) } else { (st.int_min, st.int_max) };
            row.extend([
                AnyValue::UInt64(st.n_numeric), AnyValue::UInt64(st.n_numeric_int), AnyValue::UInt64(st.n_leading_zero),
                d38v(lo), d38v(hi), u32v(st.max_int_digits), u32v(st.max_frac_digits),
                AnyValue::UInt64(st.n_iso_date), AnyValue::UInt64(st.n_iso_time), AnyValue::UInt64(st.n_iso_datetime),
                AnyValue::UInt64(st.n_iso_datetime_tz), u32v(st.iso_max_frac_digits),
                AnyValue::UInt64(st.offsets.len() as u64), AnyValue::UInt64(st.iso_n_midnight),
            ])
        }
        None => row.extend(nulls(14)),
    }
    Ok(row)
}

/// Datetime values at exactly 00:00:00 local time (column time zone, else naive).
fn n_midnight(s: &Series) -> PolarsResult<Option<u64>> {
    let DataType::Datetime(unit, tz) = s.dtype() else { return Ok(None) };
    let per_day: i64 = match unit {
        TimeUnit::Nanoseconds => 86_400_000_000_000,
        TimeUnit::Microseconds => 86_400_000_000,
        TimeUnit::Milliseconds => 86_400_000,
    };
    let phys = s.to_physical_repr();
    let values = phys.i64()?;
    let count = match tz {
        None => values.into_iter().flatten().filter(|v| v.rem_euclid(per_day) == 0).count(),
        Some(tz) => {
            let zone: chrono_tz::Tz = tz.as_str().parse().map_err(|_| polars_err!(ComputeError: "describe: unknown time zone {tz}"))?;
            let per_sec = per_day / 86_400;
            values
                .into_iter()
                .flatten()
                .filter(|&v| {
                    v.rem_euclid(per_sec) == 0
                        && chrono::DateTime::from_timestamp(v.div_euclid(per_sec), 0)
                            .is_some_and(|t| t.with_timezone(&zone).time() == chrono::NaiveTime::MIN)
                })
                .count()
        }
    };
    Ok(Some(count as u64))
}

/// Values one level down, skipping null lists — the same definition as the
/// Python `flatten` (drop_nulls, then explode the non-empty lists). Element i is
/// what `inner_argmin` / `inner_top5_idx` index into.
fn flatten(s: &Series) -> PolarsResult<Option<Series>> {
    let (inner, ranges): (Series, Vec<Option<(usize, usize)>>) = match s.dtype() {
        DataType::List(_) => {
            let ca = s.list()?.rechunk();
            (ca.get_inner(), range::list_ranges(&ca))
        }
        DataType::Array(_, width) => {
            let ca = s.array()?.rechunk();
            let valid = ca.is_not_null();
            let ranges = valid.iter().enumerate().map(|(i, ok)| (ok == Some(true)).then_some((i * width, *width))).collect();
            (ca.get_inner(), ranges)
        }
        _ => return Ok(None),
    };
    let idx: Vec<IdxSize> = ranges.into_iter().flatten().flat_map(|(start, len)| (start..start + len).map(|i| i as IdxSize)).collect();
    Ok(Some(inner.take_slice(&idx)?))
}

fn describe_one(s: &Series, seed: u64) -> PolarsResult<Row> {
    let mut row: Row = vec![
        AnyValue::StringOwned(s.name().clone()),
        AnyValue::UInt64(s.len() as u64),
        AnyValue::UInt64(s.null_count() as u64),
    ];
    row.extend(profile(s, seed)?);
    row.push(u64v(n_midnight(s)?));
    match flatten(s)? {
        Some(inner) => {
            row.push(AnyValue::UInt64(inner.len() as u64));
            row.push(AnyValue::UInt64(inner.null_count() as u64));
            row.extend(profile(&inner, seed)?);
        }
        None => row.extend(nulls(2 + value_fields().len())),
    }
    Ok(row)
}

pub(crate) fn describe_columns_impl(inputs: &[Series], seed: u64) -> PolarsResult<Series> {
    let rows: Vec<Row> = inputs.par_iter().map(|s| describe_one(s, seed)).collect::<PolarsResult<_>>()?;
    let columns = fields()
        .iter()
        .enumerate()
        .map(|(j, (name, dtype))| {
            let values: Vec<AnyValue> = rows.iter().map(|r| r[j].clone()).collect();
            Series::from_any_values_and_dtype(name.as_str().into(), &values, dtype, true)
        })
        .collect::<PolarsResult<Vec<_>>>()?;
    Ok(StructChunked::from_series("describe".into(), inputs.len(), columns.iter())?.into_series())
}

#[derive(Deserialize)]
struct DescribeKwargs {
    seed: u64,
}

#[polars_expr(output_type_func=describe_output_type)]
fn describe_columns(inputs: &[Series], kwargs: DescribeKwargs) -> PolarsResult<Series> {
    describe_columns_impl(inputs, kwargs.seed)
}
