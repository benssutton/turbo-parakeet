//! Column profile assembly: `Profile`, `Described`, `describe_one`, `assemble`.

use super::*;
use crate::common::encode_series;
use crate::techniques::describe::conclusions::{conclude, Conclusions};
use polars::prelude::*;
use polars_arrow::array::Array;
use rayon::prelude::*;

// ─────────────────────────────────────────────────────────────────────────────
// describe — column profile assembly and the `describe_columns` entry
// ─────────────────────────────────────────────────────────────────────────────

pub(crate) type Row = Vec<AnyValue<'static>>;

/// Metrics and conclusions of any value series, in output order (spec 2026-10-01 §7).
pub(crate) fn value_fields() -> Vec<(&'static str, DataType)> {
    use DataType::{Boolean, Float64 as F64, String as Str, UInt32 as U32, UInt64 as U64};
    // Arrow has no plain 128-bit integer; parse_i128 caps values at 38 digits.
    let d38 = DataType::Decimal(Some(38), Some(0));
    vec![
        ("n_unique", U64),
        ("unique", Boolean),
        ("est_cardinality", F64),
        ("est_low", F64),
        ("est_high", F64),
        ("est_method", Str),
        ("estimates_agree", Boolean),
        ("class", Str),
        ("min", Str),
        ("max", Str),
        ("min_len", U64),
        ("max_len", U64),
        ("gcd", d38.clone()),
        ("sum_len", U64),
        ("sum_len_unique", U64),
        ("n_nan", U64),
        ("n_inf", U64),
        ("n_fractional", U64),
        ("max_frac_digits", U32),
        ("n_f32_inexact", U64),
        ("n_numeric", U64),
        ("n_numeric_int", U64),
        ("n_leading_zero", U64),
        ("numeric_int_min", d38.clone()),
        ("numeric_int_max", d38),
        ("numeric_max_int_digits", U32),
        ("numeric_max_frac_digits", U32),
        ("numeric_min_frac_digits", U32),
        ("numeric_max_sig_digits", U32),
        ("n_iso_date", U64),
        ("n_iso_time", U64),
        ("n_iso_datetime", U64),
        ("n_iso_datetime_tz", U64),
        ("iso_max_frac_digits", U32),
        ("iso_max_sig_frac_digits", U32),
        ("iso_n_offsets", U64),
        ("iso_n_midnight", U64),
    ]
}

pub(crate) fn fields() -> Vec<(String, DataType)> {
    let mut f = vec![
        ("column".to_string(), DataType::String),
        ("n_rows".into(), DataType::UInt64),
        ("n_null".into(), DataType::UInt64),
    ];
    f.extend(value_fields().into_iter().map(|(n, d)| (n.to_string(), d)));
    f.push(("n_midnight".into(), DataType::UInt64));
    f.push(("inner_n_values".into(), DataType::UInt64));
    f.push(("inner_n_null".into(), DataType::UInt64));
    f.extend(
        value_fields()
            .into_iter()
            .map(|(n, d)| (format!("inner_{n}"), d)),
    );
    f
}

/// Private estimator inputs, appended by `describe_columns` only (Python's reference
/// conclusions read them; spec 2026-10-01 §13.8).
pub(crate) fn input_fields() -> Vec<(String, DataType)> {
    let level = [
        ("argmin", DataType::UInt64),
        ("argmax", DataType::UInt64),
        ("f1", DataType::UInt64),
        ("f2", DataType::UInt64),
        (
            "capture_history",
            DataType::List(Box::new(DataType::UInt64)),
        ),
    ];
    level
        .iter()
        .map(|(n, d)| (n.to_string(), d.clone()))
        .chain(level.iter().map(|(n, d)| (format!("inner_{n}"), d.clone())))
        .collect()
}

pub(crate) fn u64v(v: Option<u64>) -> AnyValue<'static> {
    v.map_or(AnyValue::Null, AnyValue::UInt64)
}
pub(crate) fn u32v(v: Option<u32>) -> AnyValue<'static> {
    v.map_or(AnyValue::Null, AnyValue::UInt32)
}
pub(crate) fn d38v(v: Option<i128>) -> AnyValue<'static> {
    v.map_or(AnyValue::Null, |v| AnyValue::Decimal(v, 0))
}
pub(crate) fn listv(v: &[u64]) -> AnyValue<'static> {
    AnyValue::List(Series::new(PlSmallStr::EMPTY, v))
}
pub(crate) fn nulls(n: usize) -> Row {
    vec![AnyValue::Null; n]
}

/// `proof`: also collect the streaming proof statistics (`StringStats::proof`).
pub(crate) fn strings(s: &Series, proof: bool) -> PolarsResult<Option<StringStats>> {
    let empty = || StringStats {
        proof,
        ..Default::default()
    };
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
                    .with_min_len(CHUNK)
                    .fold(empty, |mut acc, i| {
                        if arr.is_valid(i) {
                            acc.add(arr.value(i).as_bytes());
                        }
                        acc
                    })
                    .reduce(empty, StringStats::merge)
            })
            .fold(empty(), StringStats::merge),
    ))
}

/// Every value metric of one series (the column, or its list's inner values).
pub(crate) struct Profile {
    pub freq: Frequencies,
    pub range: Range,
    pub floats: Option<FloatStats>,
    pub strings: Option<StringStats>,
    /// GCD of the physical values (unordered/gcd.rs); None for non-integer dtypes.
    pub gcd: Option<i128>,
    /// Total byte length of the non-null values (String, Categorical, Enum or Binary columns).
    pub sum_len: Option<u64>,
    /// Float32 series: `n_f32_inexact` does not apply.
    pub is_f32: bool,
    /// Rendered extremes (`recommend::render_value`); None for nested dtypes.
    pub min: Option<String>,
    pub max: Option<String>,
    /// Numeric extremes as f64 (integer, decimal and float dtypes), for `ordinal`.
    pub numeric: Option<(f64, f64)>,
    /// Text levels (String, Categorical, Enum) within `profile`'s `rank_up_to`: every distinct
    /// value by frequency (spec 2026-10-07 §4); None otherwise.
    pub ranking: Option<Ranking>,
}

/// `proof`: see `StringStats::proof` (false on the one-shot path). `rank_up_to`: rank a text
/// level's values when it has at most that many (`Profile.ranking`).
pub(crate) fn profile(
    s: &Series,
    seed: u64,
    proof: bool,
    rank_up_to: Option<u64>,
) -> PolarsResult<Profile> {
    let lengths = byte_lengths(s)?;
    let range = range(s, lengths.as_deref())?;
    // Nested dtypes have no argmin / argmax, hence no min / max.
    let (min, max) = (render_at(s, range.argmin)?, render_at(s, range.argmax)?);
    let numeric = numeric_extremes(s, range.argmin, range.argmax)?;
    let text = matches!(
        s.dtype(),
        DataType::String | DataType::Categorical(..) | DataType::Enum(..)
    );
    let freq = frequencies(
        &encode_series(s)?,
        seed,
        lengths.as_deref(),
        rank_up_to.filter(|_| text),
    );
    let ranking = freq
        .ranked
        .as_deref()
        .map(|r| ranked_values(s, r))
        .transpose()?;
    Ok(Profile {
        freq,
        range,
        floats: float_stats(s)?,
        strings: strings(s, proof)?,
        gcd: crate::techniques::gcd::series_gcd(s)?,
        sum_len: lengths.map(|l| l.iter().sum()),
        is_f32: s.dtype() == &DataType::Float32,
        min,
        max,
        numeric,
        ranking,
    })
}

/// The text of the values at the ranked first rows, with their counts: one `take_slice` of at
/// most `rank_up_to` rows.
fn ranked_values(s: &Series, ranked: &[(u64, u64)]) -> PolarsResult<Ranking> {
    let idx: Vec<IdxSize> = ranked.iter().map(|&(_, first)| first as IdxSize).collect();
    let values = s.take_slice(&idx)?.cast(&DataType::String)?;
    Ok(values
        .str()?
        .into_iter()
        .zip(ranked)
        .map(|(v, &(count, _))| (v.unwrap_or_default().to_owned(), count))
        .collect())
}

/// Row `i` rendered as arrow-rs text (spec 2026-10-01 §13.1).
pub(crate) fn render_at(s: &Series, i: Option<u64>) -> PolarsResult<Option<String>> {
    let Some(i) = i else { return Ok(None) };
    let one = crate::common::ipc_sizes::classic_layout(&s.slice(i as i64, 1))?;
    Ok(crate::common::text::render_value(one.as_ref()))
}

/// The values at rows `lo` / `hi` as f64, for integer, decimal and float dtypes.
pub(crate) fn numeric_extremes(
    s: &Series,
    lo: Option<u64>,
    hi: Option<u64>,
) -> PolarsResult<Option<(f64, f64)>> {
    let dt = s.dtype();
    if !(dt.is_integer() || dt.is_float() || matches!(dt, DataType::Decimal(..))) {
        return Ok(None);
    }
    let at = |i: Option<u64>| -> PolarsResult<Option<f64>> {
        match i {
            None => Ok(None),
            Some(i) => Ok(s.slice(i as i64, 1).cast(&DataType::Float64)?.f64()?.get(0)),
        }
    };
    Ok(at(lo)?.zip(at(hi)?))
}

impl Profile {
    /// The metrics and conclusions in `value_fields()` order.
    pub(crate) fn row(&self, c: &Conclusions) -> Row {
        let f = &self.freq;
        let text = |s: &Option<String>| {
            s.clone()
                .map_or(AnyValue::Null, |s| AnyValue::StringOwned(s.into()))
        };
        let mut row: Row = vec![
            AnyValue::UInt64(f.n_unique),
            AnyValue::Boolean(c.unique),
            AnyValue::Float64(c.est.est_cardinality),
            c.est.est_low.map_or(AnyValue::Null, AnyValue::Float64),
            c.est.est_high.map_or(AnyValue::Null, AnyValue::Float64),
            AnyValue::StringOwned(c.est.method.name().into()),
            c.agree.map_or(AnyValue::Null, AnyValue::Boolean),
            AnyValue::StringOwned(c.class.into()),
            text(&self.min),
            text(&self.max),
            u64v(self.range.min_len),
            u64v(self.range.max_len),
            d38v(self.gcd),
            u64v(self.sum_len),
            u64v(f.sum_len_unique),
        ];
        match self.floats {
            Some(fl) => row.extend([
                AnyValue::UInt64(fl.n_nan),
                AnyValue::UInt64(fl.n_inf),
                AnyValue::UInt64(fl.n_fractional),
                u32v(fl.max_frac_digits),
                if self.is_f32 {
                    AnyValue::Null
                } else {
                    AnyValue::UInt64(fl.n_f32_inexact)
                },
            ]),
            None => row.extend(nulls(5)),
        }
        match &self.strings {
            Some(st) => {
                let (lo, hi) = if st.int_overflow {
                    (None, None)
                } else {
                    (st.int_min, st.int_max)
                };
                row.extend([
                    AnyValue::UInt64(st.n_numeric),
                    AnyValue::UInt64(st.n_numeric_int),
                    AnyValue::UInt64(st.n_leading_zero),
                    d38v(lo),
                    d38v(hi),
                    u32v(st.max_int_digits),
                    u32v(st.max_frac_digits),
                    u32v(st.min_frac_digits),
                    u32v(st.max_sig_digits),
                    AnyValue::UInt64(st.n_iso_date),
                    AnyValue::UInt64(st.n_iso_time),
                    AnyValue::UInt64(st.n_iso_datetime),
                    AnyValue::UInt64(st.n_iso_datetime_tz),
                    u32v(st.iso_max_frac_digits),
                    u32v(st.iso_max_sig_frac_digits),
                    AnyValue::UInt64(st.offsets.len() as u64),
                    AnyValue::UInt64(st.iso_n_midnight),
                ])
            }
            None => row.extend(nulls(17)),
        }
        row
    }

    /// `input_fields()` values for this level.
    pub(crate) fn input_row(&self) -> Row {
        vec![
            u64v(self.range.argmin),
            u64v(self.range.argmax),
            AnyValue::UInt64(self.freq.f1),
            AnyValue::UInt64(self.freq.f2),
            listv(&self.freq.capture_history),
        ]
    }
}

/// A list column's values one level down (`flatten`) and their profile.
pub(crate) struct Inner {
    pub values: Series,
    pub profile: Profile,
}

/// Everything Describe measures on one column (sizes excepted — common/ipc_sizes.rs).
pub(crate) struct Described {
    pub name: PlSmallStr,
    pub dtype: DataType,
    pub n_rows: u64,
    pub n_null: u64,
    pub outer: Profile,
    pub n_midnight: Option<u64>,
    pub inner: Option<Inner>,
}

impl Described {
    /// The column's and its inner values' conclusions.
    pub(crate) fn conclusions(&self, threshold: u64) -> (Conclusions, Option<Conclusions>) {
        let outer = conclude(
            &self.dtype,
            self.n_rows,
            self.n_null,
            &self.outer,
            threshold,
        );
        let inner = self.inner.as_ref().map(|i| {
            conclude(
                i.values.dtype(),
                i.values.len() as u64,
                i.values.null_count() as u64,
                &i.profile,
                threshold,
            )
        });
        (outer, inner)
    }

    /// One `fields()` row.
    pub(crate) fn row(&self, threshold: u64) -> Row {
        let (oc, ic) = self.conclusions(threshold);
        self.row_with(&oc, ic.as_ref())
    }

    /// One `fields()` row from conclusions already computed (`conclusions`).
    pub(crate) fn row_with(&self, oc: &Conclusions, ic: Option<&Conclusions>) -> Row {
        let mut row: Row = vec![
            AnyValue::StringOwned(self.name.clone()),
            AnyValue::UInt64(self.n_rows),
            AnyValue::UInt64(self.n_null),
        ];
        row.extend(self.outer.row(oc));
        row.push(u64v(self.n_midnight));
        match (&self.inner, ic) {
            (Some(i), Some(ic)) => {
                row.push(AnyValue::UInt64(i.values.len() as u64));
                row.push(AnyValue::UInt64(i.values.null_count() as u64));
                row.extend(i.profile.row(ic));
            }
            _ => row.extend(nulls(2 + value_fields().len())),
        }
        row
    }

    /// One `input_fields()` row.
    pub(crate) fn input_row(&self) -> Row {
        let mut row = self.outer.input_row();
        match &self.inner {
            Some(i) => row.extend(i.profile.input_row()),
            None => row.extend(nulls(5)),
        }
        row
    }
}

/// Datetime values at exactly 00:00:00 local time (column time zone, else naive).
pub(crate) fn n_midnight(s: &Series) -> PolarsResult<Option<u64>> {
    let DataType::Datetime(unit, tz) = s.dtype() else {
        return Ok(None);
    };
    let per_day: i64 = match unit {
        TimeUnit::Nanoseconds => 86_400_000_000_000,
        TimeUnit::Microseconds => 86_400_000_000,
        TimeUnit::Milliseconds => 86_400_000,
    };
    let phys = s.to_physical_repr();
    let values = phys.i64()?;
    let count = match tz {
        None => values
            .into_iter()
            .flatten()
            .filter(|v| v.rem_euclid(per_day) == 0)
            .count(),
        Some(tz) => {
            let zone: chrono_tz::Tz = tz
                .as_str()
                .parse()
                .map_err(|_| polars_err!(ComputeError: "describe: unknown time zone {tz}"))?;
            let per_sec = per_day / 86_400;
            values
                .into_iter()
                .flatten()
                .filter(|&v| {
                    v.rem_euclid(per_sec) == 0
                        && chrono::DateTime::from_timestamp(v.div_euclid(per_sec), 0).is_some_and(
                            |t| t.with_timezone(&zone).time() == chrono::NaiveTime::MIN,
                        )
                })
                .count()
        }
    };
    Ok(Some(count as u64))
}

/// Values one level down, skipping null lists — the same definition as the
/// Python `flatten` (drop_nulls, then explode the non-empty lists). Element i is
/// what `inner_argmin` / `inner_argmax` index into.
pub(crate) fn flatten(s: &Series) -> PolarsResult<Option<Series>> {
    let (inner, ranges): (Series, Vec<Option<(usize, usize)>>) = match s.dtype() {
        DataType::List(_) => {
            let ca = s.list()?.rechunk();
            (ca.get_inner(), list_ranges(&ca))
        }
        DataType::Array(_, width) => {
            let ca = s.array()?.rechunk();
            let valid = ca.is_not_null();
            let ranges = valid
                .iter()
                .enumerate()
                .map(|(i, ok)| (ok == Some(true)).then_some((i * width, *width)))
                .collect();
            (ca.get_inner(), ranges)
        }
        _ => return Ok(None),
    };
    let idx: Vec<IdxSize> = ranges
        .into_iter()
        .flatten()
        .flat_map(|(start, len)| (start..start + len).map(|i| i as IdxSize))
        .collect();
    Ok(Some(inner.take_slice(&idx)?))
}

/// `proof`: see `StringStats::proof` (false on the one-shot path); `rank_up_to`: see `profile`.
pub(crate) fn describe_one(
    s: &Series,
    seed: u64,
    proof: bool,
    rank_up_to: Option<u64>,
) -> PolarsResult<Described> {
    let inner = match flatten(s)? {
        Some(values) => {
            let profile = profile(&values, seed, proof, rank_up_to)?;
            Some(Inner { values, profile })
        }
        None => None,
    };
    Ok(Described {
        name: s.name().clone(),
        dtype: s.dtype().clone(),
        n_rows: s.len() as u64,
        n_null: s.null_count() as u64,
        outer: profile(s, seed, proof, rank_up_to)?,
        n_midnight: n_midnight(s)?,
        inner,
    })
}

/// A Struct series `name` with one field per `fields` entry and one row per `rows` entry.
pub(crate) fn assemble(
    name: &str,
    fields: &[(String, DataType)],
    rows: &[Row],
) -> PolarsResult<Series> {
    let columns = fields
        .iter()
        .enumerate()
        .map(|(j, (field, dtype))| {
            let values: Vec<AnyValue> = rows.iter().map(|r| r[j].clone()).collect();
            Series::from_any_values_and_dtype(field.as_str().into(), &values, dtype, true)
        })
        .collect::<PolarsResult<Vec<_>>>()?;
    Ok(StructChunked::from_series(name.into(), rows.len(), columns.iter())?.into_series())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_gcd_and_lengths() {
        let p = profile(
            &Series::new("x".into(), &[Some(10i64), None, Some(30)]),
            0,
            false,
            None,
        )
        .unwrap();
        assert_eq!((p.gcd, p.sum_len), (Some(10), None));
        let s = profile(&Series::new("x".into(), &["ab", "ab", "c"]), 0, false, None).unwrap();
        assert_eq!(
            (s.gcd, s.sum_len, s.freq.sum_len_unique),
            (None, Some(5), Some(3))
        );
    }

    #[test]
    fn described_row_matches_fields() {
        let list = Series::new("x".into(), [Some(Series::new("".into(), &[1i64, 2])), None]);
        let d = describe_one(&list, 0, false, None).unwrap();
        assert_eq!(d.row(10_000).len(), fields().len());
        assert_eq!(d.input_row().len(), input_fields().len());
        assert_eq!(d.inner.as_ref().unwrap().values.len(), 2);
        // Nested extremes are dropped; the inner values keep theirs.
        assert_eq!(
            (d.outer.min.as_deref(), d.outer.max.as_deref()),
            (None, None)
        );
        let inner = &d.inner.as_ref().unwrap().profile;
        assert_eq!(
            (inner.min.as_deref(), inner.max.as_deref()),
            (Some("1"), Some("2"))
        );
        let floats = describe_one(&Series::new("y".into(), &[1.5f64]), 0, false, None).unwrap();
        assert_eq!(floats.row(10_000).len(), fields().len());
        assert!(floats.inner.is_none() && floats.outer.floats.is_some());
    }

    #[test]
    fn rendered_extremes_are_arrow_rs_text() {
        let ts = Series::new("t".into(), &[1_704_164_645_000_000i64, 0])
            .cast(&DataType::Datetime(TimeUnit::Microseconds, None))
            .unwrap();
        let p = profile(&ts, 0, false, None).unwrap();
        assert_eq!(
            (p.min.as_deref(), p.max.as_deref()),
            (Some("1970-01-01T00:00:00"), Some("2024-01-02T03:04:05"))
        );
        let f = profile(
            &Series::new("f".into(), &[1.0f64, f64::NAN, -2.5]),
            0,
            false,
            None,
        )
        .unwrap();
        assert_eq!(
            (f.min.as_deref(), f.max.as_deref()),
            (Some("-2.5"), Some("1.0"))
        );
        assert_eq!(f.numeric, Some((-2.5, 1.0)));
        let s = profile(&Series::new("s".into(), &["b", "a"]), 0, false, None).unwrap();
        assert_eq!(
            (s.min.as_deref(), s.max.as_deref(), s.numeric),
            (Some("a"), Some("b"), None)
        );
    }

    #[test]
    fn text_levels_rank_their_values() {
        let s = Series::new(
            "x".into(),
            &[
                Some("b"),
                Some("a"),
                Some("b"),
                None,
                Some("c"),
                Some("a"),
                Some("b"),
            ],
        );
        let want = Some(vec![("b".to_string(), 3), ("a".into(), 2), ("c".into(), 1)]);
        assert_eq!(profile(&s, 0, false, Some(10)).unwrap().ranking, want);
        let cats = Categories::new("rank-test".into(), "".into(), CategoricalPhysical::U32);
        let cat = s
            .cast(&DataType::Categorical(cats.clone(), cats.mapping()))
            .unwrap();
        assert_eq!(profile(&cat, 0, false, Some(10)).unwrap().ranking, want);
        let e = DataType::from_frozen_categories(FrozenCategories::new(["c", "a", "b"]).unwrap());
        assert_eq!(
            profile(&s.cast(&e).unwrap(), 0, false, Some(10))
                .unwrap()
                .ranking,
            want
        );
        // Not text, no limit, or over the limit: no ranking.
        let ints = Series::new("i".into(), &[1i64, 1, 2]);
        assert_eq!(profile(&ints, 0, false, Some(10)).unwrap().ranking, None);
        assert_eq!(profile(&s, 0, false, None).unwrap().ranking, None);
        assert_eq!(profile(&s, 0, false, Some(2)).unwrap().ranking, None);
        // A list's inner values are ranked too.
        let l = Series::new(
            "l".into(),
            [
                Some(Series::new("".into(), &["q", "p"])),
                None,
                Some(Series::new("".into(), &["p"])),
            ],
        );
        let d = describe_one(&l, 0, false, Some(10)).unwrap();
        assert_eq!(d.outer.ranking, None);
        assert_eq!(
            d.inner.unwrap().profile.ranking,
            Some(vec![("p".to_string(), 2), ("q".into(), 1)])
        );
    }
}
