//! The recommendation for one column: `recommend`, `Rec`, `Prepared`.

use super::*;
use crate::common::ipc_sizes::{classic_layout, ipc_body_bytes, layout_sizes, sizes_of, Sizes};
use crate::techniques::describe::conclusions::Conclusions;
use crate::techniques::describe::{describe_one, value_fields, Described, Row};
use arrow_array::{Array, ArrayRef};
use arrow_schema::DataType as AT;
use polars::prelude::{
    polars_err, AnyValue, DataType as PT, Float64Chunked, IntoSeries, NewChunkedArray,
    PolarsResult, Series, StringChunked, StructChunked, UInt64Chunked,
};

// ── one column ───────────────────────────────────────────────────────────────

pub(crate) struct Rec {
    /// The recommended array has nulls (a list → scalar recast turns `[null]` into a null row).
    pub nullable: bool,
    pub arrow_type: String,
    pub arrow_size: u64,
    /// None when nothing was sampled (streaming, reservoir_rows = 0).
    pub arrow_zstd: Option<u64>,
    /// The Polars type (pl_name); always set by both recommenders.
    pub polars_type: Option<String>,
    pub polars_size: u64,
    /// None when nothing was sampled (streaming, reservoir_rows = 0).
    pub polars_zstd: Option<u64>,
    pub lossy: bool,
    pub candidates: Vec<Candidate>,
}

/// An Enum's categories, found through lists and arrays.
pub(crate) fn enum_categories(dtype: &PT) -> Option<Vec<String>> {
    match dtype {
        PT::Enum(fc, _) => Some(fc.categories().values_iter().map(str::to_owned).collect()),
        PT::List(it) | PT::Array(it, _) => enum_categories(it),
        _ => None,
    }
}

/// The recommendation for one column.
/// `values` is `s` in the classic layout (common/ipc_sizes.rs's `classic_layout`); `(oc, ic)` is
/// `d.conclusions(params.categorical_threshold)`.
pub(crate) fn recommend(
    s: &Series,
    values: &ArrayRef,
    d: &Described,
    (oc, ic): (Conclusions, Option<Conclusions>),
    sz: &Sizes,
    params: &Params,
) -> PolarsResult<Rec> {
    let [size_bytes, _, polars_bytes, polars_zstd] = *sz;
    let name = s.name().as_str();
    let err = |e: String| polars_err!(ComputeError: "recommend {}: {}", name, e);
    let outer = Level::of_values(
        s.dtype(),
        values.clone(),
        &d.outer,
        d.n_midnight,
        size_bytes,
        oc.est,
        "",
    )
    .map_err(err)?;
    let chosen = match &d.inner {
        Some(inner) if matches!(s.dtype(), PT::List(_) | PT::Array(..)) => {
            let (rows, child, width) = list_parts(values).map_err(err)?;
            if child.len() == inner.values.len() {
                let child_size = ipc_body_bytes(child.as_ref(), None)?;
                let inner_lvl = Level::of_values(
                    inner.values.dtype(),
                    child,
                    &inner.profile,
                    None,
                    child_size,
                    ic.map_or(oc.est, |c| c.est),
                    "inner: ",
                )
                .map_err(err)?;
                choose_list(&outer, &inner_lvl, &rows, width, params).map_err(err)?
            } else {
                // Defensive: list_parts builds Describe's flatten, so the lengths agree.
                let why = format!(
                    "inner values {} ≠ Describe's inner_n_values {}: inner type kept",
                    child.len(),
                    inner.values.len()
                );
                choose_original(&outer, &why)
            }
        }
        _ => choose(&outer, params).map_err(err)?,
    };
    let t = chosen.array.data_type().clone();
    let (polars_type, sizes) = if matches!(chosen.target, Target::Original(_)) {
        // The original's Polars type, spelled as the streaming recommender spells it.
        let enum_values = enum_categories(s.dtype());
        let a = chosen.array.as_ref();
        (
            Some(pl_name(&t, name, enum_values.as_deref(), &AT::UInt32)),
            [
                ipc_body_bytes(a, None)?,
                ipc_body_bytes(a, Some(params.zstd_level))?,
                polars_bytes,
                polars_zstd,
            ],
        )
    } else {
        let key = chosen.target.polars_key().unwrap_or(AT::UInt32);
        let layout = to_polars_layout(&chosen.array, &key).map_err(err)?;
        // Spec B §5.4 step 6: the Polars layout only widens, so it must cast back exactly.
        first_mismatch(
            &chosen.array,
            &arrow_cast(layout.as_ref(), &t).map_err(err)?,
        )
        .map_err(|e| err(format!("Polars layout: {e}")))?;
        (
            Some(pl_name(&t, name, None, &key)),
            layout_sizes(chosen.array.as_ref(), layout.as_ref(), params.zstd_level)?,
        )
    };
    let [arrow_size, arrow_zstd, polars_size, polars_zstd] = sizes;
    Ok(Rec {
        nullable: chosen.array.logical_null_count() > 0,
        arrow_type: pa_name(&t),
        arrow_size,
        arrow_zstd: Some(arrow_zstd),
        polars_type,
        polars_size,
        polars_zstd: Some(polars_zstd),
        lossy: chosen.lossy,
        candidates: chosen.candidates,
    })
}

// ── entry ────────────────────────────────────────────────────────────────────

pub(crate) fn candidates_series(c: &[Candidate]) -> Series {
    let text = |name: &str, v: Vec<Option<String>>| {
        StringChunked::from_iter_options(name.into(), v.into_iter()).into_series()
    };
    let cols = [
        text(
            "arrow_type",
            c.iter()
                .map(|x| Some(pa_name(&x.target.arrow_type())))
                .collect(),
        ),
        text("rule", c.iter().map(|x| Some(x.rule.clone())).collect()),
        text(
            "evidence",
            c.iter().map(|x| Some(x.evidence.clone())).collect(),
        ),
        UInt64Chunked::from_iter_values("predicted_bytes".into(), c.iter().map(|x| x.predicted))
            .into_series(),
        Float64Chunked::from_iter_values(
            "projected_population_bytes".into(),
            c.iter().map(|x| x.projected),
        )
        .into_series(),
        text(
            "outcome",
            c.iter()
                .map(|x| Some(x.outcome.name().to_string()))
                .collect(),
        ),
        text("reason", c.iter().map(|x| x.reason.clone()).collect()),
    ];
    StructChunked::from_series("candidate".into(), c.len(), cols.iter())
        .expect("equal-length fields")
        .into_series()
}

pub(crate) fn rec_row(r: &Rec) -> Row {
    let text = |s: &str| AnyValue::StringOwned(s.into());
    vec![
        AnyValue::Boolean(r.nullable),
        text(&r.arrow_type),
        AnyValue::UInt64(r.arrow_size),
        r.arrow_zstd.map_or(AnyValue::Null, AnyValue::UInt64),
        r.polars_type.as_deref().map_or(AnyValue::Null, text),
        AnyValue::UInt64(r.polars_size),
        r.polars_zstd.map_or(AnyValue::Null, AnyValue::UInt64),
        AnyValue::Boolean(r.lossy),
        AnyValue::List(candidates_series(&r.candidates)),
    ]
}

/// One column's statistics, collected by the one-shot recommender's `add` (spec
/// 2026-10-04 §3.1); `row` recommends from them on the column's actual values.
pub(crate) struct Prepared {
    series: Series,
    classic: ArrayRef,
    described: Described,
    sizes: Sizes,
    /// pyarrow spelling of the input type.
    dtype: String,
}

/// `s`'s Describe statistics, classic layout and sizes; `input_type` is its Arrow type.
pub(crate) fn prepare(s: Series, input_type: &AT, params: &Params) -> PolarsResult<Prepared> {
    let described = describe_one(&s, params.seed, false, Some(params.categorical_threshold))?;
    let classic = classic_layout(&s)?;
    let sizes = sizes_of(&s, &classic, params.zstd_level)?;
    Ok(Prepared {
        series: s,
        classic,
        described,
        sizes,
        dtype: pa_name(input_type),
    })
}

impl Prepared {
    /// The column's `recommender_fields(false)` row.
    pub(crate) fn row(&self, params: &Params) -> PolarsResult<Row> {
        let d = &self.described;
        let (oc, ic) = d.conclusions(params.categorical_threshold);
        let rec = recommend(
            &self.series,
            &self.classic,
            d,
            (oc, ic),
            &self.sizes,
            params,
        )?;
        // Describe's row: column, n_rows, n_null, value block, n_midnight, then the
        // inner block (inner_n_values, inner_n_null, inner value block).
        let mut base = d.row_with(&oc, ic.as_ref()).into_iter();
        let mut row: Row = Vec::with_capacity(base.len() + self.sizes.len() + 13);
        row.extend(base.next()); // column
        row.push(AnyValue::StringOwned("computed".into()));
        row.push(AnyValue::StringOwned(self.dtype.as_str().into()));
        row.extend(base.by_ref().take(3 + value_fields().len())); // through n_midnight
        row.extend(self.sizes.iter().map(|&v| AnyValue::UInt64(v)));
        row.extend(base); // the inner block
        let inner_ranking = d.inner.as_ref().and_then(|i| i.profile.ranking.as_ref());
        row.push(top_k_cell(
            d.outer.ranking.as_ref(),
            &rec.candidates,
            "",
            params.top_k,
        ));
        row.push(top_k_cell(
            inner_ranking,
            &rec.candidates,
            "inner: ",
            params.top_k,
        ));
        row.extend(rec_row(&rec));
        Ok(row)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::recommenders::engine::choose::tests::{describe_one, params};
    use polars::prelude::{DataType as PT, NamedFrom, Series};

    use crate::common::ipc_sizes::ipc_body_bytes;

    use crate::common::ipc_sizes::sizes;

    fn rec(s: Series) -> Rec {
        let d = describe_one(&s, 0).unwrap();
        let sz = sizes(&s, 1).unwrap();
        recommend(
            &s,
            &classic_layout(&s).unwrap(),
            &d,
            d.conclusions(10_000),
            &sz,
            &params(),
        )
        .unwrap()
    }

    fn chosen(r: &Rec) -> &Candidate {
        r.candidates
            .iter()
            .find(|c| c.outcome == Outcome::Chosen)
            .unwrap()
    }

    #[test]
    fn end_to_end_choices() {
        assert_eq!(
            rec(Series::new("x".into(), &[0i64, 5, 127])).arrow_type,
            "uint8"
        );
        let price = rec(Series::new("x".into(), &[123.45f64, 99.99]));
        assert_eq!(
            (price.arrow_type.as_str(), price.polars_type.as_deref()),
            ("decimal32(5, 2)", Some("Decimal(precision=5, scale=2)"))
        );
        assert_eq!(
            rec(Series::new(
                "x".into(),
                &["2024-01-05 10:00:00.120", "2024-01-06T11:00:00"]
            ))
            .arrow_type,
            "timestamp[ms]"
        );
        let kept = rec(Series::new("x".into(), &[0.1f64, f64::NAN]));
        assert_eq!(
            (kept.arrow_type.as_str(), kept.polars_type.as_deref()),
            ("double", Some("Float64"))
        );
    }

    #[test]
    fn failed_cast_falls_back() {
        let r = rec(Series::new(
            "x".into(),
            &["2300-01-01T00:00:00.123456789", "2024-01-05T10:00:00"],
        ));
        assert_eq!(r.arrow_type, "string");
        assert!(r
            .candidates
            .iter()
            .any(|c| c.outcome == Outcome::Failed
                && pa_name(&c.target.arrow_type()) == "timestamp[ns]"));
    }

    #[test]
    fn single_item_lists_become_scalars() {
        let s = Series::new(
            "x".into(),
            [
                Some(Series::new("".into(), &[1i64])),
                Some(Series::new("".into(), &[2i64])),
                None,
            ],
        );
        assert_eq!(rec(s).arrow_type, "uint8");
        // Null lists and null items both occur: the column stays a list (Spec B §4.4).
        let both = Series::new(
            "x".into(),
            [
                Some(Series::new("".into(), &[Some(2i64)])),
                Some(Series::new("".into(), &[None::<i64>])),
                None,
            ],
        );
        assert_eq!(rec(both).arrow_type, "list<item: uint8>");
        // Inner values {1}: 0 ≤ min, max ≤ 1 → Boolean ties UInt8 in size and wins on rank (§4.1, §4.2).
        let ones = Series::new(
            "x".into(),
            [
                Some(Series::new("".into(), &[Some(1i64)])),
                Some(Series::new("".into(), &[None::<i64>])),
                None,
            ],
        );
        assert_eq!(rec(ones).arrow_type, "list<item: bool>");
    }

    #[test]
    fn nullable_means_the_recommended_array_has_nulls() {
        // [null] items become null scalars: the recast column has a null although the source has none.
        let s = Series::new(
            "x".into(),
            [
                Some(Series::new("".into(), &[Some(1i64)])),
                Some(Series::new("".into(), &[None::<i64>])),
                Some(Series::new("".into(), &[Some(2i64)])),
            ],
        );
        let r = rec(s);
        assert_eq!((r.arrow_type.as_str(), r.nullable), ("uint8", true));
        assert!(!rec(Series::new("x".into(), &[1i64, 2])).nullable);
        assert!(rec(Series::new("x".into(), &[Some(1i64), None])).nullable);
    }

    /// A List(Int64) Series with offsets `offsets` over `values`, row `null` null — its
    /// values still behind the offsets (as `pl.when(mask).then(list).otherwise(None)` leaves them).
    fn null_list_holding_values(values: Vec<i64>, offsets: Vec<i64>, null: usize) -> Series {
        use polars_arrow::array::{ListArray as PlList, PrimitiveArray};
        use polars_arrow::bitmap::Bitmap;
        use polars_arrow::offset::Offsets;
        let values = PrimitiveArray::<i64>::from_vec(values).boxed();
        let rows = offsets.len() - 1;
        let arr = PlList::<i64>::new(
            PlList::<i64>::default_datatype(values.dtype().clone()),
            Offsets::try_from(offsets).unwrap().into(),
            values,
            Some(Bitmap::from_iter((0..rows).map(|i| i != null))),
        );
        Series::from_arrow("x".into(), arr.boxed()).unwrap()
    }

    #[test]
    fn null_lists_holding_values_are_narrowed() {
        let s = null_list_holding_values(vec![1, 2, 3, 4, 5], vec![0, 2, 4, 5], 1);
        assert_eq!(describe_one(&s, 0).unwrap().inner.unwrap().values.len(), 3); // Describe's flatten skips the null row
        let r = rec(s);
        assert_eq!(
            (r.arrow_type.as_str(), r.nullable),
            ("list<item: uint8>", true)
        );
        assert_eq!(chosen(&r).predicted, r.arrow_size);
        let inner = r
            .candidates
            .iter()
            .find(|c| c.outcome == Outcome::Chosen && c.rule.starts_with("inner: "))
            .unwrap();
        assert_eq!(pa_name(&inner.target.arrow_type()), "uint8");
        // Single-item rows: the null row (holding [9]) becomes a null scalar.
        let r = rec(null_list_holding_values(vec![1, 9, 3], vec![0, 1, 2, 3], 1));
        assert_eq!((r.arrow_type.as_str(), r.nullable), ("uint8", true));
        assert_eq!(chosen(&r).predicted, r.arrow_size);
    }

    #[test]
    fn predicted_equals_measured() {
        let cases = [
            Series::new("x".into(), &[Some(0i64), Some(5), None]),
            Series::new("x".into(), &[123.45f64, 99.99]),
            Series::new("x".into(), &["a", "b", "a", "b", "a", "b"]),
            Series::new(
                "x".into(),
                &["2024-01-05T10:00+05:00", "2024-01-05T10:00-03:30"],
            ),
            Series::new("x".into(), [Some(Series::new("".into(), &[1i64, 2])), None]),
            // list→scalar timestamp_with_offset: a null list is a null struct row, children null-free
            Series::new(
                "x".into(),
                [
                    Some(Series::new("".into(), &["2024-01-05T10:00+05:00"])),
                    None,
                    Some(Series::new("".into(), &["2024-01-05T10:00-03:30"])),
                ],
            ),
        ];
        for s in cases {
            let r = rec(s);
            assert_eq!(chosen(&r).predicted, r.arrow_size, "{}", r.arrow_type);
        }
    }

    /// The inner level's choice for a list column, as `recommend` makes it.
    fn inner_chosen(s: &Series) -> Chosen {
        let d = describe_one(s, 0).unwrap();
        let inner = d.inner.as_ref().unwrap();
        let (_, child, _) = list_parts(&classic_layout(s).unwrap()).unwrap();
        let lvl = Level::of_values(
            inner.values.dtype(),
            child.clone(),
            &inner.profile,
            None,
            ipc_body_bytes(child.as_ref(), None).unwrap(),
            d.conclusions(10_000).1.unwrap().est,
            "inner: ",
        )
        .unwrap();
        choose(&lvl, &params()).unwrap()
    }

    #[test]
    fn inner_predicted_equals_measured() {
        let many: Vec<&str> = (0..200)
            .map(|i| if i % 3 == 0 { "alpha" } else { "beta" })
            .collect();
        let cases = [
            Series::new(
                "x".into(),
                [
                    Some(Series::new("".into(), &[1i64, 2])),
                    None,
                    Some(Series::new("".into(), &[300i64])),
                ],
            ),
            Series::new(
                "x".into(),
                [
                    Some(Series::new("".into(), &[Some(1.5f64), None])),
                    Some(Series::new("".into(), &[2.25f64])),
                ],
            ),
            Series::new(
                "x".into(),
                [
                    Some(Series::new("".into(), &many)),
                    None,
                    Some(Series::new("".into(), &["gamma"])),
                ],
            ),
            Series::new(
                "x".into(),
                [
                    Some(Series::new("".into(), &["1.50", "2.2"])),
                    Some(Series::new("".into(), &["3"])),
                ],
            ),
            Series::new(
                "x".into(),
                [
                    Some(Series::new(
                        "".into(),
                        &["2024-01-05T10:00+05:00", "2024-01-05T10:00-03:30"],
                    )),
                    None,
                ],
            ),
            Series::new(
                "x".into(),
                [
                    Some(Series::new("".into(), &[1i64, 2])),
                    None,
                    Some(Series::new("".into(), &[5i64, 6])),
                ],
            )
            .cast(&PT::Array(Box::new(PT::Int64), 2))
            .unwrap(),
            null_list_holding_values(vec![1, 2, 3, 4, 5], vec![0, 2, 4, 5], 1),
        ];
        for s in cases {
            let c = inner_chosen(&s);
            assert!(
                !matches!(c.target, Target::Original(_)),
                "{}",
                pa_name(c.array.data_type())
            ); // a recast, not the measured original
            assert_eq!(
                c.predicted,
                ipc_body_bytes(c.array.as_ref(), None).unwrap(),
                "{}",
                pa_name(c.array.data_type())
            );
        }
    }

    #[test]
    fn polars_types_of_results() {
        let many: Vec<&str> = (0..200)
            .map(|i| if i % 2 == 0 { "alpha" } else { "beta" })
            .collect();
        let r = rec(Series::new("x".into(), &many));
        assert_eq!(
            r.polars_type.as_deref(),
            Some("Categorical(Categories(name=\"x\", namespace=\"\", physical=pl.UInt8))")
        );
        // List of dictionary strings: Polars layout is large_list<dictionary<string_view>>, sized without error.
        let words = Series::new("x".into(), [Some(Series::new("".into(), &many)), None]);
        let r = rec(words);
        assert_eq!(
            r.polars_type.as_deref(),
            Some("List(Categorical(Categories(name=\"x\", namespace=\"\", physical=pl.UInt8)))")
        );
        assert!(r.polars_size > 0);
    }

    fn rec_with(s: Series) -> Rec {
        let (d, sz) = (describe_one(&s, 0).unwrap(), sizes(&s, 1).unwrap());
        recommend(
            &s,
            &classic_layout(&s).unwrap(),
            &d,
            d.conclusions(10_000),
            &sz,
            &params(),
        )
        .unwrap()
    }

    #[test]
    fn polars_sizes_match_polars_view_layout() {
        // Oracle: the same data cast in Polars and measured by analytics/describe/_sizes.py
        // (size_polars_bytes). Views inline ≤ 12 bytes; longer values fill 8 KiB, 16 KiB, … blocks.
        let long = |i: usize| format!("long string number {i:06}");
        let cases: Vec<(Series, &str, u64)> = vec![
            (
                Series::new("x".into(), &[Some("x"), Some("y"), Some("x"), None]),
                "Categorical(Categories(name=\"x\", namespace=\"\", physical=pl.UInt8))",
                48,
            ),
            (
                Series::new("x".into(), ["a long category value 1", "b"].repeat(40)),
                "Categorical(Categories(name=\"x\", namespace=\"\", physical=pl.UInt8))",
                136,
            ),
            (
                Series::new(
                    "x".into(),
                    [
                        Some(Series::new("".into(), &["a", "bb"])),
                        None,
                        Some(Series::new("".into(), &["a"])),
                    ],
                ),
                "List(String)",
                88,
            ),
            (
                Series::new(
                    "x".into(),
                    [
                        Some(Series::new("".into(), &["a", "this is a long string!"])),
                        None,
                        Some(Series::new("".into(), &["a"])),
                    ],
                ),
                "List(String)",
                112,
            ),
            (
                Series::new(
                    "x".into(),
                    (0..256).map(|i| format!("s{i}")).collect::<Vec<_>>(),
                ),
                "String",
                4096,
            ),
            (
                Series::new("x".into(), (0..1000).map(long).collect::<Vec<_>>()),
                "String",
                41_008,
            ),
            (
                Series::new(
                    "x".into(),
                    (0..1000)
                        .map(|i| (i % 3 != 0).then(|| long(i)))
                        .collect::<Vec<_>>(),
                ),
                "String",
                32_784,
            ),
            (
                Series::new(
                    "x".into(),
                    &[Some(&b"ab"[..]), Some(&b"0123456789abcdefg"[..]), None],
                ),
                "Binary",
                80,
            ),
        ];
        for (s, polars_type, polars_size) in cases {
            let r = rec_with(s);
            assert_eq!(
                (r.polars_type.as_deref(), r.polars_size),
                (Some(polars_type), polars_size),
                "{}",
                r.arrow_type
            );
        }
    }

    #[test]
    fn nullable_arrays_narrow() {
        let lists = Series::new(
            "x".into(),
            [
                Some(Series::new("".into(), &[1i64, 2])),
                None,
                Some(Series::new("".into(), &[5i64, 6])),
                Some(Series::new("".into(), &[7i64, 8])),
            ],
        );
        let s = lists.cast(&PT::Array(Box::new(PT::Int64), 2)).unwrap();
        let r = rec(s);
        assert_eq!(r.arrow_type, "fixed_size_list<item: uint8>[2]");
        assert_eq!(chosen(&r).predicted, r.arrow_size);
        assert_eq!(r.polars_type.as_deref(), Some("Array(UInt8, shape=(2,))"));
    }
}
