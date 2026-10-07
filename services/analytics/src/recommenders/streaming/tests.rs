use std::sync::Arc;

use arrow_array::builder::{FixedSizeListBuilder, Int64Builder};
use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{
    new_null_array, Array, ArrayRef, FixedSizeListArray, Float64Array, Int64Array, ListArray,
    NullArray, StringArray,
};
use arrow_schema::DataType as AT;
use arrow_select::concat::concat;

use crate::common::arrow_io::export_series;
use crate::common::text::arrow_cast;
use polars::prelude::{CompatLevel, IntoSeries, NamedFrom};

use super::*;

pub(crate) fn params() -> Params {
    Params {
        seed: 0,
        zstd_level: 1,
        categorical_threshold: 10_000,
        boolean_pairs: vec![("true".into(), "false".into())],
        top_k: 256,
    }
}

pub(crate) fn batch(cols: Vec<(&str, ArrayRef)>) -> RecordBatch {
    RecordBatch::try_from_iter(cols).unwrap()
}

pub(crate) fn ints(v: &[Option<i64>]) -> ArrayRef {
    Arc::new(Int64Array::from(v.to_vec()))
}

pub(crate) fn strs(v: &[Option<&str>]) -> ArrayRef {
    Arc::new(StringArray::from(v.to_vec()))
}

pub(crate) fn streaming() -> Streaming {
    Streaming::new(params(), 1 << 20, 1 << 16)
}

fn col<'a>(s: &'a Streaming, name: &str) -> &'a Column {
    &s.columns[s.index[name]]
}

#[test]
fn a_new_column_is_backfilled_with_nulls() {
    let mut s = streaming();
    s.add(&batch(vec![("a", ints(&[Some(1), Some(2)]))]))
        .unwrap();
    s.add(&batch(vec![
        ("a", ints(&[Some(3)])),
        ("b", strs(&[Some("x")])),
    ]))
    .unwrap();
    let b = col(&s, "b");
    assert_eq!((b.first_row, b.outer.n, b.outer.n_null), (2, 3, 2));
}

#[test]
fn an_absent_column_counts_as_null() {
    let mut s = streaming();
    s.add(&batch(vec![
        ("a", ints(&[Some(1)])),
        ("b", ints(&[Some(1)])),
    ]))
    .unwrap();
    s.add(&batch(vec![("a", ints(&[Some(2), Some(3)]))]))
        .unwrap();
    let b = col(&s, "b");
    assert_eq!((b.outer.n, b.outer.n_null), (3, 2));
}

#[test]
fn a_null_typed_column_adopts_the_first_concrete_type() {
    let mut s = streaming();
    s.add(&batch(vec![("b", Arc::new(NullArray::new(2)) as ArrayRef)]))
        .unwrap();
    assert!(col(&s, "b").dtype.is_none());
    s.add(&batch(vec![("b", strs(&[Some("x")]))])).unwrap();
    let b = col(&s, "b");
    assert_eq!(b.dtype, Some(PT::String));
    assert_eq!((b.outer.n, b.outer.n_null), (3, 2));
}

#[test]
fn a_type_change_is_rejected_and_changes_nothing() {
    let mut s = streaming();
    s.add(&batch(vec![("a", ints(&[Some(1)]))])).unwrap();
    let err = s
        .add(&batch(vec![
            ("a", strs(&[Some("x")])),
            ("c", ints(&[Some(1)])),
        ]))
        .unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput(ref m) if m.contains("type changed")),
        "{err:?}"
    );
    assert_eq!((s.n_rows, s.columns.len(), col(&s, "a").outer.n), (1, 1, 1));
}

#[test]
fn a_duplicate_column_is_rejected() {
    let mut s = streaming();
    let err = s
        .add(&batch(vec![
            ("a", ints(&[Some(1)])),
            ("a", ints(&[Some(2)])),
        ]))
        .unwrap_err();
    assert_eq!(err, Error::InvalidInput("duplicate column \"a\"".into()));
}

#[test]
fn ineligible_columns_are_kept_apart() {
    let mut s = streaming();
    s.mark_ineligible("w", "Int128").unwrap();
    s.add(&batch(vec![("a", ints(&[Some(1)]))])).unwrap();
    assert!(col(&s, "w").ineligible);
    assert!(s.add(&batch(vec![("w", ints(&[Some(1)]))])).is_err());
    assert!(s.mark_ineligible("a", "Object").is_err());
}

#[test]
fn a_nested_null_column_is_ineligible() {
    // One-shot lists a column holding Null below the top as ineligible; Polars
    // exports such a column's Null level with a buffer arrow-rs cannot import.
    let item = Arc::new(arrow_schema::Field::new("item", AT::Null, true));
    let nulls: ArrayRef = Arc::new(ListArray::new(
        item,
        arrow_buffer::OffsetBuffer::from_lengths([1, 0]),
        Arc::new(NullArray::new(1)),
        Some(arrow_buffer::NullBuffer::from(vec![true, false])),
    ));
    let b = || batch(vec![("n", nulls.clone()), ("a", ints(&[Some(1), Some(2)]))]);
    let mut s = streaming();
    s.add(&b()).unwrap();
    s.add(&b()).unwrap();
    let n = col(&s, "n");
    assert!(n.ineligible);
    assert_eq!(n.input_type, "list<item: null>");
    assert_eq!(col(&s, "a").outer.n, 4);
    let out = s.result().unwrap();
    assert_eq!(
        texts(&out, "status"),
        vec![Some("ineligible".into()), Some("computed".into())]
    );
    // A column already typed cannot turn into one.
    let mut s = streaming();
    s.add(&batch(vec![("n", ints(&[Some(1)]))])).unwrap();
    let err = s.add(&b()).unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput(ref m) if m.contains("type changed")),
        "{err:?}"
    );
    assert_eq!(s.n_rows, 1);
}

fn one_shot(b: &RecordBatch) -> RecordBatch {
    let mut r = crate::recommenders::oneshot::OneShot::new(params());
    r.add(b).unwrap();
    r.result().unwrap()
}

pub(crate) fn texts(b: &RecordBatch, name: &str) -> Vec<Option<String>> {
    let a = arrow_cast(b.column_by_name(name).unwrap().as_ref(), &AT::Utf8).unwrap();
    a.as_string::<i32>()
        .iter()
        .map(|v| v.map(str::to_owned))
        .collect()
}

fn mixed() -> RecordBatch {
    let list = ListArray::from_iter_primitive::<Int64Type, _, _>(vec![
        Some(vec![Some(1)]),
        Some(vec![Some(2)]),
        None,
        Some(vec![Some(3)]),
        Some(vec![Some(4)]),
        Some(vec![Some(500)]),
    ]);
    batch(vec![
        (
            "i",
            ints(&[Some(1), None, Some(300), Some(-5), Some(7), Some(7)]),
        ),
        (
            "s_num",
            strs(&[
                Some("1.50"),
                Some("2.25"),
                None,
                Some("3"),
                Some("4.5"),
                Some("5"),
            ]),
        ),
        (
            "s_cat",
            strs(&[
                Some("red"),
                Some("green"),
                Some("red"),
                None,
                Some("blue"),
                Some("a long value over twelve bytes"),
            ]),
        ),
        (
            "s_dt",
            strs(&[
                Some("2024-01-01T10:00:00"),
                Some("2024-01-02T11:30:00"),
                None,
                Some("2024-01-03T00:00:00"),
                Some("2024-01-04T12:00:00.250"),
                Some("2024-01-05 13:00:00"),
            ]),
        ),
        (
            "f",
            Arc::new(Float64Array::from(vec![
                Some(1.5),
                Some(-0.0),
                None,
                Some(2.25),
                Some(0.5),
                Some(3.0),
            ])) as ArrayRef,
        ),
        ("l", Arc::new(list) as ArrayRef),
        // Kept in its original type by construction: beyond float32's range and
        // with ~200 integer digits, no narrower float or decimal holds it.
        (
            "f_orig",
            Arc::new(Float64Array::from(
                (0..6)
                    .map(|i| (i % 3 != 1).then(|| 3.3e200 * (i + 1) as f64 / 7.0))
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
        ),
        // Values over 12 bytes: Polars packs them into view blocks.
        (
            "free",
            strs(&[
                Some("free text value number one xx"),
                None,
                Some("free text value number three xx"),
                Some("free text value number four xx"),
                Some("free text value number five xx"),
                Some("free text value number six xx"),
            ]),
        ),
        ("arr", Arc::new(fixed_list()) as ArrayRef),
    ])
}

/// Array[Int64, 2] with a null row (its slots hold values) and a null item.
fn fixed_list() -> FixedSizeListArray {
    let mut b = FixedSizeListBuilder::new(Int64Builder::new(), 2);
    for i in 0..6i64 {
        b.values().append_value(i * 1_000_000_007);
        if i == 4 {
            b.values().append_null();
        } else {
            b.values().append_value(i);
        }
        b.append(i != 2);
    }
    b.finish()
}

const REC: [&str; 12] = [
    "rec_arrow_type",
    "rec_nullable",
    "rec_lossy_formatting",
    "rec_arrow_size_bytes",
    "rec_polars_size_bytes",
    "rec_arrow_size_zstd_bytes",
    "rec_polars_size_zstd_bytes",
    "rec_polars_type",
    "size_bytes",
    "size_polars_bytes",
    "size_zstd_bytes",
    "size_polars_zstd_bytes",
];

/// `batches` streamed recommend and size like `whole` in one shot.
fn assert_like_one_shot(batches: &[RecordBatch], whole: &RecordBatch, label: &str) {
    assert_like_one_shot_but(batches, whole, label, &[]);
}

/// As `assert_like_one_shot`, the fields `skip` aside.
fn assert_like_one_shot_but(
    batches: &[RecordBatch],
    whole: &RecordBatch,
    label: &str,
    skip: &[&str],
) {
    let reference = one_shot(whole);
    let mut s = streaming();
    batches.iter().for_each(|b| s.add(b).unwrap());
    let out = s.result().unwrap();
    let columns = texts(&out, "column");
    assert_eq!(columns, texts(&reference, "column"), "{label}");
    for name in REC.into_iter().filter(|n| !skip.contains(n)) {
        let (got, want) = (texts(&out, name), texts(&reference, name));
        for (row, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(g, w, "{label} {:?} {name}", columns[row]);
        }
    }
    let rows: u64 = batches.iter().map(|b| b.num_rows() as u64).sum();
    assert_eq!(
        texts(&out, "n_sampled_rows")[0],
        Some(rows.to_string()),
        "{label}"
    );
}

#[test]
fn output_is_the_shared_streaming_schema() {
    let mut s = streaming();
    s.add(&mixed()).unwrap();
    let out = s.result().unwrap();
    let names: Vec<String> = out
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();
    let want: Vec<String> = crate::recommenders::schema::recommender_fields(true)
        .into_iter()
        .map(|(n, _)| n)
        .collect();
    assert_eq!(names, want);
}

#[test]
fn inner_columns_match_one_shot() {
    let whole = mixed();
    let reference = one_shot(&whole);
    let mut s = streaming();
    s.add(&whole.slice(0, 4)).unwrap();
    s.add(&whole.slice(4, 2)).unwrap();
    let out = s.result().unwrap();
    for name in [
        "inner_n_values",
        "inner_n_null",
        "inner_n_unique",
        "inner_unique",
        "inner_class",
        "inner_min",
        "inner_max",
        "inner_min_len",
        "inner_max_len",
        "inner_sum_len",
        "inner_gcd",
    ] {
        assert_eq!(texts(&out, name), texts(&reference, name), "{name}");
    }
    let columns = texts(&out, "column");
    let l = columns
        .iter()
        .position(|c| c.as_deref() == Some("l"))
        .unwrap();
    assert_eq!(texts(&out, "inner_n_values")[l].as_deref(), Some("5"));
}

#[test]
fn streamed_batches_recommend_like_one_shot() {
    let whole = mixed();
    for k in [1, 2, 6] {
        let batches: Vec<_> = (0..whole.num_rows())
            .step_by(k)
            .map(|off| whole.slice(off, k.min(whole.num_rows() - off)))
            .collect();
        assert_like_one_shot(&batches, &whole, &format!("k={k}"));
    }
}

/// A frame as Polars exports it through `__arrow_c_stream__`: string_view columns
/// whose data blocks Polars built (8 KiB, doubling; only values over 12 bytes).
fn polars_views_frame() -> RecordBatch {
    let text = |i: usize| match i % 3 {
        0 => None,
        1 => Some(format!("s{}", i % 7)),
        _ => Some(format!("a value longer than twelve bytes {i}")),
    };
    let n = 900;
    let s = Series::new("s".into(), (0..n).map(text).collect::<Vec<_>>());
    let l = Series::new(
        "l".into(),
        (0..n)
            .map(|i| {
                (!i.is_multiple_of(5))
                    .then(|| Series::new("".into(), (i..i + i % 3).map(text).collect::<Vec<_>>()))
            })
            .collect::<Vec<_>>(),
    );
    let cols: Vec<(&str, ArrayRef)> = [&s, &l]
        .into_iter()
        .map(|s| {
            (
                s.name().as_str(),
                export_series(s, CompatLevel::newest()).unwrap(),
            )
        })
        .collect();
    assert_eq!(cols[0].1.data_type(), &AT::Utf8View);
    batch(cols)
}

/// `columns` as Polars exports them: native layout plus Polars' field metadata
/// (which restores Categorical / Enum on import).
fn polars_frame(columns: &[Series]) -> RecordBatch {
    let fields: Vec<(arrow_schema::Field, ArrayRef)> = columns
        .iter()
        .map(|s| {
            let a = export_series(s, CompatLevel::newest()).unwrap();
            let md: HashMap<String, String> = s
                .field()
                .to_arrow(CompatLevel::newest())
                .metadata
                .as_deref()
                .map(|m| {
                    m.iter()
                        .map(|(k, v)| (k.to_string(), v.to_string()))
                        .collect()
                })
                .unwrap_or_default();
            let f = arrow_schema::Field::new(s.name().as_str(), a.data_type().clone(), true)
                .with_metadata(md);
            (f, a)
        })
        .collect();
    let (fields, arrays): (Vec<_>, Vec<_>) = fields.into_iter().unzip();
    RecordBatch::try_new(Arc::new(arrow_schema::Schema::new(fields)), arrays).unwrap()
}

/// The original candidate's predicted size, per column.
fn original_predicted(b: &RecordBatch) -> Vec<Option<String>> {
    let col = b.column_by_name("rec_candidates").unwrap().as_list::<i64>();
    (0..b.num_rows())
        .map(|row| {
            let v = col.value(row);
            let s = v.as_struct();
            let text =
                |f: &str| arrow_cast(s.column_by_name(f).unwrap().as_ref(), &AT::Utf8).unwrap();
            let (rule, predicted) = (text("rule"), text("predicted_bytes"));
            let (rule, predicted) = (rule.as_string::<i32>(), predicted.as_string::<i32>());
            (0..s.len())
                .find(|&i| rule.value(i) == "original")
                .map(|i| predicted.value(i).to_string())
        })
        .collect()
}

#[test]
fn dictionary_sources_size_like_one_shot() {
    use polars::prelude::{CategoricalPhysical, Categories, FrozenCategories};
    let long = "a category name longer than twelve bytes";
    let values: Vec<Option<&str>> = (0..60)
        .map(|i| match i % 5 {
            0 => None,
            1 | 2 => Some("b"),
            3 => Some("a"),
            _ => Some(long),
        })
        .collect();
    let e = PT::from_frozen_categories(
        FrozenCategories::new(["b", "a", "unused", long, "another long unused category"]).unwrap(),
    );
    let cats = Categories::new(
        "streaming-dictionary-test".into(),
        "".into(),
        CategoricalPhysical::U32,
    );
    let c = PT::Categorical(cats.clone(), cats.mapping());
    let series = |name: &str, dt: &PT| Series::new(name.into(), &values).cast(dt).unwrap();
    let l = Series::new(
        "l".into(),
        (0..60)
            .map(|i| {
                Series::new("".into(), &values[i..(i + i % 3).min(60)])
                    .cast(&e)
                    .unwrap()
            })
            .collect::<Vec<_>>(),
    );
    let whole = polars_frame(&[series("enum", &e), series("cat", &c), l]);
    let reference = one_shot(&whole);
    for k in [60, 7, 1] {
        let batches: Vec<_> = (0..whole.num_rows())
            .step_by(k)
            .map(|off| whole.slice(off, k.min(whole.num_rows() - off)))
            .collect();
        assert_like_one_shot(&batches, &whole, &format!("dictionary k={k}"));
        let mut s = streaming();
        batches.iter().for_each(|b| s.add(b).unwrap());
        let out = s.result().unwrap();
        assert_eq!(
            original_predicted(&out),
            original_predicted(&reference),
            "k={k}"
        );
    }
}

#[test]
fn a_list_of_structs_keeps_its_inner_type() {
    // Rows of 0-2 structs {a: f64, b: string (0-14 bytes, some over 12)}, every
    // 5th row null: the kept inner struct has no closed-form Polars size.
    let row = |i: usize| {
        (!i.is_multiple_of(5)).then(|| {
            let k = i % 3;
            let a = Series::new("a".into(), vec![i as f64 / 7.0; k]);
            let b = Series::new("b".into(), vec!["x".repeat(i % 15); k]);
            polars::prelude::StructChunked::from_series("".into(), k, [a, b].iter())
                .unwrap()
                .into_series()
        })
    };
    let s = Series::new("s".into(), (0..100).map(row).collect::<Vec<_>>());
    let whole = polars_frame(&[s]);
    let reference = one_shot(&whole);
    assert!(texts(&reference, "rec_arrow_type")[0]
        .as_deref()
        .is_some_and(|t| t.contains("struct")));
    // One batch: every recommendation and size as one-shot's, but the original's
    // Polars ZSTD size: the input's struct field holds its views in many small
    // variadic buffers (as py-polars builds them too), each compressed on its
    // own, which the sample's rebuilt block does not reproduce.
    assert_like_one_shot_but(
        std::slice::from_ref(&whole),
        &whole,
        "one batch",
        &["size_polars_zstd_bytes"],
    );
    // Several: the same recommendation.
    let mut st = streaming();
    st.add(&whole.slice(0, 40)).unwrap();
    st.add(&whole.slice(40, 60)).unwrap();
    let out = st.result().unwrap();
    assert_eq!(
        texts(&out, "rec_arrow_type"),
        texts(&reference, "rec_arrow_type")
    );
}

/// [[{a:1}]], None, [[{a:3}, None], None], [[{a:5}]], sliced to rows 1..3: a
/// Polars Series over it holds structs whose validity starts mid-bitmap, which
/// polars-arrow exports as invalid Arrow (arrow-rs panicked on it).
fn sliced_lists_of_lists_of_structs() -> RecordBatch {
    use arrow_array::builder::{LargeListBuilder, StructBuilder};
    type B = LargeListBuilder<LargeListBuilder<StructBuilder>>;
    let sb = StructBuilder::from_fields(vec![arrow_schema::Field::new("a", AT::Int64, true)], 8);
    let mut b: B = LargeListBuilder::new(LargeListBuilder::new(sb));
    let item = |b: &mut B, v: Option<i64>| {
        let s = b.values().values();
        s.field_builder::<Int64Builder>(0).unwrap().append_option(v);
        s.append(v.is_some());
    };
    item(&mut b, Some(1));
    b.values().append(true);
    b.append(true);
    b.append(false);
    item(&mut b, Some(3));
    item(&mut b, None);
    b.values().append(true);
    b.values().append(false);
    b.append(true);
    item(&mut b, Some(5));
    b.values().append(true);
    b.append(true);
    let a: ArrayRef = Arc::new(b.finish());
    batch(vec![("c", a.slice(1, 2))])
}

#[test]
fn sliced_nested_structs_with_nulls_size_like_one_shot() {
    let b = sliced_lists_of_lists_of_structs();
    assert_like_one_shot(std::slice::from_ref(&b), &b, "sliced list<list<struct>>");
}

#[test]
fn view_input_sizes_like_one_shot() {
    let whole = polars_views_frame();
    for k in [900, 250, 7] {
        let batches: Vec<_> = (0..whole.num_rows())
            .step_by(k)
            .map(|off| whole.slice(off, k.min(whole.num_rows() - off)))
            .collect();
        assert_like_one_shot(&batches, &whole, &format!("views k={k}"));
    }
}

#[test]
fn a_column_absent_from_the_first_batches_sizes_like_its_nulls() {
    // "i" in every batch; the rest only from row 3 on.
    let full = mixed();
    let b = full.slice(3, 3);
    let first = batch(vec![("i", full.slice(0, 3).column(0).clone())]);
    let cols: Vec<(String, ArrayRef)> = full
        .schema()
        .fields()
        .iter()
        .zip(full.columns())
        .map(|(f, c)| {
            let col = if f.name() == "i" {
                c.clone()
            } else {
                let nulls = new_null_array(c.data_type(), 3);
                concat(&[nulls.as_ref(), c.slice(3, 3).as_ref()]).unwrap()
            };
            (f.name().clone(), col)
        })
        .collect();
    let whole = batch(cols.iter().map(|(n, c)| (n.as_str(), c.clone())).collect());
    assert_like_one_shot(&[first.clone(), b.clone()], &whole, "absent");
    assert_like_one_shot(
        &[first, b.slice(0, 1), b.slice(1, 2)],
        &whole,
        "absent, split",
    );
}

#[test]
fn no_sample_means_no_zstd_sizes() {
    let mut s = Streaming::new(params(), 0, 1 << 16);
    s.add(&mixed()).unwrap();
    let out = s.result().unwrap();
    assert!(texts(&out, "rec_arrow_size_zstd_bytes")
        .iter()
        .all(Option::is_none));
    assert_eq!(texts(&out, "n_sampled_blocks")[0].as_deref(), Some("0"));
    assert!(texts(&out, "rec_arrow_type").iter().all(Option::is_some));
}

/// Column "s" streamed with `categorical_threshold` 3.
fn threshold_3(v: &[Option<&str>]) -> RecordBatch {
    let mut p = params();
    p.categorical_threshold = 3;
    let mut s = Streaming::new(p, 1 << 20, 1 << 16);
    s.add(&batch(vec![("s", strs(v))])).unwrap();
    s.result().unwrap()
}

/// The first row's string→dictionary candidate outcome.
fn dictionary_outcome(out: &RecordBatch) -> String {
    let list = out
        .column_by_name("rec_candidates")
        .unwrap()
        .as_list::<i64>()
        .value(0);
    let cands = list.as_struct();
    let text = |f: &str| arrow_cast(cands.column_by_name(f).unwrap().as_ref(), &AT::Utf8).unwrap();
    let (rules, outcomes) = (text("rule"), text("outcome"));
    let (rules, outcomes) = (rules.as_string::<i32>(), outcomes.as_string::<i32>());
    let i = (0..rules.len())
        .find(|&i| rules.value(i) == "string→dictionary")
        .unwrap();
    outcomes.value(i).to_string()
}

#[test]
fn exact_count_past_the_threshold_rejects_the_dictionary() {
    let out = threshold_3(&[
        Some("a"),
        Some("b"),
        Some("c"),
        Some("d"),
        Some("e"),
        Some("a"),
    ]);
    assert_eq!(texts(&out, "n_unique")[0].as_deref(), Some("5"));
    assert_eq!(texts(&out, "est_method")[0].as_deref(), Some("observed"));
    assert_eq!(dictionary_outcome(&out), "rejected");
}

#[test]
fn overflow_rejects_the_dictionary() {
    // More distinct values than the sample holds (k = max(threshold, 1000)).
    let v: Vec<String> = (0..2_000).map(|i| format!("v{i}")).collect();
    let v: Vec<Option<&str>> = v.iter().map(|x| Some(x.as_str())).collect();
    let out = threshold_3(&v);
    let n_unique: u64 = texts(&out, "n_unique")[0]
        .as_deref()
        .unwrap()
        .parse()
        .unwrap();
    assert!(n_unique >= 1_001, "{n_unique}");
    assert_eq!(texts(&out, "est_method")[0].as_deref(), Some("hll"));
    assert_eq!(dictionary_outcome(&out), "rejected");
}

#[test]
fn shared_columns_are_filled() {
    let mut s = Streaming::new(params(), 0, 1);
    s.add(&batch(vec![(
        "a",
        ints(&[Some(0), Some(5), Some(7), Some(0), Some(5), Some(7)]),
    )]))
    .unwrap();
    let out = s.result().unwrap();
    let some = |v: &[&str]| v.iter().map(|x| Some(x.to_string())).collect::<Vec<_>>();
    assert_eq!(texts(&out, "class"), some(&["ordinal"])); // 7 ≤ 2·6
    assert_eq!(texts(&out, "min"), some(&["0"]));
    assert_eq!(texts(&out, "max"), some(&["7"]));
    assert_eq!(texts(&out, "unique"), some(&["false"]));
    assert_eq!(texts(&out, "est_method"), some(&["observed"]));
    assert_eq!(texts(&out, "n_midnight"), vec![None]);
}

#[test]
fn ineligible_null_typed_and_empty() {
    assert_eq!(streaming().result().unwrap().num_rows(), 0);
    let mut s = streaming();
    s.mark_ineligible("w", "Int128").unwrap();
    s.add(&batch(vec![
        ("a", ints(&[Some(1), Some(2)])),
        ("n", Arc::new(NullArray::new(2)) as ArrayRef),
    ]))
    .unwrap();
    let out = s.result().unwrap();
    let some = |v: &[&str]| v.iter().map(|x| Some(x.to_string())).collect::<Vec<_>>();
    assert_eq!(texts(&out, "column"), some(&["w", "a", "n"]));
    assert_eq!(
        texts(&out, "status"),
        some(&["ineligible", "computed", "ineligible"])
    );
    assert_eq!(texts(&out, "dtype")[0].as_deref(), Some("Int128"));
    assert_eq!(texts(&out, "n_null")[0], None);
    assert_eq!(texts(&out, "n_rows")[0].as_deref(), Some("2"));
}

#[test]
fn an_enum_is_named_by_its_categories() {
    use polars::prelude::FrozenCategories;
    let e = PT::from_frozen_categories(FrozenCategories::new(["b", "a"]).unwrap());
    let want = Some(vec!["b".to_string(), "a".to_string()]);
    assert_eq!(enum_categories(&e), want);
    assert_eq!(enum_categories(&PT::List(Box::new(e.clone()))), want);
    assert_eq!(enum_categories(&PT::String), None);
    let t = AT::Dictionary(Box::new(AT::UInt8), Box::new(AT::Utf8));
    assert_eq!(
        pl_name(&t, "x", enum_categories(&e).as_deref(), &AT::UInt8),
        "Enum(categories=['b', 'a'])"
    );
}

/// One-shot's rankings of `whole` (the column, then its list's inner values) vs the
/// streaming `col`'s, which saw the same rows in any batch split.
fn assert_ranks_like_one_shot(s: &Streaming, col_name: &str, whole: &ArrayRef) {
    use crate::techniques::describe::describe_one;
    let field = arrow_schema::Field::new(col_name, whole.data_type().clone(), true);
    let series = crate::common::arrow_io::import_array(&field, whole).unwrap();
    let one = describe_one(&series, 0, false, Some(10_000)).unwrap();
    let c = col(s, col_name);
    let dtype = series.dtype();
    assert_eq!(c.outer.profile(dtype).ranking, one.outer.ranking);
    if let Some(inner) = &one.inner {
        let got = c
            .inner
            .as_ref()
            .unwrap()
            .profile(inner.values.dtype())
            .ranking;
        assert!(got.is_some());
        assert_eq!(got, inner.profile.ranking);
    }
}

#[test]
fn text_levels_rank_like_one_shot() {
    use arrow_array::builder::{ListBuilder, StringBuilder};
    let v = [
        Some("b"),
        Some("a"),
        Some("b"),
        None,
        Some("c"),
        Some("a"),
        Some("b"),
    ];
    // Ties across batches: x and y both twice, x first.
    let t = [Some("x"), Some("y"), Some("y"), Some("x"), Some("z")];
    let mut lb = ListBuilder::new(StringBuilder::new());
    for row in [
        Some(vec!["q", "p"]),
        None,
        Some(vec!["p"]),
        Some(vec!["r", "q"]),
        Some(vec![]),
    ] {
        match row {
            Some(items) => {
                items.iter().for_each(|x| lb.values().append_value(x));
                lb.append(true);
            }
            None => lb.append(false),
        }
    }
    let lists: ArrayRef = Arc::new(lb.finish());
    let (sv, st): (ArrayRef, ArrayRef) = (strs(&v), strs(&t));
    let mut s = streaming();
    // One row per batch: every value is admitted from its own batch.
    for i in 0..v.len() {
        s.add(&batch(vec![("s", sv.slice(i, 1))])).unwrap();
    }
    for i in 0..t.len() {
        s.add(&batch(vec![("t", st.slice(i, 1))])).unwrap();
    }
    for i in 0..lists.len() {
        s.add(&batch(vec![("l", lists.slice(i, 1))])).unwrap();
    }
    let want = Some(vec![("b".to_string(), 3), ("a".into(), 2), ("c".into(), 1)]);
    assert_eq!(col(&s, "s").outer.profile(&PT::String).ranking, want);
    assert_ranks_like_one_shot(&s, "s", &sv);
    assert_eq!(
        col(&s, "t").outer.profile(&PT::String).ranking,
        Some(vec![("x".to_string(), 2), ("y".into(), 2), ("z".into(), 1)])
    );
    assert_ranks_like_one_shot(&s, "t", &st);
    let inner = col(&s, "l").inner.as_ref().unwrap();
    assert_eq!(
        inner.profile(&PT::String).ranking,
        Some(vec![("q".to_string(), 2), ("p".into(), 2), ("r".into(), 1)])
    );
    assert_ranks_like_one_shot(&s, "l", &lists);
    let list_type = PT::List(Box::new(PT::String));
    assert_eq!(col(&s, "l").outer.profile(&list_type).ranking, None);
}

#[test]
fn dictionary_batches_with_different_category_orders_rank_like_one_shot() {
    use arrow_array::types::Int8Type;
    use arrow_array::DictionaryArray;
    let b1: DictionaryArray<Int8Type> = vec!["p", "q", "q"].into_iter().collect();
    let b2: DictionaryArray<Int8Type> = vec!["q", "p", "q", "q"].into_iter().collect();
    // Each batch's dictionary orders its categories by first appearance: [p, q] and [q, p].
    let (a, b): (ArrayRef, ArrayRef) = (Arc::new(b1), Arc::new(b2));
    let mut s = streaming();
    s.add(&batch(vec![("d", a)])).unwrap();
    s.add(&batch(vec![("d", b)])).unwrap();
    let whole = strs(&[
        Some("p"),
        Some("q"),
        Some("q"),
        Some("q"),
        Some("p"),
        Some("q"),
        Some("q"),
    ]);
    let want = Some(vec![("q".to_string(), 5), ("p".into(), 2)]);
    let dtype = col(&s, "d").dtype.clone().unwrap();
    assert_eq!(col(&s, "d").outer.profile(&dtype).ranking, want);
    assert_ranks_like_one_shot(&s, "d", &whole);
}

/// The `name` column's map cells as (key, value) lists.
pub(crate) fn top_k(b: &RecordBatch, name: &str) -> Vec<Option<Vec<(String, u64)>>> {
    let m = b.column_by_name(name).unwrap().as_map();
    (0..m.len())
        .map(|i| {
            m.is_valid(i).then(|| {
                let e = m.value(i);
                let k = e.column(0).as_string::<i32>();
                let v = e.column(1).as_primitive::<arrow_array::types::UInt64Type>();
                (0..e.len())
                    .map(|j| (k.value(j).to_string(), v.value(j)))
                    .collect()
            })
        })
        .collect()
}

pub(crate) fn pairs(v: &[(&str, u64)]) -> Option<Vec<(String, u64)>> {
    Some(v.iter().map(|&(s, n)| (s.to_string(), n)).collect())
}

#[test]
fn top_k_streams_like_one_shot() {
    let v: Vec<Option<&str>> = [
        Some("b"),
        Some("a"),
        Some("b"),
        None,
        Some("c"),
        Some("a"),
        Some("b"),
    ]
    .repeat(3);
    // Lists with ties (p, q), whose first occurrences fall in different batches, null and
    // empty lists.
    let lists = [
        Some(vec!["p"]),
        None,
        Some(vec![]),
        Some(vec!["q"]),
        Some(vec!["q", "p"]),
        None,
        Some(vec!["r"]),
    ];
    let mut lb = arrow_array::builder::ListBuilder::new(arrow_array::builder::StringBuilder::new());
    for row in lists.iter().cycle().take(v.len()) {
        match row {
            Some(items) => {
                items.iter().for_each(|x| lb.values().append_value(x));
                lb.append(true);
            }
            None => lb.append(false),
        }
    }
    let dict: arrow_array::DictionaryArray<arrow_array::types::Int32Type> =
        v.iter().copied().collect();
    let whole = batch(vec![
        ("s", strs(&v)),
        ("i", ints(&vec![Some(1); v.len()])),
        ("l", Arc::new(lb.finish()) as ArrayRef),
        ("d", Arc::new(dict) as ArrayRef),
    ]);
    let reference = one_shot(&whole);
    for rows in [1, 5, v.len()] {
        let mut s = streaming();
        (0..v.len())
            .step_by(rows)
            .for_each(|off| s.add(&whole.slice(off, rows.min(v.len() - off))).unwrap());
        let out = s.result().unwrap();
        for name in ["top_k", "inner_top_k"] {
            assert_eq!(
                top_k(&out, name),
                top_k(&reference, name),
                "{name} rows={rows}"
            );
        }
        let words = pairs(&[("b", 9), ("a", 6), ("c", 3)]);
        assert_eq!(top_k(&out, "top_k"), [words.clone(), None, None, words]);
        assert_eq!(
            top_k(&out, "inner_top_k"),
            [None, None, pairs(&[("p", 6), ("q", 6), ("r", 3)]), None]
        );
    }
}

#[test]
fn top_k_is_null_past_the_sample() {
    let v: Vec<String> = (0..2_000).map(|i| format!("v{i}")).collect();
    let v: Vec<Option<&str>> = v.iter().map(|x| Some(x.as_str())).collect();
    assert_eq!(top_k(&threshold_3(&v), "top_k"), [None]);
}
