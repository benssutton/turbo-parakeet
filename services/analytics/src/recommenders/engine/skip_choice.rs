//! The skip-index choice for one column: measure `minmax` / `set(N)` per
//! granularity (and `bloom_filter` lazily, as the fallback) on the recommended array in its stored
//! order, then decide with `decide_skip`.

use super::*;
use crate::common::arrow_io::import_array;
use crate::common::clickhouse_sizes::{
    skip_index_bytes, ChLayout, IndexKind, BLOOM_BITS_PER_VALUE,
};
use crate::techniques::skip_index::{
    minmax, over_segments, population, set, windows, Common, Population,
};
use arrow_array::Array;
use arrow_schema::Field;

/// ClickHouse rows per granule.
const GRANULE_ROWS: u64 = 8_192;

/// Types on which ClickHouse 26.8 rejects `bloom_filter` (services/bench/spikes/FINDINGS.md §9).
fn bloom_unsupported(ch_type: &str) -> bool {
    let mut t = ch_type;
    while let Some(inner) = t
        .strip_prefix("Nullable(")
        .and_then(|s| s.strip_suffix(')'))
    {
        t = inner;
    }
    ["Decimal", "Date32", "Time64"]
        .iter()
        .any(|p| t.starts_with(p))
}

/// A ClickHouse `Enum8` / `Enum16` (possibly `Nullable`).
fn is_enum_type(ch_type: &str) -> bool {
    let t = ch_type.strip_prefix("Nullable(").unwrap_or(ch_type);
    t.starts_with("Enum8(") || t.starts_with("Enum16(")
}

/// The chosen index (`"<ddl> GRANULARITY <g>"`) and every candidate. `segments` are the row ranges
/// of `arr` whose windows are scanned (one-shot: the whole column); `ch` is the column's
/// ClickHouse type and layout (None: no candidates); `ch_size` its predicted compressed bytes.
/// `bytes_scale` multiplies every index size (streaming: stream rows / sampled rows; one-shot 1.0)
/// and `extra_pruned` are rows the caller already pruned (streaming: granularities beyond a block).
#[allow(clippy::too_many_arguments)]
pub(crate) fn skip_choice(
    arr: &dyn Array,
    ch: Option<(&str, &ChLayout)>,
    enum_values: Option<&[String]>,
    ch_size: Option<u64>,
    segments: &[(usize, usize)],
    p: &SkipParams,
    bytes_scale: f64,
    extra_pruned: Vec<SkipCandidate>,
) -> (Option<String>, Vec<SkipCandidate>) {
    let none = (None, Vec::new());
    let Some((ch_type, layout)) = ch else {
        return none;
    };
    // Ordered scalars only (`population` re-checks the Polars dtype); nested types never reach the import.
    if arr.data_type().is_nested() || matches!(arr.data_type(), arrow_schema::DataType::Null) {
        return none;
    }
    let Some(decoded) = decode_for_population(arr, ch_type, enum_values) else {
        return none;
    };
    let field = Field::new("x", decoded.data_type().clone(), true);
    let Ok(series) = import_array(&field, &decoded) else {
        return none;
    };
    let Ok(pop) = population(&series) else {
        return none;
    };
    if pop.d() == 0 {
        return none;
    }

    let mut pruned = extra_pruned;
    let (bloom_ok, bloom_fprs) = screen_bloom_fprs(ch_type, p, &mut pruned);
    let scan = Scan {
        pop: &pop,
        arr,
        layout,
        segments,
        bytes_scale,
    };
    let mut cheap = Vec::new();
    for &g in &p.granularities {
        cheap.push(scan.measure(Kind::MinMax, g, None));
        for &m in &p.set_max_sizes {
            cheap.push(scan.measure(Kind::Set(m), g, Some(m)));
        }
    }
    let bloom = || {
        if bloom_ok {
            scan.bloom(p, &bloom_fprs)
        } else {
            Vec::new()
        }
    };
    let rows = decide_skip(cheap, pruned, bloom, ch_size, p);
    let chosen = rows
        .iter()
        .find(|r| r.outcome == "chosen")
        .map(|r| format!("{} GRANULARITY {}", r.kind.ddl(), r.granularity));
    (chosen, rows)
}

/// The array the population ranks. A dictionary is decoded first: Polars would import it as a
/// Categorical in the process-global category mapping, and a global Categorical column's exported
/// size counts every category in it, so the size of unrelated columns would depend on which columns
/// were imported before. The population ranks values (a Categorical by string) either way.
/// An Enum column is the exception: ClickHouse compares an Enum's category codes, so it is
/// ranked by category index (its position in `enum_values`), as the skip-index techniques do.
fn decode_for_population(
    arr: &dyn Array,
    ch_type: &str,
    enum_values: Option<&[String]>,
) -> Option<arrow_array::ArrayRef> {
    let enum_index = enum_values.filter(|_| is_enum_type(ch_type));
    match (enum_index, arr.data_type()) {
        (Some(cats), _) => enum_codes(arr, cats),
        (None, arrow_schema::DataType::Dictionary(_, values)) => arrow_cast::cast(arr, values).ok(),
        _ => Some(arrow_array::make_array(arr.to_data())),
    }
}

/// An Enum column's category codes (`UInt32`); None when a value is not a category.
fn enum_codes(arr: &dyn Array, cats: &[String]) -> Option<arrow_array::ArrayRef> {
    let index: std::collections::HashMap<&str, u32> = cats
        .iter()
        .enumerate()
        .map(|(i, c)| (c.as_str(), i as u32))
        .collect();
    let text = arrow_cast::cast(arr, &arrow_schema::DataType::Utf8).ok()?;
    let text = text.as_any().downcast_ref::<arrow_array::StringArray>()?;
    let codes: arrow_array::UInt32Array = text
        .iter()
        .map(|v| v.and_then(|s| index.get(s).copied()))
        .collect();
    (codes.null_count() == text.null_count()).then(|| arrow_array::make_array(codes.to_data()))
}

/// A `bloom_filter` candidate that is pruned before measuring.
fn bloom_pruned(fpr: f64, p: &SkipParams, evidence: String) -> SkipCandidate {
    SkipCandidate {
        measured: None,
        kind: Kind::Bloom(fpr),
        granularity: p.granularities.first().copied().unwrap_or(1),
        outcome: "pruned",
        evidence,
    }
}

/// Whether `bloom_filter` applies to the column, and the false-positive rates to measure. Rates
/// on an unsupported type, or outside the measured bloom size table (no size model), go to
/// `pruned` with the rule.
fn screen_bloom_fprs(
    ch_type: &str,
    p: &SkipParams,
    pruned: &mut Vec<SkipCandidate>,
) -> (bool, Vec<f64>) {
    let bloom_ok = !bloom_unsupported(ch_type);
    if !bloom_ok {
        for &fpr in &p.bloom_fprs {
            let evidence =
                format!("bloom_filter is not supported on {ch_type} (ClickHouse 26.8 rejects it)");
            pruned.push(bloom_pruned(fpr, p, evidence));
        }
    }
    let known: Vec<f64> = BLOOM_BITS_PER_VALUE.iter().map(|(f, _)| *f).collect();
    let mut bloom_fprs = Vec::new();
    for &fpr in &p.bloom_fprs {
        if bloom_ok && !known.contains(&fpr) {
            let list: Vec<String> = known.iter().map(f64::to_string).collect();
            let evidence = format!(
                "bloom_filter fpr {fpr} not measured (measured fprs: {})",
                list.join(", ")
            );
            pruned.push(bloom_pruned(fpr, p, evidence));
        } else {
            bloom_fprs.push(fpr);
        }
    }
    (bloom_ok, bloom_fprs)
}

/// What every measurement of one column shares.
struct Scan<'a> {
    pop: &'a Population,
    arr: &'a dyn Array,
    layout: &'a ChLayout,
    segments: &'a [(usize, usize)],
    bytes_scale: f64,
}

impl Scan<'_> {
    /// One index kind at one granularity: `max_size` None is `minmax`, Some(m) a `set(m)`.
    fn measure(&self, kind: Kind, g: u32, max_size: Option<u64>) -> Measured {
        let pop = self.pop;
        let segments = self.segments;
        let block_rows = u64::from(g) * GRANULE_ROWS;
        let (w, _) = windows(pop.n(), block_rows);
        let (sums, expected, index) = match max_size {
            None => (
                over_segments(segments, w, 0, |r| minmax::scan(pop, w as usize, r)),
                minmax::baseline(pop, w),
                IndexKind::MinMax,
            ),
            Some(m) => {
                let b = set::baseline(pop, w, m);
                (
                    over_segments(segments, w, pop.d(), |r| set::scan(pop, w as usize, m, r)),
                    (b.skip_rows, b.skip_values),
                    match kind {
                        Kind::Bloom(fpr) => IndexKind::Bloom { fpr },
                        _ => IndexKind::Set { max_size: m },
                    },
                )
            }
        };
        measured(
            pop,
            w,
            &sums,
            expected,
            kind,
            g,
            index,
            self.arr,
            self.layout,
            self.bytes_scale,
        )
    }

    /// The `bloom_filter` fallback: the unlimited set's skip shares thinned by (1 − fpr).
    fn bloom(&self, p: &SkipParams, fprs: &[f64]) -> Vec<Measured> {
        let mut out = Vec::new();
        for &g in &p.granularities {
            let unlimited = self.measure(Kind::Set(0), g, Some(0));
            for &fpr in fprs {
                let keep = 1.0 - fpr;
                let bytes = skip_index_bytes(
                    self.arr,
                    self.layout,
                    IndexKind::Bloom { fpr },
                    u64::from(g) * GRANULE_ROWS,
                )
                .ok()
                .map(|b| scaled(b, self.bytes_scale));
                out.push(Measured {
                    kind: Kind::Bloom(fpr),
                    skip_rows: unlimited.skip_rows * keep,
                    skip_values: unlimited.skip_values * keep,
                    expected_skip_rows: unlimited.expected_skip_rows * keep,
                    bytes,
                    overflow: None,
                    ..unlimited.clone()
                });
            }
        }
        out
    }
}

fn scaled(bytes: u64, scale: f64) -> u64 {
    if scale == 1.0 {
        bytes
    } else {
        (bytes as f64 * scale).round() as u64
    }
}

#[allow(clippy::too_many_arguments)]
fn measured(
    pop: &Population,
    w: u64,
    sums: &crate::techniques::skip_index::WindowSums,
    expected: (f64, f64),
    kind: Kind,
    g: u32,
    index: IndexKind,
    arr: &dyn Array,
    layout: &ChLayout,
    bytes_scale: f64,
) -> Measured {
    let c = Common::new(pop, w, sums, expected);
    let per_window = |x: f64| x / sums.windows as f64;
    Measured {
        kind,
        granularity: g,
        skip_rows: c.skip_rows_mean,
        skip_values: c.skip_values_mean,
        expected_skip_rows: c.expected_skip_rows,
        bytes: match kind {
            // Bloom bytes are computed per fpr by the caller.
            Kind::Bloom(_) => None,
            _ if matches!(index, IndexKind::Set { max_size: 0 }) => None,
            _ => skip_index_bytes(arr, layout, index, u64::from(g) * GRANULE_ROWS)
                .ok()
                .map(|b| scaled(b, bytes_scale)),
        },
        overflow: matches!(kind, Kind::Set(_)).then(|| {
            (
                per_window(sums.entries as f64),
                sums.max_entries,
                per_window(sums.overflowed as f64),
            )
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::types::Int32Type;
    use arrow_array::DictionaryArray;
    use polars::prelude::{Categories, DataType, NamedFrom, Series};

    /// A global Categorical column exports every category of the process-wide mapping, so the
    /// skip-index import of a dictionary candidate must not register its values there (it made
    /// a Categorical column's size depend on what ran beside it: flaky size parity).
    #[test]
    fn dictionary_values_stay_out_of_the_global_categories() {
        let cats = Categories::global();
        // Keeps the global mapping alive (it is dropped with its last column).
        let _held = Series::new("held".into(), &["keep"])
            .cast(&DataType::Categorical(cats.clone(), cats.mapping()))
            .unwrap();
        let probes: Vec<String> = (0..3).map(|i| format!("skip-choice-probe-{i}")).collect();
        let d: DictionaryArray<Int32Type> = probes
            .iter()
            .flat_map(|p| std::iter::repeat_n(Some(p.as_str()), 50))
            .collect();
        let layout = ChLayout::LowCardinality;
        let (_, candidates) = skip_choice(
            &d,
            Some(("LowCardinality(String)", &layout)),
            None,
            Some(1_000),
            &[(0, 150)],
            &SkipParams::default(),
            1.0,
            Vec::new(),
        );
        assert!(!candidates.is_empty(), "the column was not measured");
        let global = cats.mapping();
        for p in &probes {
            assert!(global.get_cat(p).is_none(), "{p} leaked");
        }
    }
}
