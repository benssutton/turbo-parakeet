//! The skip-index choice for one column (spec 2026-10-20 §2–§5): measure `minmax` / `set(N)` per
//! granularity (and `bloom_filter` lazily, as the fallback) on the recommended array in its stored
//! order, then decide with `decide_skip`.

use super::*;
use crate::common::arrow_io::import_array;
use crate::common::clickhouse_sizes::{skip_index_bytes, ChLayout, IndexKind};
use crate::techniques::skip_index::{
    minmax, over_segments, population, set, windows, Common, Population,
};
use arrow_array::Array;
use arrow_schema::Field;

/// ClickHouse rows per granule.
const GRANULE_ROWS: u64 = 8_192;

/// Chbench R36: types on which ClickHouse rejects `bloom_filter`.
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

/// The chosen index (`"<ddl> GRANULARITY <g>"`) and every candidate. `segments` are the row ranges
/// of `arr` whose windows are scanned (one-shot: the whole column); `ch` is the column's
/// ClickHouse type and layout (None: no candidates); `ch_size` its predicted compressed bytes.
/// `bytes_scale` multiplies every index size (streaming: stream rows / sampled rows; one-shot 1.0)
/// and `extra_pruned` are rows the caller already pruned (streaming: granularities beyond a block).
pub(crate) fn skip_choice(
    arr: &dyn Array,
    ch: Option<(&str, &ChLayout)>,
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
    let field = Field::new("x", arr.data_type().clone(), true);
    let Ok(series) = import_array(&field, &arrow_array::make_array(arr.to_data())) else {
        return none;
    };
    let Ok(pop) = population(&series) else {
        return none;
    };
    if pop.d() == 0 {
        return none;
    }

    let mut pruned = extra_pruned;
    let bloom_ok = !bloom_unsupported(ch_type);
    if !bloom_ok {
        for &fpr in &p.bloom_fprs {
            pruned.push(SkipCandidate {
                measured: None,
                kind: Kind::Bloom(fpr),
                granularity: p.granularities.first().copied().unwrap_or(1),
                outcome: "pruned",
                evidence: format!("bloom_filter is not supported on {ch_type} (chbench R36)"),
            });
        }
    }

    let n = pop.n();
    let measure = |kind: Kind, g: u32, max_size: Option<u64>| -> Measured {
        let block_rows = u64::from(g) * GRANULE_ROWS;
        let (w, _) = windows(n, block_rows);
        let (sums, expected, index) = match max_size {
            None => (
                over_segments(segments, w, 0, |r| minmax::scan(&pop, w as usize, r)),
                minmax::baseline(&pop, w),
                IndexKind::MinMax,
            ),
            Some(m) => {
                let b = set::baseline(&pop, w, m);
                (
                    over_segments(segments, w, pop.d(), |r| set::scan(&pop, w as usize, m, r)),
                    (b.skip_rows, b.skip_values),
                    match kind {
                        Kind::Bloom(fpr) => IndexKind::Bloom { fpr },
                        _ => IndexKind::Set { max_size: m },
                    },
                )
            }
        };
        measured(
            &pop,
            w,
            &sums,
            expected,
            kind,
            g,
            index,
            arr,
            layout,
            bytes_scale,
        )
    };

    let mut cheap = Vec::new();
    for &g in &p.granularities {
        cheap.push(measure(Kind::MinMax, g, None));
        for &m in &p.set_max_sizes {
            cheap.push(measure(Kind::Set(m), g, Some(m)));
        }
    }
    let bloom = || {
        if !bloom_ok {
            return Vec::new();
        }
        let mut out = Vec::new();
        for &g in &p.granularities {
            let unlimited = measure(Kind::Set(0), g, Some(0));
            for &fpr in &p.bloom_fprs {
                let keep = 1.0 - fpr;
                let bytes = skip_index_bytes(
                    arr,
                    layout,
                    IndexKind::Bloom { fpr },
                    u64::from(g) * GRANULE_ROWS,
                )
                .ok()
                .map(|b| scaled(b, bytes_scale));
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
    };
    let rows = decide_skip(cheap, pruned, bloom, ch_size, p);
    let chosen = rows
        .iter()
        .find(|r| r.outcome == "chosen")
        .map(|r| format!("{} GRANULARITY {}", r.kind.ddl(), r.granularity));
    (chosen, rows)
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
