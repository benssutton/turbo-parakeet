//! The stateful recommenders (OneShotRecommender, StreamingRecommender) and their parameters.

use super::*;

use arrow_array::RecordBatch;

pub use crate::recommenders::engine::EntropyEstimator;
use crate::recommenders::engine::{CompressionParams, Params, SkipParams};

/// The checks both recommenders' constructors share.
pub(crate) fn validate_common(zstd_level: i32, boolean_pairs: &[(String, String)]) -> Result<()> {
    let levels = zstd::compression_level_range();
    if !levels.contains(&zstd_level) {
        return Err(Error::InvalidInput(format!(
            "zstd_level {zstd_level} is outside {levels:?}"
        )));
    }
    if let Some((t, f)) = boolean_pairs
        .iter()
        .find(|(t, f)| t.is_empty() || f.is_empty() || t.to_lowercase() == f.to_lowercase())
    {
        return Err(Error::InvalidInput(format!(
            "boolean_pairs must be pairs of distinct non-empty strings, got ({t:?}, {f:?})"
        )));
    }
    Ok(())
}

/// The compression keywords as engine parameters, checked.
fn compression(
    zstd_levels: Vec<i32>,
    zstd_level_tolerance: f64,
    zstd_min_saving: f64,
    transform_min_saving: f64,
) -> Result<CompressionParams> {
    // ClickHouse accepts ZSTD(1) … ZSTD(22).
    let range = 1..=22;
    if zstd_levels.is_empty() {
        return Err(Error::InvalidInput("zstd_levels must be non-empty".into()));
    }
    let mut seen = std::collections::HashSet::new();
    for l in &zstd_levels {
        if !range.contains(l) {
            return Err(Error::InvalidInput(format!(
                "zstd_levels: {l} is outside {range:?}"
            )));
        }
        if !seen.insert(*l) {
            return Err(Error::InvalidInput(format!(
                "zstd_levels must be distinct, {l} repeats"
            )));
        }
    }
    for (name, v) in [
        ("zstd_level_tolerance", zstd_level_tolerance),
        ("zstd_min_saving", zstd_min_saving),
        ("transform_min_saving", transform_min_saving),
    ] {
        if !(v.is_finite() && (0.0..1.0).contains(&v)) {
            return Err(Error::InvalidInput(format!(
                "{name} must be in [0, 1), got {v}"
            )));
        }
    }
    Ok(CompressionParams {
        zstd_levels,
        zstd_level_tolerance,
        zstd_min_saving,
        transform_min_saving,
    })
}

/// The chain the ClickHouse codec rule picks from (chain, bytes) rows (`redecide`): `chains`
/// in DDL, `bytes` their sizes; the keywords checked as the recommenders check them.
/// Private: the evaluation loop's decision agreement.
pub fn clickhouse_redecide(
    chains: &[String],
    bytes: &[u64],
    zstd_level_tolerance: f64,
    zstd_min_saving: f64,
    transform_min_saving: f64,
) -> Result<String> {
    if chains.len() != bytes.len() {
        return Err(Error::InvalidInput(
            "chains and bytes differ in length".into(),
        ));
    }
    let rows = chains
        .iter()
        .zip(bytes)
        .map(|(c, &b)| super::techniques::parse_chain(c).map(|ch| (ch, b)))
        .collect::<Result<Vec<_>>>()?;
    let p = compression(
        CompressionParams::default().zstd_levels,
        zstd_level_tolerance,
        zstd_min_saving,
        transform_min_saving,
    )?;
    crate::recommenders::engine::redecide(&rows, &p)
        .map(|c| c.ddl())
        .map_err(Error::InvalidInput)
}

/// The skip-index keywords as engine parameters, checked.
fn skip_params(
    granularities: Vec<u32>,
    set_max_sizes: Vec<u64>,
    bloom_fprs: Vec<f64>,
    min_skip: f64,
    min_gain: f64,
    max_overflow: f64,
) -> Result<SkipParams> {
    fn distinct<T: PartialEq + Copy + std::fmt::Display>(name: &str, v: &[T]) -> Result<()> {
        if v.is_empty() {
            return Err(Error::InvalidInput(format!("{name} must be non-empty")));
        }
        for (i, x) in v.iter().enumerate() {
            if v[..i].contains(x) {
                return Err(Error::InvalidInput(format!(
                    "{name} must be distinct, {x} repeats"
                )));
            }
        }
        Ok(())
    }
    distinct("skip_granularities", &granularities)?;
    if granularities.contains(&0) {
        return Err(Error::InvalidInput(
            "skip_granularities must be at least 1".into(),
        ));
    }
    distinct("set_max_sizes", &set_max_sizes)?;
    if set_max_sizes.contains(&0) {
        return Err(Error::InvalidInput(
            "set_max_sizes must be at least 1".into(),
        ));
    }
    if let Some(m) = set_max_sizes.iter().find(|m| **m > u64::from(u32::MAX)) {
        return Err(Error::InvalidInput(format!(
            "set_max_sizes must be at most {}, got {m}",
            u32::MAX
        )));
    }
    distinct("bloom_fprs", &bloom_fprs)?;
    if let Some(f) = bloom_fprs
        .iter()
        .find(|f| !(f.is_finite() && **f > 0.0 && **f < 1.0))
    {
        return Err(Error::InvalidInput(format!(
            "bloom_fprs must be in (0, 1), got {f}"
        )));
    }
    for (name, v) in [
        ("min_skip", min_skip),
        ("min_gain", min_gain),
        ("max_overflow", max_overflow),
    ] {
        if !(v.is_finite() && (0.0..=1.0).contains(&v)) {
            return Err(Error::InvalidInput(format!(
                "{name} must be in [0, 1], got {v}"
            )));
        }
    }
    Ok(SkipParams {
        granularities,
        set_max_sizes,
        bloom_fprs,
        min_skip,
        min_gain,
        max_overflow,
    })
}

/// Duplicate id column names are InvalidInput.
fn validate_ids(id_columns: &[String]) -> Result<()> {
    let mut seen = std::collections::HashSet::new();
    match id_columns.iter().find(|n| !seen.insert(n.as_str())) {
        Some(n) => Err(Error::InvalidInput(format!("duplicate id column {n:?}"))),
        None => Ok(()),
    }
}

/// Keywords of the one-shot recommender.
#[derive(Clone, Debug)]
pub struct OneShotParams {
    pub categorical_threshold: u64,
    pub zstd_level: i32,
    pub seed: u64,
    pub boolean_pairs: Vec<(String, String)>,
    /// Entries per `top_k` / `inner_top_k` cell: 0 none, u64::MAX every ranked value.
    pub top_k: u64,
    /// Columns identifying the exemplar rows (typed `exemplar_id` structs); may be empty.
    pub id_columns: Vec<String>,
    /// ZSTD levels tried for the Arrow IPC / ClickHouse choices.
    pub zstd_levels: Vec<i32>,
    pub zstd_level_tolerance: f64,
    pub zstd_min_saving: f64,
    pub transform_min_saving: f64,
    /// Skip-index keywords.
    pub skip_granularities: Vec<u32>,
    pub set_max_sizes: Vec<u64>,
    pub bloom_fprs: Vec<f64>,
    pub min_skip: f64,
    pub min_gain: f64,
    pub max_overflow: f64,
}

/// Recommends dtypes for one frame from exact statistics, each candidate verified on
/// every row; all state stays in Rust.
pub struct OneShotRecommender(crate::recommenders::oneshot::OneShot);

impl OneShotRecommender {
    pub fn new(p: OneShotParams) -> Result<Self> {
        validate_common(p.zstd_level, &p.boolean_pairs)?;
        validate_ids(&p.id_columns)?;
        let compression = compression(
            p.zstd_levels,
            p.zstd_level_tolerance,
            p.zstd_min_saving,
            p.transform_min_saving,
        )?;
        let skip = skip_params(
            p.skip_granularities,
            p.set_max_sizes,
            p.bloom_fprs,
            p.min_skip,
            p.min_gain,
            p.max_overflow,
        )?;
        Ok(Self(crate::recommenders::oneshot::OneShot::new(Params {
            seed: p.seed,
            zstd_level: p.zstd_level,
            categorical_threshold: p.categorical_threshold,
            boolean_pairs: p.boolean_pairs,
            top_k: p.top_k,
            entropy_estimator: EntropyEstimator::Sample,
            id_columns: p.id_columns,
            compression,
            skip,
        })))
    }

    /// Adds the frame and collects its statistics; a second call is InvalidInput. On
    /// error the state is unchanged.
    pub fn add(&mut self, batch: &RecordBatch) -> Result<()> {
        self.0.add(batch)
    }

    /// Marks a column the caller cannot send (Int128 / UInt128, Object) as ineligible;
    /// before `add` only.
    #[cfg_attr(not(feature = "python"), allow(dead_code))] // only the Python binding calls it
    pub fn mark_ineligible(&mut self, name: &str, dtype: &str) -> Result<()> {
        self.0.mark_ineligible(name, dtype)
    }

    /// One row per column; computed on the first call, then cached.
    pub fn result(&self) -> Result<RecordBatch> {
        self.0.result()
    }
}

/// Keywords of the streaming recommender.
#[derive(Clone, Debug)]
pub struct StreamingParams {
    /// Rows of contiguous blocks kept for ZSTD sizes and the cross-check (0: none).
    pub reservoir_rows: u64,
    pub block_rows: u64,
    pub categorical_threshold: u64,
    pub zstd_level: i32,
    pub seed: u64,
    pub boolean_pairs: Vec<(String, String)>,
    /// Entries per `top_k` / `inner_top_k` cell: 0 none, u64::MAX every ranked value.
    pub top_k: u64,
    /// The entropy estimator past the exact phase (the bindings default to `Ams`).
    pub entropy_estimator: EntropyEstimator,
    /// Columns identifying the exemplar rows (typed `exemplar_id` structs); may be empty.
    pub id_columns: Vec<String>,
    /// ZSTD levels tried for the Arrow IPC / ClickHouse choices.
    pub zstd_levels: Vec<i32>,
    pub zstd_level_tolerance: f64,
    pub zstd_min_saving: f64,
    pub transform_min_saving: f64,
    /// Skip-index keywords.
    pub skip_granularities: Vec<u32>,
    pub set_max_sizes: Vec<u64>,
    pub bloom_fprs: Vec<f64>,
    pub min_skip: f64,
    pub min_gain: f64,
    pub max_overflow: f64,
}

/// Recommends dtypes from record batches added over time; all state stays in Rust.
pub struct StreamingRecommender(crate::recommenders::streaming::Streaming);

impl StreamingRecommender {
    pub fn new(p: StreamingParams) -> Result<Self> {
        if p.block_rows == 0 {
            return Err(Error::InvalidInput("block_rows must be at least 1".into()));
        }
        if p.reservoir_rows != 0 && p.reservoir_rows < p.block_rows {
            return Err(Error::InvalidInput(format!(
                "reservoir_rows {} is below block_rows {}: use 0 (no sample) or at least one block",
                p.reservoir_rows, p.block_rows
            )));
        }
        validate_common(p.zstd_level, &p.boolean_pairs)?;
        validate_ids(&p.id_columns)?;
        let compression = compression(
            p.zstd_levels,
            p.zstd_level_tolerance,
            p.zstd_min_saving,
            p.transform_min_saving,
        )?;
        let skip = skip_params(
            p.skip_granularities,
            p.set_max_sizes,
            p.bloom_fprs,
            p.min_skip,
            p.min_gain,
            p.max_overflow,
        )?;
        let params = Params {
            seed: p.seed,
            zstd_level: p.zstd_level,
            categorical_threshold: p.categorical_threshold,
            boolean_pairs: p.boolean_pairs,
            top_k: p.top_k,
            entropy_estimator: p.entropy_estimator,
            id_columns: p.id_columns,
            compression,
            skip,
        };
        Ok(Self(crate::recommenders::streaming::Streaming::new(
            params,
            p.reservoir_rows,
            p.block_rows,
        )))
    }

    /// Adds one batch. On error the state is unchanged.
    pub fn add(&mut self, batch: &RecordBatch) -> Result<()> {
        self.0.add(batch)
    }

    /// Marks a column the caller cannot send (Int128 / UInt128, Object) as ineligible.
    pub fn mark_ineligible(&mut self, name: &str, dtype: &str) -> Result<()> {
        self.0.mark_ineligible(name, dtype)
    }

    /// The recommendation for every column seen so far; the state is kept.
    pub fn result(&self) -> Result<RecordBatch> {
        self.0.result()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::cast::AsArray;

    use arrow_array::{ArrayRef, Int64Array, StringArray};

    use super::*;

    fn ints(v: &[i64]) -> ArrayRef {
        Arc::new(Int64Array::from(v.to_vec()))
    }

    fn batch(columns: Vec<(&str, ArrayRef)>) -> RecordBatch {
        RecordBatch::try_from_iter(columns).unwrap()
    }

    fn streaming_params() -> StreamingParams {
        StreamingParams {
            reservoir_rows: 1 << 20,
            block_rows: 1 << 16,
            categorical_threshold: 10_000,
            zstd_level: 1,
            seed: 0,
            boolean_pairs: vec![("true".into(), "false".into())],
            top_k: 256,
            entropy_estimator: EntropyEstimator::Sample,
            id_columns: Vec::new(),
            zstd_levels: vec![1, 3, 9],
            zstd_level_tolerance: 0.01,
            zstd_min_saving: 0.10,
            transform_min_saving: 0.01,
            skip_granularities: vec![1, 2, 4, 8],
            set_max_sizes: vec![125, 250],
            bloom_fprs: vec![0.025],
            min_skip: 0.3,
            min_gain: 0.1,
            max_overflow: 0.1,
        }
    }

    #[test]
    fn streaming_parameters_are_validated() {
        let bad = [
            StreamingParams {
                block_rows: 0,
                ..streaming_params()
            },
            StreamingParams {
                reservoir_rows: 10,
                block_rows: 100,
                ..streaming_params()
            },
            StreamingParams {
                zstd_level: 99,
                ..streaming_params()
            },
            StreamingParams {
                boolean_pairs: vec![("Y".into(), "y".into())],
                ..streaming_params()
            },
            StreamingParams {
                boolean_pairs: vec![("".into(), "n".into())],
                ..streaming_params()
            },
            StreamingParams {
                zstd_levels: vec![],
                ..streaming_params()
            },
            StreamingParams {
                zstd_levels: vec![3, 3],
                ..streaming_params()
            },
            StreamingParams {
                zstd_levels: vec![99],
                ..streaming_params()
            },
            StreamingParams {
                zstd_levels: vec![0],
                ..streaming_params()
            },
            StreamingParams {
                zstd_levels: vec![23],
                ..streaming_params()
            },
            StreamingParams {
                zstd_min_saving: 1.0,
                ..streaming_params()
            },
            StreamingParams {
                transform_min_saving: -0.1,
                ..streaming_params()
            },
            StreamingParams {
                zstd_level_tolerance: f64::NAN,
                ..streaming_params()
            },
            StreamingParams {
                skip_granularities: vec![],
                ..streaming_params()
            },
            StreamingParams {
                skip_granularities: vec![1, 1],
                ..streaming_params()
            },
            StreamingParams {
                skip_granularities: vec![0],
                ..streaming_params()
            },
            StreamingParams {
                set_max_sizes: vec![0],
                ..streaming_params()
            },
            StreamingParams {
                bloom_fprs: vec![1.0],
                ..streaming_params()
            },
            StreamingParams {
                min_skip: 1.5,
                ..streaming_params()
            },
            StreamingParams {
                max_overflow: -0.1,
                ..streaming_params()
            },
        ];
        for p in bad {
            assert!(
                matches!(
                    StreamingRecommender::new(p.clone()),
                    Err(Error::InvalidInput(_))
                ),
                "{p:?}"
            );
        }
        assert!(StreamingRecommender::new(StreamingParams {
            reservoir_rows: 0,
            ..streaming_params()
        })
        .is_ok());
    }

    fn oneshot_params() -> OneShotParams {
        OneShotParams {
            categorical_threshold: 10_000,
            zstd_level: 1,
            seed: 0,
            boolean_pairs: vec![("true".into(), "false".into())],
            top_k: 256,
            id_columns: Vec::new(),
            zstd_levels: vec![1, 3, 9],
            zstd_level_tolerance: 0.01,
            zstd_min_saving: 0.10,
            transform_min_saving: 0.01,
            skip_granularities: vec![1, 2, 4, 8],
            set_max_sizes: vec![125, 250],
            bloom_fprs: vec![0.025],
            min_skip: 0.3,
            min_gain: 0.1,
            max_overflow: 0.1,
        }
    }

    #[test]
    fn oneshot_parameters_are_validated() {
        let bad = [
            OneShotParams {
                zstd_level: 99,
                ..oneshot_params()
            },
            OneShotParams {
                boolean_pairs: vec![("Y".into(), "y".into())],
                ..oneshot_params()
            },
            OneShotParams {
                boolean_pairs: vec![("".into(), "n".into())],
                ..oneshot_params()
            },
            OneShotParams {
                zstd_levels: vec![],
                ..oneshot_params()
            },
            OneShotParams {
                zstd_min_saving: 1.0,
                ..oneshot_params()
            },
            OneShotParams {
                bloom_fprs: vec![0.0],
                ..oneshot_params()
            },
            OneShotParams {
                min_gain: f64::NAN,
                ..oneshot_params()
            },
        ];
        for p in bad {
            assert!(
                matches!(
                    OneShotRecommender::new(p.clone()),
                    Err(Error::InvalidInput(_))
                ),
                "{p:?}"
            );
        }
    }

    #[test]
    fn oneshot_round_trip() {
        let b = batch(vec![
            ("a", ints(&[0, 5, 7])),
            (
                "s",
                Arc::new(StringArray::from(vec!["x", "y", "x"])) as ArrayRef,
            ),
        ]);
        let mut rec = OneShotRecommender::new(oneshot_params()).unwrap();
        rec.add(&b).unwrap();
        let out = rec.result().unwrap();
        assert_eq!(
            out.column_by_name("datafusion_type")
                .unwrap()
                .as_string_view()
                .value(0),
            "uint8"
        );
        assert_eq!(rec.result().unwrap(), out);
        assert!(matches!(rec.add(&b), Err(Error::InvalidInput(_))));
    }

    #[test]
    fn streaming_round_trip() {
        let batch = RecordBatch::try_from_iter(vec![
            ("a", ints(&[0, 5, 7])),
            (
                "s",
                Arc::new(StringArray::from(vec!["x", "y", "x"])) as ArrayRef,
            ),
        ])
        .unwrap();
        let mut rec = StreamingRecommender::new(streaming_params()).unwrap();
        rec.add(&batch).unwrap();
        rec.add(&batch).unwrap();
        let out = rec.result().unwrap();
        assert_eq!(out.num_rows(), 2);
        let types = out
            .column_by_name("datafusion_type")
            .unwrap()
            .as_string_view();
        assert_eq!(types.value(0), "uint8");
        let n_rows = out.column_by_name("n_rows").unwrap();
        assert_eq!(
            n_rows
                .as_primitive::<arrow_array::types::UInt64Type>()
                .value(0),
            6
        );
    }
}
