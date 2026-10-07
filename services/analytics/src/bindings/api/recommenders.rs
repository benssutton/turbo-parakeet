//! The stateful recommenders (OneShotRecommender, StreamingRecommender) and their parameters.

use super::*;

use arrow_array::RecordBatch;

use crate::recommenders::engine::Params;

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

/// Keywords of the one-shot recommender (spec 2026-10-04 §3.4).
#[derive(Clone, Debug)]
pub struct OneShotParams {
    pub categorical_threshold: u64,
    pub zstd_level: i32,
    pub seed: u64,
    pub boolean_pairs: Vec<(String, String)>,
    /// Entries per `top_k` / `inner_top_k` cell: 0 none, u64::MAX every ranked value.
    pub top_k: u64,
}

/// Recommends dtypes for one frame from exact statistics, each candidate verified on
/// every row; all state stays in Rust.
pub struct OneShotRecommender(crate::recommenders::oneshot::OneShot);

impl OneShotRecommender {
    pub fn new(p: OneShotParams) -> Result<Self> {
        validate_common(p.zstd_level, &p.boolean_pairs)?;
        Ok(Self(crate::recommenders::oneshot::OneShot::new(Params {
            seed: p.seed,
            zstd_level: p.zstd_level,
            categorical_threshold: p.categorical_threshold,
            boolean_pairs: p.boolean_pairs,
            top_k: p.top_k,
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

/// Keywords of the streaming recommender (spec 2026-09-29 §7.1).
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
        let params = Params {
            seed: p.seed,
            zstd_level: p.zstd_level,
            categorical_threshold: p.categorical_threshold,
            boolean_pairs: p.boolean_pairs,
            top_k: p.top_k,
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
            out.column_by_name("rec_arrow_type")
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
            .column_by_name("rec_arrow_type")
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
