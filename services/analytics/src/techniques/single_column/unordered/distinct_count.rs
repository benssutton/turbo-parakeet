//! HyperLogLog distinct count of every column (the `distinct_count` technique's Rust
//! implementation, spec 2026-10-24-cardinality-entropy-techniques §2): one sketch per
//! 64K-row chunk, merged by register max, columns and chunks in parallel.

use polars::prelude::*;
use rayon::prelude::*;

use crate::common::encode_series;
use crate::techniques::hll::{hash_key, Hll};

const CHUNK: usize = 1 << 16;

/// `col_name`, `n_non_null`, `estimate`, `std_error`, `sketch_bytes` per column.
pub(crate) fn hll_distinct_impl(inputs: &[Series], p: u8) -> PolarsResult<Series> {
    let rows = inputs
        .par_iter()
        .map(|s| {
            let col = encode_series(s)?;
            let sketch = col
                .values
                .par_chunks(CHUNK)
                .zip(col.is_null.par_chunks(CHUNK))
                .map(|(values, nulls)| {
                    let mut h = Hll::new(p);
                    for (&v, &null) in values.iter().zip(nulls) {
                        if !null {
                            h.insert(hash_key(v));
                        }
                    }
                    h
                })
                .reduce(
                    || Hll::new(p),
                    |mut a, b| {
                        a.merge(&b);
                        a
                    },
                );
            let n_non_null = col.is_null.iter().filter(|&&null| !null).count() as u64;
            Ok((n_non_null, sketch.estimate(), sketch.std_error()))
        })
        .collect::<PolarsResult<Vec<_>>>()?;
    let n = inputs.len();
    let names = StringChunked::from_iter(inputs.iter().map(|s| s.name().as_str()))
        .into_series()
        .with_name("col_name".into());
    let series = [
        names,
        UInt64Chunked::from_vec("n_non_null".into(), rows.iter().map(|r| r.0).collect())
            .into_series(),
        Float64Chunked::from_vec("estimate".into(), rows.iter().map(|r| r.1).collect())
            .into_series(),
        Float64Chunked::from_vec("std_error".into(), rows.iter().map(|r| r.2).collect())
            .into_series(),
        UInt64Chunked::from_vec("sketch_bytes".into(), vec![1u64 << p; n]).into_series(),
    ];
    Ok(StructChunked::from_series("distinct_count".into(), n, series.iter())?.into_series())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(s: Series, p: u8) -> (u64, f64, f64, u64) {
        let df = hll_distinct_impl(&[s], p)
            .unwrap()
            .into_frame()
            .unnest(["distinct_count"])
            .unwrap();
        let get_u = |c: &str| df.column(c).unwrap().u64().unwrap().get(0).unwrap();
        let get_f = |c: &str| df.column(c).unwrap().f64().unwrap().get(0).unwrap();
        (
            get_u("n_non_null"),
            get_f("estimate"),
            get_f("std_error"),
            get_u("sketch_bytes"),
        )
    }

    #[test]
    fn small_column_is_near_exact_and_skips_nulls() {
        let values: Vec<Option<i64>> = (0..1_000).map(|i| (i % 7 != 0).then_some(i % 50)).collect();
        let (n, est, se, bytes) = row(Series::new("a".into(), values), 14);
        assert_eq!(n, 1_000 - 143);
        // p = 14: at most a register collision or two (~0.1 expected) at 50 values.
        assert!((est - 50.0).abs() <= 1.5, "estimate {est}");
        assert!((se - 1.04 / 128.0).abs() < 1e-12);
        assert_eq!(bytes, 1 << 14);
    }

    #[test]
    fn chunks_merge_to_the_single_sketch() {
        // 3 chunks of 2^16 rows; the merged estimate equals one sketch over every value.
        let v: Vec<i64> = (0..(3 * CHUNK as i64)).map(|i| i % 100_000).collect();
        let (_, est, _, _) = row(Series::new("a".into(), v.clone()), 12);
        let mut one = Hll::new(12);
        for x in v {
            one.insert(hash_key(x as u64));
        }
        assert_eq!(est, one.estimate());
    }

    #[test]
    fn canonical_floats_are_one_value() {
        let (_, est, _, _) = row(
            Series::new("f".into(), [0.0f64, -0.0, f64::NAN, f64::NAN]),
            14,
        );
        assert!((est - 2.0).abs() <= 0.01, "estimate {est}");
    }
}
