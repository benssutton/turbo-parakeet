//! Consecutive differences of an integer-backed column, in row order (spec
//! docs/superpowers/specs/2026-10-23-delta-codecs-design.md §2): ranges, mean significant
//! bytes, zero shares and entropies of the deltas and double-deltas, and the mean significant
//! bytes of deltas under a random row order. Nulls are skipped: deltas are taken between
//! consecutive non-null values.

use std::collections::HashMap;

use foldhash::fast::RandomState;
use polars::prelude::*;
use rayon::prelude::*;

use crate::common::entropy_math::entropy_from_counts_iter;

/// Bytes of the smallest signed integer holding `v`; 0 for 0 (spec §2).
pub(crate) fn sig(v: i128) -> u32 {
    if v == 0 {
        return 0;
    }
    let m = if v >= 0 { v } else { !v };
    let bits = 128 - m.leading_zeros(); // magnitude bits; + 1 sign bit
    (bits + 8) / 8
}

const CHUNK: usize = 1 << 16;

/// Byte-level statistics of one column's non-null values in row order.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct DeltaStats {
    pub n_values: u64,
    pub n_deltas: u64,
    pub n_double_deltas: u64,
    pub delta_min: Option<i128>,
    pub delta_max: Option<i128>,
    pub double_delta_min: Option<i128>,
    pub double_delta_max: Option<i128>,
    pub value_bytes: u64,        // Σ sig(value)
    pub delta_bytes: u64,        // Σ sig(delta)
    pub double_delta_bytes: u64, // Σ sig(double delta)
    pub zero_deltas: u64,
    pub zero_double_deltas: u64,
}

impl DeltaStats {
    pub(crate) fn value_bytes_mean(&self) -> f64 {
        mean(self.value_bytes, self.n_values)
    }
    pub(crate) fn delta_bytes_mean(&self) -> f64 {
        mean(self.delta_bytes, self.n_deltas)
    }
    pub(crate) fn double_delta_bytes_mean(&self) -> f64 {
        mean(self.double_delta_bytes, self.n_double_deltas)
    }
}

/// `sum / n`, NaN when `n` is 0.
fn mean(sum: u64, n: u64) -> f64 {
    if n == 0 {
        f64::NAN
    } else {
        sum as f64 / n as f64
    }
}

fn opt_min(a: Option<i128>, b: Option<i128>) -> Option<i128> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x.min(y)),
        (x, None) => x,
        (None, y) => y,
    }
}

fn opt_max(a: Option<i128>, b: Option<i128>) -> Option<i128> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x.max(y)),
        (x, None) => x,
        (None, y) => y,
    }
}

/// Statistics of the rows `start..end` of `values`: the delta owned by row `i` is
/// `values[i + 1] − values[i]`, the double-delta `d[i + 1] − d[i]`; reads past `end`.
fn scan_range(values: &[i128], start: usize, end: usize) -> DeltaStats {
    let n = values.len();
    let mut s = DeltaStats {
        n_values: (end - start) as u64,
        ..Default::default()
    };
    for i in start..end {
        s.value_bytes += sig(values[i]) as u64;
        if i + 1 < n {
            let d = values[i + 1] - values[i];
            s.n_deltas += 1;
            s.delta_bytes += sig(d) as u64;
            s.zero_deltas += (d == 0) as u64;
            s.delta_min = opt_min(s.delta_min, Some(d));
            s.delta_max = opt_max(s.delta_max, Some(d));
            if i + 2 < n {
                let dd = (values[i + 2] - values[i + 1]) - d;
                s.n_double_deltas += 1;
                s.double_delta_bytes += sig(dd) as u64;
                s.zero_double_deltas += (dd == 0) as u64;
                s.double_delta_min = opt_min(s.double_delta_min, Some(dd));
                s.double_delta_max = opt_max(s.double_delta_max, Some(dd));
            }
        }
    }
    s
}

/// Byte-level statistics (no entropies, no baseline): the recommender's pruning input.
/// Deltas are taken within `values` only (a caller with several blocks merges with `merge`).
pub(crate) fn scan(values: &[i128]) -> DeltaStats {
    let n = values.len();
    (0..n.div_ceil(CHUNK))
        .into_par_iter()
        .map(|c| scan_range(values, c * CHUNK, ((c + 1) * CHUNK).min(n)))
        .reduce(DeltaStats::default, merge)
}

/// Sum of two scans of disjoint blocks (no delta across the blocks).
pub(crate) fn merge(a: DeltaStats, b: DeltaStats) -> DeltaStats {
    DeltaStats {
        n_values: a.n_values + b.n_values,
        n_deltas: a.n_deltas + b.n_deltas,
        n_double_deltas: a.n_double_deltas + b.n_double_deltas,
        delta_min: opt_min(a.delta_min, b.delta_min),
        delta_max: opt_max(a.delta_max, b.delta_max),
        double_delta_min: opt_min(a.double_delta_min, b.double_delta_min),
        double_delta_max: opt_max(a.double_delta_max, b.double_delta_max),
        value_bytes: a.value_bytes + b.value_bytes,
        delta_bytes: a.delta_bytes + b.delta_bytes,
        double_delta_bytes: a.double_delta_bytes + b.double_delta_bytes,
        zero_deltas: a.zero_deltas + b.zero_deltas,
        zero_double_deltas: a.zero_double_deltas + b.zero_double_deltas,
    }
}

/// Shannon entropy (bits) of a multiset of i128 keys; NaN when empty.
fn entropy(keys: impl ParallelIterator<Item = i128>) -> f64 {
    let counts: HashMap<i128, u64, RandomState> = keys
        .fold(HashMap::default, |mut m, k| {
            *m.entry(k).or_insert(0) += 1;
            m
        })
        .reduce(HashMap::default, |mut a, b| {
            for (k, c) in b {
                *a.entry(k).or_insert(0) += c;
            }
            a
        });
    let r: u64 = counts.values().sum();
    if r == 0 {
        return f64::NAN;
    }
    if counts.len() == 1 {
        return 0.0; // exactly: the SIMD reduction leaves ~1e-15 of rounding
    }
    entropy_from_counts_iter(counts.into_values(), (r as f64).log2(), r as f64).max(0.0)
}

/// Mean sig(Δ) of Δ = x_j − x_i over ordered pairs of distinct positions (spec §2):
/// Σ_{k=1..9} P(sig ≥ k); sig ≥ 1 iff Δ ≠ 0, sig ≥ k (k ≥ 2) iff Δ ∉ [−B, B), B = 2^(8(k−1)−1).
pub(crate) fn expected_delta_bytes(values: &[i128]) -> f64 {
    let n = values.len();
    if n < 2 {
        return f64::NAN;
    }
    let mut v = values.to_vec();
    v.par_sort_unstable();
    let pairs = (n as f64) * (n as f64 - 1.0);
    let mut equal: u128 = 0;
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && v[j] == v[i] {
            j += 1;
        }
        let c = (j - i) as u128;
        equal += c * (c - 1);
        i = j;
    }
    let tail: Vec<f64> = (2..=9u32)
        .into_par_iter()
        .map(|k| {
            let b: i128 = 1i128 << (8 * (k - 1) - 1);
            // inside = #ordered (i, j), i ≠ j, with v[i] − b ≤ v[j] < v[i] + b
            let (mut lo, mut hi, mut inside) = (0usize, 0usize, 0u128);
            for &x in &v {
                while lo < n && v[lo] < x - b {
                    lo += 1;
                }
                while hi < n && v[hi] < x + b {
                    hi += 1;
                }
                inside += (hi - lo - 1) as u128;
            }
            1.0 - inside as f64 / pairs
        })
        .collect();
    1.0 - equal as f64 / pairs + tail.iter().sum::<f64>()
}

/// The non-null physical values as i128; `None` for dtypes the technique does not cover
/// (same set as `gcd.rs`: integer-backed only).
fn physical_values(s: &Series) -> PolarsResult<Option<Vec<i128>>> {
    if !matches!(
        s.dtype(),
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Decimal(_, _)
            | DataType::Date
            | DataType::Datetime(_, _)
            | DataType::Duration(_)
            | DataType::Time
    ) {
        return Ok(None);
    }
    let phys = s.to_physical_repr();
    macro_rules! collect {
        ($ca:expr) => {
            $ca.into_iter().flatten().map(|v| v as i128).collect()
        };
    }
    let v: Vec<i128> = match phys.dtype() {
        DataType::Int8 => collect!(phys.i8()?),
        DataType::Int16 => collect!(phys.i16()?),
        DataType::Int32 => collect!(phys.i32()?),
        DataType::Int64 => collect!(phys.i64()?),
        DataType::UInt8 => collect!(phys.u8()?),
        DataType::UInt16 => collect!(phys.u16()?),
        DataType::UInt32 => collect!(phys.u32()?),
        DataType::UInt64 => collect!(phys.u64()?),
        DataType::Int128 => collect!(phys.i128()?),
        dt => {
            return Err(PolarsError::ComputeError(
                format!("delta: unexpected physical dtype {dt}").into(),
            ))
        }
    };
    Ok(Some(v))
}

struct ColumnDelta {
    n_rows: u64,
    stats: DeltaStats,
    delta_entropy: f64,
    double_delta_entropy: f64,
    expected: f64,
}

fn column_delta(s: &Series) -> PolarsResult<ColumnDelta> {
    let Some(v) = physical_values(s)? else {
        return Err(PolarsError::InvalidOperation(
            format!(
                "delta: column '{}' has ineligible dtype {}",
                s.name(),
                s.dtype()
            )
            .into(),
        ));
    };
    let n = v.len();
    let stats = scan(&v);
    let delta_entropy = entropy(
        (0..n.saturating_sub(1))
            .into_par_iter()
            .map(|i| v[i + 1] - v[i]),
    );
    let double_delta_entropy = entropy(
        (0..n.saturating_sub(2))
            .into_par_iter()
            .map(|i| (v[i + 2] - v[i + 1]) - (v[i + 1] - v[i])),
    );
    Ok(ColumnDelta {
        n_rows: s.len() as u64,
        stats,
        delta_entropy,
        double_delta_entropy,
        expected: expected_delta_bytes(&v),
    })
}

/// `column, n_rows, n_non_null, n_deltas, delta_min, delta_max, double_delta_min,
/// double_delta_max, value_bytes_mean, delta_bytes_mean, double_delta_bytes_mean,
/// zero_delta_fraction, zero_double_delta_fraction, delta_entropy, double_delta_entropy,
/// expected_delta_bytes_mean` per column (columns in parallel). Ineligible columns: an
/// InvalidOperation error naming the column (the Python base never sends them).
pub(crate) fn delta_impl(inputs: &[Series]) -> PolarsResult<Series> {
    let cols: Vec<ColumnDelta> = inputs
        .par_iter()
        .map(column_delta)
        .collect::<PolarsResult<_>>()?;
    let column = StringChunked::from_iter(inputs.iter().map(|s| s.name().as_str()))
        .into_series()
        .with_name("column".into());
    let u64s = |name: &str, f: fn(&ColumnDelta) -> u64| {
        UInt64Chunked::from_vec(name.into(), cols.iter().map(f).collect()).into_series()
    };
    let f64s = |name: &str, f: fn(&ColumnDelta) -> f64| {
        Float64Chunked::from_vec(name.into(), cols.iter().map(f).collect()).into_series()
    };
    let dec = |name: &str, f: fn(&ColumnDelta) -> Option<i128>| {
        Int128Chunked::from_iter_options(name.into(), cols.iter().map(f))
            .into_decimal_unchecked(Some(38), 0)
            .into_series()
    };
    let fields = [
        column,
        u64s("n_rows", |c| c.n_rows),
        u64s("n_non_null", |c| c.stats.n_values),
        u64s("n_deltas", |c| c.stats.n_deltas),
        dec("delta_min", |c| c.stats.delta_min),
        dec("delta_max", |c| c.stats.delta_max),
        dec("double_delta_min", |c| c.stats.double_delta_min),
        dec("double_delta_max", |c| c.stats.double_delta_max),
        f64s("value_bytes_mean", |c| c.stats.value_bytes_mean()),
        f64s("delta_bytes_mean", |c| c.stats.delta_bytes_mean()),
        f64s("double_delta_bytes_mean", |c| {
            c.stats.double_delta_bytes_mean()
        }),
        f64s("zero_delta_fraction", |c| {
            mean(c.stats.zero_deltas, c.stats.n_deltas)
        }),
        f64s("zero_double_delta_fraction", |c| {
            mean(c.stats.zero_double_deltas, c.stats.n_double_deltas)
        }),
        f64s("delta_entropy", |c| c.delta_entropy),
        f64s("double_delta_entropy", |c| c.double_delta_entropy),
        f64s("expected_delta_bytes_mean", |c| c.expected),
    ];
    Ok(StructChunked::from_series("delta".into(), inputs.len(), fields.iter())?.into_series())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 33
        }
    }

    #[test]
    fn sig_values() {
        let cases: [(i128, u32); 10] = [
            (0, 0),
            (1, 1),
            (-1, 1),
            (127, 1),
            (128, 2),
            (-128, 1),
            (-129, 2),
            (i64::MAX as i128, 8),
            (u64::MAX as i128, 9),
            (i64::MIN as i128, 8),
        ];
        for (v, k) in cases {
            assert_eq!(sig(v), k, "sig({v})");
        }
    }

    #[test]
    fn regular_series() {
        let s = scan(&[10, 20, 30, 40]);
        assert_eq!(s.n_values, 4);
        assert_eq!(s.n_deltas, 3);
        assert_eq!((s.delta_min, s.delta_max), (Some(10), Some(10)));
        assert_eq!(s.n_double_deltas, 2);
        assert_eq!(s.zero_double_deltas, 2);
        assert_eq!(s.double_delta_bytes, 0);
        assert_eq!(s.value_bytes, 4);
    }

    #[test]
    fn short_inputs() {
        for v in [&[][..], &[5][..]] {
            let s = scan(v);
            assert_eq!((s.n_deltas, s.n_double_deltas), (0, 0));
            assert_eq!(s.delta_min, None);
            assert_eq!(s.double_delta_max, None);
            assert!(s.delta_bytes_mean().is_nan());
            assert!(s.double_delta_bytes_mean().is_nan());
        }
        assert_eq!(scan(&[5]).value_bytes_mean(), 1.0);
    }

    #[test]
    fn chunk_boundaries_match_sequential() {
        let mut rng = Lcg(7);
        let mut x = 0i128;
        let v: Vec<i128> = (0..200_000)
            .map(|_| {
                x += rng.next() as i128 % 2001 - 1000;
                x
            })
            .collect();
        let got = scan(&v);
        assert_eq!(got, scan_range(&v, 0, v.len()));
        let d: Vec<i128> = v.windows(2).map(|w| w[1] - w[0]).collect();
        let dd: Vec<i128> = d.windows(2).map(|w| w[1] - w[0]).collect();
        assert_eq!(got.n_deltas, d.len() as u64);
        assert_eq!(got.n_double_deltas, dd.len() as u64);
        assert_eq!(
            got.delta_bytes,
            d.iter().map(|&x| sig(x) as u64).sum::<u64>()
        );
        assert_eq!(
            got.double_delta_bytes,
            dd.iter().map(|&x| sig(x) as u64).sum::<u64>()
        );
        assert_eq!(got.delta_min, d.iter().copied().min());
        assert_eq!(got.double_delta_max, dd.iter().copied().max());
        assert_eq!(
            got.zero_deltas,
            d.iter().filter(|&&x| x == 0).count() as u64
        );
    }

    #[test]
    fn merge_is_fieldwise_sum() {
        let (sa, sb) = (scan(&[1, 5, 9, 200]), scan(&[-3, 0, 70_000]));
        let m = merge(sa.clone(), sb.clone());
        assert_eq!(m.n_values, 7);
        assert_eq!(m.n_deltas, sa.n_deltas + sb.n_deltas);
        assert_eq!(m.delta_bytes, sa.delta_bytes + sb.delta_bytes);
        assert_eq!(m.delta_min, opt_min(sa.delta_min, sb.delta_min));
        assert_eq!(
            m.double_delta_max,
            opt_max(sa.double_delta_max, sb.double_delta_max)
        );
    }

    fn brute(v: &[i128]) -> f64 {
        let n = v.len();
        let mut total = 0u64;
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    total += sig(v[j] - v[i]) as u64;
                }
            }
        }
        total as f64 / (n * (n - 1)) as f64
    }

    #[test]
    fn expected_matches_brute_force() {
        let mut rng = Lcg(42);
        for t in 0..50 {
            let n = 2 + (rng.next() % 39) as usize;
            let range = if t % 2 == 0 { 300 } else { 70_000 };
            let v: Vec<i128> = (0..n)
                .map(|_| (rng.next() % (2 * range + 1)) as i128 - range as i128)
                .collect();
            let (got, want) = (expected_delta_bytes(&v), brute(&v));
            assert!((got - want).abs() < 1e-12, "{v:?}: {got} vs {want}");
        }
        let v = [i64::MIN as i128, i64::MAX as i128, 0];
        assert!((expected_delta_bytes(&v) - brute(&v)).abs() < 1e-12);
        assert!(expected_delta_bytes(&[1]).is_nan());
    }

    #[test]
    fn delta_entropy_known() {
        let v = [0i128, 1, 2, 4];
        let h = entropy((0..3).into_par_iter().map(|i| v[i + 1] - v[i]));
        assert!((h - 0.9182958340544896).abs() < 1e-12);
        assert!(entropy((0..0).into_par_iter().map(|_| 0i128)).is_nan());
    }

    #[test]
    fn series_entry_with_nulls() {
        let s = Series::new("t".into(), [Some(10i64), None, Some(20), Some(31)]);
        let out = delta_impl(&[s]).unwrap();
        let st = out.struct_().unwrap();
        let f = |n: &str| st.field_by_name(n).unwrap();
        assert_eq!(f("n_rows").u64().unwrap().get(0), Some(4));
        assert_eq!(f("n_non_null").u64().unwrap().get(0), Some(3));
        assert_eq!(f("n_deltas").u64().unwrap().get(0), Some(2));
        let bad = Series::new("x".into(), [1.5f64, 2.5]);
        assert!(delta_impl(&[bad]).is_err());
    }
}
