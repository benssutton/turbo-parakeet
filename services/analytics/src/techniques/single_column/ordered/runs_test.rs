//! One-sample runs test over k categories (spec
//! docs/superpowers/specs/2026-10-13-runs-test-design.md).
//!
//! R = the number of maximal runs of equal consecutive values — the length of an Arrow
//! run-end-encoded array's run_ends. Under random ordering of the same multiset (n rows,
//! category counts nᵢ, S₂ = Σnᵢ², S₃ = Σnᵢ³; null is a category), Barton & David (1957):
//!
//!   E[R]   = 1 + (n² − S₂) / n
//!   Var[R] = [S₂(S₂ + n(n+1)) − 2nS₃ − n³] / (n²(n−1))
//!
//! z = (R − E[R]) / √Var[R]; p = Φ(z), ONE-SIDED LOWER TAIL: P(R ≤ observed), "fewer runs
//! than chance" — the only direction relevant to REE. Normal approximation, no continuity
//! correction. Var = 0 (n = 1, one distinct value, every value distinct) → z = p = NaN.
//!
//! One pass per column, hashing once per RUN, not per row: each 2^16-row chunk is scanned
//! comparing neighbours (null-aware; floats by canonical bits), adding each run's length to
//! its value's count. Chunk partials merge in row order (rayon's `reduce` keeps operand
//! order): runs add, minus one when the left chunk's last value equals the right's first.
//! Keys are compared, not only hashed, so results are exact.

use std::collections::HashMap;
use std::hash::Hash;

use foldhash::fast::RandomState;
use polars::prelude::*;
use polars_arrow::array::Array;
use rayon::prelude::*;

use crate::common::encode::{canon_f32, canon_f64};

const CHUNK: usize = 1 << 16;
const ONE_CHUNK: &str = "rechunked to one chunk";

// ─────────────────────────────────────────────────────────────────────────────
// Runs over row ranges
// ─────────────────────────────────────────────────────────────────────────────

/// Summary of one contiguous range of rows. Keys: `None` = null. `first` / `last` are
/// `None` only for an empty range.
#[derive(Debug, Clone, PartialEq)]
struct Partial<K: Eq + Hash> {
    rows: u64,
    runs: u64,
    first: Option<Option<K>>,
    last: Option<Option<K>>,
    counts: HashMap<Option<K>, u64, RandomState>,
}

impl<K: Copy + Eq + Hash> Partial<K> {
    fn empty() -> Self {
        Self {
            rows: 0,
            runs: 0,
            first: None,
            last: None,
            counts: HashMap::default(),
        }
    }

    /// One pass over `values` (in row order): one hash-map update per run.
    fn scan(mut values: impl Iterator<Item = Option<K>>) -> Self {
        let mut p = Self::empty();
        let Some(mut cur) = values.next() else {
            return p;
        };
        p.first = Some(cur);
        let mut len = 1u64;
        for v in values {
            if v == cur {
                len += 1;
            } else {
                p.close(cur, len);
                cur = v;
                len = 1;
            }
        }
        p.close(cur, len);
        p.last = Some(cur);
        p
    }

    fn close(&mut self, value: Option<K>, len: u64) {
        *self.counts.entry(value).or_insert(0) += len;
        self.runs += 1;
        self.rows += len;
    }

    /// `self` followed by `other` (row order matters; associative). A run spanning the
    /// boundary was counted once on each side, so one is taken off.
    fn merge(mut self, mut other: Self) -> Self {
        if self.rows == 0 {
            return other;
        }
        if other.rows == 0 {
            return self;
        }
        let joined = self.last == other.first;
        if self.counts.len() < other.counts.len() {
            std::mem::swap(&mut self.counts, &mut other.counts);
        }
        self.counts.reserve(other.counts.len());
        for (k, c) in other.counts {
            *self.counts.entry(k).or_insert(0) += c;
        }
        Self {
            rows: self.rows + other.rows,
            runs: self.runs + other.runs - joined as u64,
            first: self.first,
            last: other.last,
            counts: self.counts,
        }
    }

    fn tally(&self) -> Tally {
        let (mut s2, mut s3) = (0u128, 0u128);
        for &c in self.counts.values() {
            let c = c as u128;
            s2 += c * c;
            s3 += c * c * c; // fits u128 for n up to ≈ 7·10¹²
        }
        Tally {
            rows: self.rows,
            runs: self.runs,
            s2,
            s3,
        }
    }
}

/// What the statistics need from a column: n, R, S₂ = Σnᵢ², S₃ = Σnᵢ³.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct Tally {
    rows: u64,
    runs: u64,
    s2: u128,
    s3: u128,
}

/// Rows `0..n` in parallel 2^16-row chunks, merged in row order; `get(i)` is row i's key.
fn runs_of<K, G>(n: usize, get: G) -> Tally
where
    K: Copy + Eq + Hash + Send,
    G: Fn(usize) -> Option<K> + Sync,
{
    (0..n.div_ceil(CHUNK))
        .into_par_iter()
        .map(|c| Partial::scan((c * CHUNK..((c + 1) * CHUNK).min(n)).map(&get)))
        .reduce(Partial::empty, Partial::merge)
        .tally()
}

// ─────────────────────────────────────────────────────────────────────────────
// Statistics
// ─────────────────────────────────────────────────────────────────────────────

/// One column's result row.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct RunsStats {
    pub n_rows: u64,
    pub n_runs: u64,
    pub expected_runs: f64,
    pub z: f64,
    pub p_value: f64,
    pub mean_run_length: f64,
}

/// S₂(S₂ + n(n+1)) − 2nS₃ − n³: exact in i128 — its terms (~n⁴) cancel heavily on
/// near-constant columns, which f64 would lose — and f64 once i128 overflows
/// (n ≳ 3·10⁹).
fn variance_numerator(n: u128, s2: u128, s3: u128) -> f64 {
    let exact = || -> Option<i128> {
        let (n, s2, s3) = (
            i128::try_from(n).ok()?,
            i128::try_from(s2).ok()?,
            i128::try_from(s3).ok()?,
        );
        let a = s2.checked_mul(s2.checked_add(n.checked_mul(n + 1)?)?)?;
        let b = n.checked_mul(2)?.checked_mul(s3)?;
        let c = n.checked_mul(n)?.checked_mul(n)?;
        a.checked_sub(b)?.checked_sub(c)
    };
    exact().map_or_else(
        || {
            let (n, s2, s3) = (n as f64, s2 as f64, s3 as f64);
            s2 * (s2 + n * (n + 1.0)) - 2.0 * n * s3 - n * n * n
        },
        |v| v as f64,
    )
}

/// n(R − 1) − (n² − S₂), the numerator of R − E[R] (= d / n): exact in i128, f64 on overflow.
fn run_excess_numerator(n: u128, runs: u64, s2: u128) -> f64 {
    let exact = || -> Option<i128> {
        let n = i128::try_from(n).ok()?;
        let s2 = i128::try_from(s2).ok()?;
        let r1 = i128::from(runs) - 1;
        n.checked_mul(r1)?
            .checked_sub(n.checked_mul(n)?.checked_sub(s2)?)
    };
    exact().map_or_else(
        || n as f64 * (runs as f64 - 1.0) - ((n as f64) * (n as f64) - s2 as f64),
        |v| v as f64,
    )
}

/// R − E is computed exactly as (n(R−1) − (n² − S₂))/n so a fixed R gives NaN, not ±inf.
/// NaN (z, p) when Var is not positive: n ≤ 1, one distinct value, all distinct.
fn statistics(t: Tally) -> RunsStats {
    let n = t.rows as u128;
    let (nf, rf) = (t.rows as f64, t.runs as f64);
    let expected = 1.0 + (n * n - t.s2) as f64 / nf;
    let var_num = variance_numerator(n, t.s2, t.s3);
    let variance = var_num / ((n * n) as f64 * (nf - 1.0));
    let z = if variance > 0.0 {
        (run_excess_numerator(n, t.runs, t.s2) / nf) / variance.sqrt()
    } else {
        f64::NAN
    };
    let p_value = if z.is_nan() {
        f64::NAN
    } else {
        0.5 * libm::erfc(-z / std::f64::consts::SQRT_2)
    };
    RunsStats {
        n_rows: t.rows,
        n_runs: t.runs,
        expected_runs: expected,
        z,
        p_value,
        mean_run_length: nf / rf,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Dispatch
// ─────────────────────────────────────────────────────────────────────────────

/// Logical dtypes the order-dependent kernels accept (runs test, skip indexes) — mirrored by
/// analytics/_dtypes.py::ORDERED_SCALARS.
pub(crate) fn supported(dtype: &DataType) -> bool {
    matches!(
        dtype,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Boolean
            | DataType::Float32
            | DataType::Float64
            | DataType::Decimal(_, _)
            | DataType::Date
            | DataType::Datetime(_, _)
            | DataType::Duration(_)
            | DataType::Time
            | DataType::String
            | DataType::Binary
            | DataType::Categorical(_, _)
            | DataType::Enum(_, _)
    )
}

/// Integer keys (temporal values, Decimal's unscaled i128, Categorical / Enum codes).
fn numeric<T>(ca: &ChunkedArray<T>) -> Tally
where
    T: PolarsNumericType,
    T::Native: Eq + Hash,
{
    let arr = ca.downcast_iter().next().expect(ONE_CHUNK);
    runs_of(arr.len(), |i| arr.is_valid(i).then(|| arr.value(i)))
}

/// The run statistics of one column, in its row order. Keys by physical type:
/// Categorical / Enum compare codes (bijective with their strings within a column),
/// Decimal its unscaled i128, temporal types their integers, floats canonical bits.
pub(crate) fn series_runs(s: &Series) -> PolarsResult<RunsStats> {
    if !supported(s.dtype()) {
        polars_bail!(
            InvalidOperation: "runs_test: unsupported dtype {} (column {})", s.dtype(), s.name()
        );
    }
    if s.is_empty() {
        return Ok(statistics(Tally::default()));
    }
    // One chunk, so row i is index i of a single Arrow array (zero-copy if already one).
    let phys = s.to_physical_repr().rechunk();
    let tally = match phys.dtype() {
        DataType::Int8 => numeric(phys.i8()?),
        DataType::Int16 => numeric(phys.i16()?),
        DataType::Int32 => numeric(phys.i32()?),
        DataType::Int64 => numeric(phys.i64()?),
        DataType::UInt8 => numeric(phys.u8()?),
        DataType::UInt16 => numeric(phys.u16()?),
        DataType::UInt32 => numeric(phys.u32()?),
        DataType::UInt64 => numeric(phys.u64()?),
        DataType::Int128 => numeric(phys.i128()?),
        DataType::Float32 => {
            let arr = phys.f32()?.downcast_iter().next().expect(ONE_CHUNK);
            runs_of(arr.len(), |i| {
                arr.is_valid(i).then(|| canon_f32(arr.value(i)))
            })
        }
        DataType::Float64 => {
            let arr = phys.f64()?.downcast_iter().next().expect(ONE_CHUNK);
            runs_of(arr.len(), |i| {
                arr.is_valid(i).then(|| canon_f64(arr.value(i)))
            })
        }
        DataType::Boolean => {
            let arr = phys.bool()?.downcast_iter().next().expect(ONE_CHUNK);
            runs_of(arr.len(), |i| arr.is_valid(i).then(|| arr.value(i)))
        }
        DataType::String => {
            let arr = phys.str()?.downcast_iter().next().expect(ONE_CHUNK);
            runs_of(arr.len(), |i| arr.is_valid(i).then(|| arr.value(i)))
        }
        DataType::Binary => {
            let arr = phys.binary()?.downcast_iter().next().expect(ONE_CHUNK);
            runs_of(arr.len(), |i| arr.is_valid(i).then(|| arr.value(i)))
        }
        dt => polars_bail!(ComputeError: "runs_test: unexpected physical dtype {dt}"),
    };
    Ok(statistics(tally))
}

// ─────────────────────────────────────────────────────────────────────────────
// Implementation
// ─────────────────────────────────────────────────────────────────────────────

/// `column, n_rows, n_runs, expected_runs, z, p_value, mean_run_length` per column;
/// columns in parallel (and each column's chunks).
pub(crate) fn runs_test_impl(inputs: &[Series]) -> PolarsResult<Series> {
    let stats: Vec<RunsStats> = inputs
        .par_iter()
        .map(series_runs)
        .collect::<PolarsResult<_>>()?;
    let column = StringChunked::from_iter(inputs.iter().map(|s| s.name().as_str()))
        .into_series()
        .with_name("column".into());
    let u64s = |name: &str, f: fn(&RunsStats) -> u64| {
        UInt64Chunked::from_vec(name.into(), stats.iter().map(f).collect()).into_series()
    };
    let f64s = |name: &str, f: fn(&RunsStats) -> f64| {
        Float64Chunked::from_vec(name.into(), stats.iter().map(f).collect()).into_series()
    };
    let fields = [
        column,
        u64s("n_rows", |s| s.n_rows),
        u64s("n_runs", |s| s.n_runs),
        f64s("expected_runs", |s| s.expected_runs),
        f64s("z", |s| s.z),
        f64s("p_value", |s| s.p_value),
        f64s("mean_run_length", |s| s.mean_run_length),
    ];
    Ok(StructChunked::from_series("runs_test".into(), inputs.len(), fields.iter())?.into_series())
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn scan(v: &[Option<i64>]) -> Partial<i64> {
        Partial::scan(v.iter().copied())
    }

    const SAMPLE: [Option<i64>; 10] = [
        Some(1),
        Some(1),
        None,
        None,
        Some(2),
        Some(1),
        Some(1),
        Some(1),
        None,
        Some(2),
    ];

    #[test]
    fn scan_counts_runs_and_values() {
        let p = scan(&SAMPLE);
        assert_eq!((p.rows, p.runs), (10, 6));
        assert_eq!(p.counts[&Some(1)], 5);
        assert_eq!(p.counts[&None], 3);
        assert_eq!(p.counts[&Some(2)], 2);
        assert_eq!((p.first, p.last), (Some(Some(1)), Some(Some(2))));
        assert_eq!(scan(&[]), Partial::empty());
    }

    #[test]
    fn merge_equals_one_scan_at_every_split() {
        let whole = scan(&SAMPLE);
        for i in 0..=SAMPLE.len() {
            assert_eq!(
                scan(&SAMPLE[..i]).merge(scan(&SAMPLE[i..])),
                whole,
                "split {i}"
            );
            for j in i..=SAMPLE.len() {
                let (a, b, c) = (scan(&SAMPLE[..i]), scan(&SAMPLE[i..j]), scan(&SAMPLE[j..]));
                let left = a.clone().merge(b.clone()).merge(c.clone());
                assert_eq!(left, a.merge(b.merge(c)), "splits {i}, {j}");
                assert_eq!(left, whole, "splits {i}, {j}");
            }
        }
    }

    #[test]
    fn runs_of_crosses_chunk_boundaries() {
        let n = 3 * CHUNK + 17; // runs of 1 000 rows straddle every chunk boundary
        let t = runs_of(n, |i| Some((i / 1000) as i64));
        assert_eq!((t.rows, t.runs), (n as u64, n.div_ceil(1000) as u64));

        // runs end exactly on chunk boundaries (never joined across them)
        let t = runs_of(n, |i| Some((i / CHUNK) as i64));
        assert_eq!((t.rows, t.runs), (n as u64, n.div_ceil(CHUNK) as u64));
        assert_eq!(t.s2, 3 * (CHUNK as u128).pow(2) + 17u128.pow(2));
        assert_eq!(t.s3, 3 * (CHUNK as u128).pow(3) + 17u128.pow(3));
    }

    #[test]
    fn statistics_all_distinct_large_is_nan_not_inf() {
        let n = 231_383_004u64;
        let r = statistics(Tally {
            rows: n,
            runs: n,
            s2: n as u128,
            s3: n as u128,
        });
        assert!(r.z.is_nan() && r.p_value.is_nan());
        assert!((r.expected_runs - n as f64).abs() < 1.0);
    }

    #[test]
    fn statistics_known_answer() {
        // A A B B B C: n = 6, counts 2, 3, 1 → S₂ = 14, S₃ = 36, numerator 136
        let r = statistics(Tally {
            rows: 6,
            runs: 3,
            s2: 14,
            s3: 36,
        });
        assert!((r.expected_runs - 14.0 / 3.0).abs() < 1e-12);
        let z = (3.0 - 14.0 / 3.0) / (136.0f64 / 180.0).sqrt();
        assert!((r.z - z).abs() < 1e-12);
        // Φ(z), computed independently in Python: 0.5 * math.erfc(-z / math.sqrt(2))
        assert!((r.p_value - 0.02759277549556695).abs() < 1e-15);
        assert_eq!(r.mean_run_length, 2.0);
    }

    #[test]
    fn statistics_undefined_variance_is_nan() {
        let one = statistics(Tally {
            rows: 1,
            runs: 1,
            s2: 1,
            s3: 1,
        });
        assert_eq!((one.expected_runs, one.mean_run_length), (1.0, 1.0));
        assert!(one.z.is_nan() && one.p_value.is_nan());
        let constant = statistics(Tally {
            rows: 4,
            runs: 1,
            s2: 16,
            s3: 64,
        });
        assert_eq!(constant.expected_runs, 1.0);
        assert!(constant.z.is_nan() && constant.p_value.is_nan());
        let distinct = statistics(Tally {
            rows: 4,
            runs: 4,
            s2: 4,
            s3: 4,
        });
        assert_eq!(distinct.expected_runs, 4.0);
        assert!(distinct.z.is_nan() && distinct.p_value.is_nan());
        let empty = statistics(Tally::default());
        assert!(empty.expected_runs.is_nan() && empty.mean_run_length.is_nan());
    }

    #[test]
    fn variance_numerator_is_exact_when_it_fits() {
        assert_eq!(variance_numerator(6, 14, 36), 136.0);
        // near-constant: n = 10⁶, counts [n − 10, 10]; k = 2 closed form 2p(2p − n), p = 10(n − 10)
        let (n, a, b) = (1_000_000i128, 999_990i128, 10i128);
        let p = b * a;
        let want = (2 * p * (2 * p - n)) as f64;
        let got = variance_numerator(
            n as u128,
            (a * a + b * b) as u128,
            (a * a * a + b * b * b) as u128,
        );
        assert_eq!(got, want);
    }

    #[test]
    fn variance_numerator_overflow_falls_back_to_f64() {
        // n = 4·10⁹ split evenly in two overflows i128; k = 2 closed form: 2p(2p − n), p = n₁n₂.
        let (n, half) = (4_000_000_000u128, 2_000_000_000u128);
        let (s2, s3) = (2 * half * half, 2 * half * half * half);
        let p = (half * half) as f64;
        let want = 2.0 * p * (2.0 * p - n as f64);
        let got = variance_numerator(n, s2, s3);
        assert!(((got - want) / want).abs() < 1e-12, "{got} vs {want}");
    }

    fn stats_of(s: Series) -> RunsStats {
        series_runs(&s).unwrap()
    }

    #[test]
    fn strings_known_answer() {
        let r = stats_of(Series::new("x".into(), &["A", "A", "B", "B", "B", "C"]));
        assert_eq!((r.n_rows, r.n_runs), (6, 3));
        assert!((r.expected_runs - 14.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn alternating_gives_positive_z_and_upper_p() {
        let r = stats_of(Series::new("x".into(), &[0i64, 1, 0, 1, 0, 1]));
        assert_eq!((r.n_runs, r.expected_runs), (6, 4.0));
        assert!((r.z - 2.0 / 1.2f64.sqrt()).abs() < 1e-12);
        assert!(r.p_value > 0.5);
    }

    #[test]
    fn nulls_form_one_run() {
        let r = stats_of(Series::new("x".into(), &[Some(1i64), None, None, Some(1)]));
        assert_eq!(r.n_runs, 3);
        let all_null = stats_of(Series::new("x".into(), &[None::<i64>, None, None]));
        assert_eq!((all_null.n_runs, all_null.expected_runs), (1, 1.0));
        assert!(all_null.z.is_nan());
    }

    #[test]
    fn floats_compare_canonically() {
        let other_nan = f64::from_bits(0x7ff8_0000_0000_0001);
        let r = stats_of(Series::new(
            "x".into(),
            &[0.0f64, -0.0, f64::NAN, other_nan],
        ));
        assert_eq!(r.n_runs, 2);
        let r32 = stats_of(Series::new("x".into(), &[0.0f32, -0.0, f32::NAN, f32::NAN]));
        assert_eq!(r32.n_runs, 2);
    }

    #[test]
    fn every_physical_family() {
        let ints = Series::new("x".into(), &[0i64, 0, 1, 1, 1, 2]);
        for dt in [
            DataType::Int8,
            DataType::UInt16,
            DataType::Date,
            DataType::Datetime(TimeUnit::Microseconds, None),
            DataType::Duration(TimeUnit::Milliseconds),
            DataType::Time,
            DataType::Decimal(Some(10), Some(2)),
            DataType::String,
            DataType::Binary,
        ] {
            let s = if dt == DataType::Binary {
                ints.cast(&DataType::String).unwrap().cast(&dt).unwrap()
            } else {
                ints.cast(&dt).unwrap()
            };
            assert_eq!(stats_of(s).n_runs, 3, "{dt}");
        }
        let b = Series::new("x".into(), &[true, true, false, false, false, true]);
        assert_eq!(stats_of(b).n_runs, 3);
    }

    #[test]
    fn multi_chunk_input_matches_one_chunk() {
        let mut s = Series::new("x".into(), &[1i64, 1]);
        s.append(&Series::new("x".into(), &[1i64, 2])).unwrap();
        s.append(&Series::new("x".into(), &[2i64])).unwrap();
        assert!(s.n_chunks() > 1);
        assert_eq!(stats_of(s).n_runs, 2);
    }

    #[test]
    fn empty_column_is_nan() {
        let r = stats_of(Series::new_empty("x".into(), &DataType::Int64));
        assert_eq!((r.n_rows, r.n_runs), (0, 0));
        assert!(r.expected_runs.is_nan() && r.mean_run_length.is_nan());
    }

    #[test]
    fn unsupported_dtype_is_invalid_operation() {
        let s = Series::new("x".into(), &[1i128, 2]);
        assert!(matches!(
            series_runs(&s),
            Err(PolarsError::InvalidOperation(_))
        ));
    }

    #[test]
    fn impl_returns_one_struct_row_per_column() {
        let out = runs_test_impl(&[
            Series::new("a".into(), &[1i64, 1, 2]),
            Series::new("b".into(), &["x", "y", "y"]),
        ])
        .unwrap();
        assert_eq!(out.len(), 2);
        let fields = out.struct_().unwrap().fields_as_series();
        let names: Vec<_> = fields.iter().map(|f| f.name().to_string()).collect();
        assert_eq!(
            names,
            [
                "column",
                "n_rows",
                "n_runs",
                "expected_runs",
                "z",
                "p_value",
                "mean_run_length"
            ]
        );
        assert_eq!(fields[2].u64().unwrap().get(1), Some(2));
    }
}
