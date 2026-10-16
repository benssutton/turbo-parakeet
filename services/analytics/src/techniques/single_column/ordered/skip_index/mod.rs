//! Skip-index techniques (spec docs/superpowers/specs/2026-10-16-skip-index-techniques-design.md):
//! how well a column, in its row order, suits ClickHouse's `minmax` and `set(max_size)`
//! data-skipping indexes.
//!
//! A window of w rows (one index block: `index_granularity × GRANULARITY`) slides one row at a
//! time, so every granule alignment counts once. For a point query `col = v`, a window's skip is
//! the share of query values it lets a query skip — v drawn by row frequency (`rows`, over the N′
//! non-null rows) or uniformly over the d distinct values (`values`). Each skip is an integer
//! numerator over N′ or d, summed exactly in `WindowSums`, so sums over any split of the windows
//! merge exactly. Closed-form baselines give the same expectations under a uniformly random row
//! order (`avoid`).
//!
//! Results = merge(scan(population, segment) for each segment) + baseline(population, w).
//! One-shot passes the whole column as one segment; the streaming recommender will pass its
//! reservoir's blocks.

pub(crate) mod minmax;
pub(crate) mod set;

use std::collections::HashMap;
use std::hash::Hash;
use std::ops::Range;

use foldhash::fast::RandomState;
use polars::prelude::*;
use polars_arrow::array::Array;
use rayon::prelude::*;

use crate::common::encode::{canon_f32, canon_f64};
use crate::techniques::runs_test::supported;

/// The id of a null row.
pub(crate) const NULL: u32 = u32::MAX;
/// Fewest window starts per parallel range.
const RANGE: usize = 1 << 16;
const ONE_CHUNK: &str = "rechunked to one chunk";

// ─────────────────────────────────────────────────────────────────────────────
// Population
// ─────────────────────────────────────────────────────────────────────────────

/// A column as dense rank ids in value order, with its value counts: what the window scans and
/// the baselines need.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Population {
    /// Each row's rank among the distinct non-null values (0 = smallest); `NULL` for a null.
    pub ids: Vec<u32>,
    /// Rows per id.
    pub counts: Vec<u64>,
    /// `cum[i]` = rows with id < i (d + 1 entries).
    pub cum: Vec<u64>,
    pub n_null: u64,
}

impl Population {
    /// Ranks rows `0..n` by key; `get(i)` is row i's key (None = null).
    fn from_keys<K, G>(n: usize, get: G) -> PolarsResult<Self>
    where
        K: Copy + Ord + Hash,
        G: Fn(usize) -> Option<K>,
    {
        let mut rank: HashMap<K, u32, RandomState> = HashMap::default();
        for k in (0..n).filter_map(&get) {
            rank.entry(k).or_insert(0);
        }
        if rank.len() >= NULL as usize {
            polars_bail!(ComputeError: "skip_index: {} distinct values exceed u32 ids", rank.len());
        }
        let mut keys: Vec<K> = rank.keys().copied().collect();
        keys.sort_unstable();
        for (i, k) in keys.iter().enumerate() {
            rank.insert(*k, i as u32);
        }
        let mut counts = vec![0u64; keys.len()];
        let ids: Vec<u32> = (0..n)
            .map(|i| match get(i) {
                Some(k) => {
                    let id = rank[&k];
                    counts[id as usize] += 1;
                    id
                }
                None => NULL,
            })
            .collect();
        Ok(Self::from_ids(ids, counts))
    }

    /// A population from row ids (`NULL` for nulls) and rows per id.
    pub(crate) fn from_ids(ids: Vec<u32>, counts: Vec<u64>) -> Self {
        let mut cum = Vec::with_capacity(counts.len() + 1);
        let mut total = 0u64;
        cum.push(0);
        for &c in &counts {
            total += c;
            cum.push(total);
        }
        let n_null = ids.len() as u64 - total;
        Self {
            ids,
            counts,
            cum,
            n_null,
        }
    }

    pub(crate) fn n(&self) -> u64 {
        self.ids.len() as u64
    }

    pub(crate) fn d(&self) -> u64 {
        self.counts.len() as u64
    }

    pub(crate) fn non_null(&self) -> u64 {
        self.cum[self.counts.len()]
    }
}

/// Canonical f64 bits (−0.0 = 0.0, one NaN) mapped so that integer order is numeric order, the
/// NaN after every number.
fn ordered_f64(x: f64) -> u64 {
    let b = canon_f64(x);
    if b >> 63 == 1 {
        !b
    } else {
        b | 1 << 63
    }
}

/// `ordered_f64` for f32.
fn ordered_f32(x: f32) -> u32 {
    let b = canon_f32(x) as u32;
    if b >> 31 == 1 {
        !b
    } else {
        b | 1 << 31
    }
}

/// Integer keys (temporal values, Decimal's unscaled i128, Enum codes).
fn numeric<T>(ca: &ChunkedArray<T>) -> PolarsResult<Population>
where
    T: PolarsNumericType,
    T::Native: Ord + Hash,
{
    let arr = ca.downcast_iter().next().expect(ONE_CHUNK);
    Population::from_keys(arr.len(), |i| arr.is_valid(i).then(|| arr.value(i)))
}

/// A supported column's population, ordered as ClickHouse compares values: numbers and
/// temporals by value (Decimal by its unscaled i128 at the column's fixed scale), Boolean
/// false < true, String / Binary bytewise, Categorical by its string, Enum by category order
/// (its codes), floats canonical with the NaN last.
pub(crate) fn population(s: &Series) -> PolarsResult<Population> {
    if !supported(s.dtype()) {
        polars_bail!(
            InvalidOperation: "skip_index: unsupported dtype {} (column {})", s.dtype(), s.name()
        );
    }
    let s = match s.dtype() {
        DataType::Categorical(_, _) => s.cast(&DataType::String)?,
        _ => s.clone(),
    };
    // One chunk, so row i is index i of a single Arrow array (zero-copy if already one).
    let phys = s.to_physical_repr().rechunk();
    match phys.dtype() {
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
            Population::from_keys(arr.len(), |i| {
                arr.is_valid(i).then(|| ordered_f32(arr.value(i)))
            })
        }
        DataType::Float64 => {
            let arr = phys.f64()?.downcast_iter().next().expect(ONE_CHUNK);
            Population::from_keys(arr.len(), |i| {
                arr.is_valid(i).then(|| ordered_f64(arr.value(i)))
            })
        }
        DataType::Boolean => {
            let arr = phys.bool()?.downcast_iter().next().expect(ONE_CHUNK);
            Population::from_keys(arr.len(), |i| arr.is_valid(i).then(|| arr.value(i)))
        }
        DataType::String => {
            let arr = phys.str()?.downcast_iter().next().expect(ONE_CHUNK);
            Population::from_keys(arr.len(), |i| arr.is_valid(i).then(|| arr.value(i)))
        }
        DataType::Binary => {
            let arr = phys.binary()?.downcast_iter().next().expect(ONE_CHUNK);
            Population::from_keys(arr.len(), |i| arr.is_valid(i).then(|| arr.value(i)))
        }
        dt => polars_bail!(ComputeError: "skip_index: unexpected physical dtype {dt}"),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Windows and their sums
// ─────────────────────────────────────────────────────────────────────────────

/// (effective window, number of windows): w = min(n, window_rows); n − w + 1 windows (none for
/// an empty column).
pub(crate) fn windows(n: u64, window_rows: u64) -> (u64, u64) {
    let w = n.min(window_rows);
    (w, if n == 0 { 0 } else { n - w + 1 })
}

/// Σa and Σa² of one skip numerator over windows.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Moments {
    pub sum: u128,
    pub sum_sq: u128,
}

impl Moments {
    fn push(&mut self, a: u64) {
        let a = u128::from(a);
        self.sum += a;
        self.sum_sq += a * a;
    }

    fn merge(self, o: Self) -> Self {
        Self {
            sum: self.sum + o.sum,
            sum_sq: self.sum_sq + o.sum_sq,
        }
    }

    /// (mean, population std) of a / `den` over `k` windows; NaN without a window or a
    /// denominator.
    pub(crate) fn mean_std(self, k: u64, den: u64) -> (f64, f64) {
        if k == 0 || den == 0 {
            return (f64::NAN, f64::NAN);
        }
        let scale = k as f64 * den as f64;
        (
            self.sum as f64 / scale,
            spread(k, self.sum, self.sum_sq).sqrt() / scale,
        )
    }
}

/// K·Σa² − (Σa)² (≥ 0): exact in i128, f64 once that overflows.
fn spread(k: u64, sum: u128, sum_sq: u128) -> f64 {
    let exact = || -> Option<i128> {
        let (k, s, s2) = (
            i128::from(k),
            i128::try_from(sum).ok()?,
            i128::try_from(sum_sq).ok()?,
        );
        k.checked_mul(s2)?.checked_sub(s.checked_mul(s)?)
    };
    exact().map_or_else(
        || (k as f64 * sum_sq as f64 - (sum as f64).powi(2)).max(0.0),
        |v| v as f64,
    )
}

/// Sums over a set of windows. `merge` is exact and associative, so any split of the windows
/// gives the same result.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct WindowSums {
    pub windows: u64,
    /// Numerators over N′.
    pub rows: Moments,
    /// Numerators over d.
    pub values: Moments,
    /// Set only: Σ entries, the most entries in a window, windows whose set overflowed.
    pub entries: u128,
    pub max_entries: u64,
    pub overflowed: u64,
}

impl WindowSums {
    pub(crate) fn push(&mut self, rows: u64, values: u64) {
        self.windows += 1;
        self.rows.push(rows);
        self.values.push(values);
    }

    pub(crate) fn merge(self, o: Self) -> Self {
        Self {
            windows: self.windows + o.windows,
            rows: self.rows.merge(o.rows),
            values: self.values.merge(o.values),
            entries: self.entries + o.entries,
            max_entries: self.max_entries.max(o.max_entries),
            overflowed: self.overflowed + o.overflowed,
        }
    }
}

/// Window starts `0..k`, in parallel ranges of at least max(2^16, 4w, `min_len`) starts,
/// merged. Each range re-reads the w − 1 rows before its first full window, so the ranges are
/// long next to w; `min_len` lets the set scan amortise its per-id array.
pub(crate) fn over_ranges<F>(k: u64, w: u64, min_len: u64, scan: F) -> WindowSums
where
    F: Fn(Range<usize>) -> WindowSums + Sync,
{
    let k = k as usize;
    let len = (RANGE as u64).max(4 * w).max(min_len) as usize;
    (0..k.div_ceil(len))
        .into_par_iter()
        .map(|r| scan(r * len..((r + 1) * len).min(k)))
        .reduce(WindowSums::default, WindowSums::merge)
}

// ─────────────────────────────────────────────────────────────────────────────
// Baselines' building blocks
// ─────────────────────────────────────────────────────────────────────────────

/// P(a w-row window of a uniformly random order of n rows holds none of m given rows)
/// = C(n − m, w) / C(n, w), in log space.
pub(crate) fn avoid(n: u64, w: u64, m: u64) -> f64 {
    debug_assert!(m <= n, "avoid: m > n");
    if m == 0 {
        return 1.0;
    }
    if n - m < w {
        return 0.0;
    }
    let lg = |x: u64| libm::lgamma(x as f64);
    (lg(n - m + 1) - lg(n - m - w + 1) - lg(n + 1) + lg(n - w + 1)).exp()
}

/// Φ, the standard normal CDF (libm's erfc: statrs' CDF is only ~1e-10 accurate).
pub(crate) fn phi(t: f64) -> f64 {
    0.5 * libm::erfc(-t / std::f64::consts::SQRT_2)
}

// ─────────────────────────────────────────────────────────────────────────────
// Result rows
// ─────────────────────────────────────────────────────────────────────────────

/// The metrics both techniques report for one column.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Common {
    pub n_rows: u64,
    pub n_non_null: u64,
    pub n_distinct: u64,
    pub window_rows: u64,
    pub n_windows: u64,
    pub skip_rows_mean: f64,
    pub skip_rows_std: f64,
    pub skip_values_mean: f64,
    pub skip_values_std: f64,
    pub expected_skip_rows: f64,
    pub expected_skip_values: f64,
}

impl Common {
    /// `expected` = (expected_skip_rows, expected_skip_values).
    pub(crate) fn new(pop: &Population, w: u64, sums: &WindowSums, expected: (f64, f64)) -> Self {
        let (skip_rows_mean, skip_rows_std) = sums.rows.mean_std(sums.windows, pop.non_null());
        let (skip_values_mean, skip_values_std) = sums.values.mean_std(sums.windows, pop.d());
        Self {
            n_rows: pop.n(),
            n_non_null: pop.non_null(),
            n_distinct: pop.d(),
            window_rows: w,
            n_windows: sums.windows,
            skip_rows_mean,
            skip_rows_std,
            skip_values_mean,
            skip_values_std,
            expected_skip_rows: expected.0,
            expected_skip_values: expected.1,
        }
    }

    /// One series per metric, named and ordered as the Python base's COMMON_METRICS.
    pub(crate) fn fields(rows: &[Self]) -> Vec<Series> {
        vec![
            u64s("n_rows", rows.iter().map(|r| r.n_rows)),
            u64s("n_non_null", rows.iter().map(|r| r.n_non_null)),
            u64s("n_distinct", rows.iter().map(|r| r.n_distinct)),
            u64s("window_rows", rows.iter().map(|r| r.window_rows)),
            u64s("n_windows", rows.iter().map(|r| r.n_windows)),
            f64s("skip_rows_mean", rows.iter().map(|r| r.skip_rows_mean)),
            f64s("skip_rows_std", rows.iter().map(|r| r.skip_rows_std)),
            f64s("skip_values_mean", rows.iter().map(|r| r.skip_values_mean)),
            f64s("skip_values_std", rows.iter().map(|r| r.skip_values_std)),
            f64s(
                "expected_skip_rows",
                rows.iter().map(|r| r.expected_skip_rows),
            ),
            f64s(
                "expected_skip_values",
                rows.iter().map(|r| r.expected_skip_values),
            ),
        ]
    }
}

pub(crate) fn u64s(name: &str, values: impl Iterator<Item = u64>) -> Series {
    UInt64Chunked::from_vec(name.into(), values.collect()).into_series()
}

pub(crate) fn f64s(name: &str, values: impl Iterator<Item = f64>) -> Series {
    Float64Chunked::from_vec(name.into(), values.collect()).into_series()
}

/// `column` then `fields`: one struct row per input column.
pub(crate) fn output(name: &str, inputs: &[Series], fields: Vec<Series>) -> PolarsResult<Series> {
    let column = StringChunked::from_iter(inputs.iter().map(|s| s.name().as_str()))
        .into_series()
        .with_name("column".into());
    let all: Vec<Series> = std::iter::once(column).chain(fields).collect();
    Ok(StructChunked::from_series(name.into(), inputs.len(), all.iter())?.into_series())
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_keys_order_numerically_nan_last() {
        let xs = [
            f64::NEG_INFINITY,
            -1.0,
            -0.0,
            0.0,
            1e-300,
            1.0,
            f64::INFINITY,
            f64::NAN,
            f64::from_bits(0x7ff8_0000_0000_0001),
        ];
        let keys: Vec<u64> = xs.iter().map(|&x| ordered_f64(x)).collect();
        assert!(keys.windows(2).all(|p| p[0] <= p[1]), "{keys:?}");
        assert_eq!(keys[2], keys[3]); // −0.0 = 0.0
        assert_eq!(keys[7], keys[8]); // one NaN
        assert!(keys[6] < keys[7]); // NaN after +inf
        let k32: Vec<u32> = [
            f32::NEG_INFINITY,
            -1.0,
            -0.0,
            0.0,
            1.0,
            f32::INFINITY,
            f32::NAN,
        ]
        .iter()
        .map(|&x| ordered_f32(x))
        .collect();
        assert!(k32.windows(2).all(|p| p[0] <= p[1]), "{k32:?}");
        assert_eq!(k32[2], k32[3]);
        assert!(k32[5] < k32[6]);
    }

    #[test]
    fn population_ranks_in_value_order() {
        let s = Series::new(
            "x".into(),
            &[Some("b"), None, Some("a"), Some("b"), Some("c"), None],
        );
        let p = population(&s).unwrap();
        assert_eq!(p.ids, [1, NULL, 0, 1, 2, NULL]);
        assert_eq!(p.counts, [1, 2, 1]);
        assert_eq!(p.cum, [0, 1, 3, 4]);
        assert_eq!((p.n(), p.non_null(), p.d(), p.n_null), (6, 4, 3, 2));
    }

    #[test]
    fn population_of_every_physical_family() {
        let ints = Series::new("x".into(), &[2i64, 0, 1, 0]);
        for dt in [
            DataType::Int8,
            DataType::UInt16,
            DataType::Float32,
            DataType::Float64,
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
            assert_eq!(population(&s).unwrap().ids, [2, 0, 1, 0], "{dt}");
        }
        let b = Series::new("x".into(), &[true, false, true]);
        assert_eq!(population(&b).unwrap().ids, [1, 0, 1]);
    }

    #[test]
    fn unsupported_dtype_is_invalid_operation() {
        let s = Series::new("x".into(), &[1i128, 2]);
        assert!(matches!(
            population(&s),
            Err(PolarsError::InvalidOperation(_))
        ));
    }

    #[test]
    fn windows_clamp_to_the_column() {
        assert_eq!(windows(10, 4), (4, 7));
        assert_eq!(windows(3, 8), (3, 1));
        assert_eq!(windows(0, 8), (0, 0));
    }

    #[test]
    fn avoid_edges_and_small_cases() {
        assert_eq!(avoid(5, 2, 0), 1.0);
        assert_eq!(avoid(4, 2, 3), 0.0);
        assert_eq!(avoid(3, 3, 1), 0.0); // the window is every row
        assert!((avoid(4, 2, 1) - 0.5).abs() < 1e-12);
        assert!((avoid(10, 3, 2) - 56.0 / 120.0).abs() < 1e-12);
    }

    #[test]
    fn mean_std_from_integer_sums() {
        let mut m = Moments::default();
        for a in [1u64, 3, 3, 5] {
            m.push(a);
        }
        let (mean, std) = m.mean_std(4, 8);
        let xs = [1.0f64 / 8.0, 3.0 / 8.0, 3.0 / 8.0, 5.0 / 8.0];
        let mu = xs.iter().sum::<f64>() / 4.0;
        let sd = (xs.iter().map(|x| (x - mu).powi(2)).sum::<f64>() / 4.0).sqrt();
        assert!((mean - mu).abs() < 1e-15 && (std - sd).abs() < 1e-15);
        assert!(Moments::default().mean_std(0, 8).0.is_nan());
        assert!(m.mean_std(4, 0).1.is_nan());
    }

    #[test]
    fn spread_overflow_falls_back_to_f64() {
        // K·Σa² = 10⁴⁰ overflows i128: 10¹⁰ windows of a = 10¹⁰ each, true spread 0
        let (k, a) = (10_000_000_000u64, 10_000_000_000u128);
        let s = spread(k, u128::from(k) * a, u128::from(k) * a * a);
        assert!((0.0..=1e25).contains(&s), "{s}");
    }

    #[test]
    fn window_sums_merge_adds() {
        let mut a = WindowSums::default();
        a.push(1, 2);
        a.entries = 3;
        a.max_entries = 3;
        let mut b = WindowSums::default();
        b.push(4, 0);
        b.entries = 5;
        b.max_entries = 5;
        b.overflowed = 1;
        let m = a.merge(b);
        assert_eq!(
            (
                m.windows,
                m.rows.sum,
                m.rows.sum_sq,
                m.values.sum,
                m.entries,
                m.max_entries,
                m.overflowed
            ),
            (2, 5, 17, 2, 8, 5, 1)
        );
        assert_eq!(WindowSums::default().merge(a), a);
    }

    #[test]
    fn over_ranges_covers_every_start_once() {
        let k = 3 * RANGE as u64 + 17;
        let sums = over_ranges(k, 4, 0, |r| {
            let mut s = WindowSums::default();
            for i in r {
                s.push(i as u64, 1);
            }
            s
        });
        assert_eq!(sums.windows, k);
        assert_eq!(sums.rows.sum, (u128::from(k) - 1) * u128::from(k) / 2);
        assert_eq!(
            over_ranges(0, 4, 0, |_| unreachable!()),
            WindowSums::default()
        );
    }

    #[test]
    fn supported_rejects_nested_and_wide_types() {
        use crate::techniques::runs_test::supported;
        for dt in [
            DataType::List(Box::new(DataType::Int64)),
            DataType::Array(Box::new(DataType::Int64), 2),
            DataType::Struct(vec![Field::new("a".into(), DataType::Int64)]),
            DataType::Null,
            DataType::Int128,
        ] {
            assert!(!supported(&dt), "{dt}");
        }
    }
}
