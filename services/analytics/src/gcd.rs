// ─────────────────────────────────────────────────────────────────────────────
// Whole-column GCD (ClickHouse GCD-codec method)
// ─────────────────────────────────────────────────────────────────────────────
//
// For each integer-backed column, the GCD of the magnitudes of its raw physical
// values — the quantity ClickHouse's `GCD` codec divides by. Decimal uses its
// unscaled i128; Date/Datetime/Duration/Time use their i32/i64 physical values.
//
// Null policy: nulls are skipped (masked to 0, the GCD identity). All-null,
// all-zero and zero-row columns → 0. Non-integer-backed dtypes → null.
//
// `encode_series` is deliberately not used: it hashes decimals and
// reinterprets signed bits, destroying the magnitudes a GCD needs.
//
// Parallelism: columns in parallel, and each column's values in CHUNK-sized
// slices in parallel (GCD is associative and commutative).
//
// Early exit: the smallest non-zero GCD any supported dtype can have is one
// physical unit — physical 1 (1 for integers, 10^-scale for Decimal, 1 day
// for Date, 1 time-unit for Datetime/Duration, 1 ns for Time). Once any block
// of a column reaches 1 the column's GCD is final, so a shared flag stops
// every other chunk at its next block boundary.

use gcd::{binary_u128, binary_u64};
use polars::prelude::*;
use polars_arrow::bitmap::Bitmap;
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};

const CHUNK: usize = 1 << 16;
/// Values folded between checks of the early-exit flag.
const BLOCK: usize = 1 << 10;

// ─────────────────────────────────────────────────────────────────────────────
// Kernel
// ─────────────────────────────────────────────────────────────────────────────

/// Running-GCD step: `gcd(g, v)` = `binary_gcd(g, v % g)`.
///
/// Folding a column keeps a small running GCD `g` against large values `v`.
/// Stein's algorithm alone clears roughly one bit of `v` per subtract/shift
/// round (~50 rounds for a 52-bit value); one hardware remainder first brings
/// `v` below `g`, leaving Stein only a few rounds on two small operands.
/// `binary_gcd(g, 0) = g`, and `g == 0` (nothing folded yet) returns `v`.
#[inline]
fn step_u64(g: u64, v: u64) -> u64 {
    if g == 0 { v } else { binary_u64(g, v % g) }
}

/// u128 twin of [`step_u64`] (Decimal).
#[inline]
fn step_u128(g: u128, v: u128) -> u128 {
    if g == 0 { v } else { binary_u128(g, v % g) }
}

/// GCD of `mag(v)` over the valid slots of one Arrow array's values.
///
/// `values` and `validity` are both indexed from the array's logical start
/// (Arrow slices carry their own offsets), so chunk `i` covers bits
/// `i*CHUNK .. i*CHUNK + len`. Null slots contribute 0 via a select, not a
/// branch; without nulls the mask is skipped entirely.
///
/// Early exit: each chunk folds BLOCK values at a time, checking `done` before
/// each block and setting it when its running GCD reaches 1. A chunk that sees
/// `done` returns 1 — correct, because another chunk's GCD is 1, so the whole
/// column's is. Masked (null) slots contribute 0, so a 1 stored under a null
/// can never trigger the exit.
fn gcd_slice<T, U, M, G>(
    values: &[T],
    validity: Option<&Bitmap>,
    mag: M,
    gcd: G,
    done: &AtomicBool,
) -> U
where
    T: Copy + Sync,
    U: Copy + Send + Sync + Default + PartialEq + From<u8>,
    M: Fn(T) -> U + Sync,
    G: Fn(U, U) -> U + Sync + Send,
{
    let zero = U::default();
    let one = U::from(1u8);
    let validity = validity.filter(|bm| bm.unset_bits() > 0);
    values
        .par_chunks(CHUNK)
        .enumerate()
        .map(|(i, chunk)| {
            let chunk_bits = validity.map(|bm| bm.clone().sliced(i * CHUNK, chunk.len()));
            let mut bits = chunk_bits.as_ref().map(|bm| bm.iter());
            let mut g = zero;
            for block in chunk.chunks(BLOCK) {
                if done.load(Ordering::Relaxed) {
                    return one;
                }
                g = match bits.as_mut() {
                    None => block.iter().fold(g, |g, &v| gcd(g, mag(v))),
                    // `zip` pulls from `block` first, so `it` advances exactly
                    // block.len() bits and stays aligned for the next block.
                    Some(it) => block
                        .iter()
                        .zip(it.by_ref())
                        .fold(g, |g, (&v, ok)| gcd(g, if ok { mag(v) } else { zero })),
                };
                if g == one {
                    done.store(true, Ordering::Relaxed);
                    return one;
                }
            }
            g
        })
        .reduce(|| zero, &gcd)
}

/// GCD across every Arrow chunk of a ChunkedArray. One early-exit flag spans
/// all of the column's Arrow chunks.
fn ca_gcd<T, U, M, G>(ca: &ChunkedArray<T>, mag: M, gcd: G) -> U
where
    T: PolarsNumericType,
    U: Copy + Send + Sync + Default + PartialEq + From<u8>,
    M: Fn(T::Native) -> U + Sync + Copy,
    G: Fn(U, U) -> U + Sync + Send + Copy,
{
    let done = AtomicBool::new(false);
    ca.downcast_iter()
        .map(|arr| gcd_slice(arr.values().as_slice(), arr.validity(), mag, gcd, &done))
        .fold(U::default(), gcd)
}

// ─────────────────────────────────────────────────────────────────────────────
// Dispatch
// ─────────────────────────────────────────────────────────────────────────────

/// Checked on the LOGICAL dtype: Categorical/Enum are physically integer
/// codes and must not qualify.
fn is_integer_backed(dtype: &DataType) -> bool {
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
            | DataType::Decimal(_, _)
            | DataType::Date
            | DataType::Datetime(_, _)
            | DataType::Duration(_)
            | DataType::Time
    )
}

/// Whole-column GCD of `s`; `None` for non-integer-backed dtypes. The Int128 arm
/// reads Decimal's physical values, whose magnitudes fit in 38 digits.
pub(crate) fn series_gcd(s: &Series) -> PolarsResult<Option<i128>> {
    if !is_integer_backed(s.dtype()) {
        return Ok(None);
    }
    let phys = s.to_physical_repr();
    let g: u128 = match phys.dtype() {
        DataType::Int8 => ca_gcd(phys.i8()?, |v: i8| v.unsigned_abs() as u64, step_u64) as u128,
        DataType::Int16 => ca_gcd(phys.i16()?, |v: i16| v.unsigned_abs() as u64, step_u64) as u128,
        DataType::Int32 => ca_gcd(phys.i32()?, |v: i32| v.unsigned_abs() as u64, step_u64) as u128,
        DataType::Int64 => ca_gcd(phys.i64()?, |v: i64| v.unsigned_abs(), step_u64) as u128,
        DataType::UInt8 => ca_gcd(phys.u8()?, |v: u8| v as u64, step_u64) as u128,
        DataType::UInt16 => ca_gcd(phys.u16()?, |v: u16| v as u64, step_u64) as u128,
        DataType::UInt32 => ca_gcd(phys.u32()?, |v: u32| v as u64, step_u64) as u128,
        DataType::UInt64 => ca_gcd(phys.u64()?, |v: u64| v, step_u64) as u128,
        DataType::Int128 => ca_gcd(phys.i128()?, |v: i128| v.unsigned_abs(), step_u128),
        dt => {
            return Err(PolarsError::ComputeError(
                format!("column_gcd: unexpected physical dtype {dt}").into(),
            ))
        }
    };
    Ok(Some(g as i128))
}

// ─────────────────────────────────────────────────────────────────────────────
// Implementation
// ─────────────────────────────────────────────────────────────────────────────

pub(crate) fn column_gcd_impl(inputs: &[Series]) -> PolarsResult<Series> {
    let gcds: Vec<Option<i128>> = inputs
        .par_iter()
        .map(series_gcd)
        .collect::<PolarsResult<_>>()?;

    let dtypes: Vec<String> = inputs.iter().map(|s| s.dtype().to_string()).collect();

    let column_s = StringChunked::from_iter(inputs.iter().map(|s| s.name().as_str()))
        .into_series()
        .with_name("column".into());
    let dtype_s = StringChunked::from_iter(dtypes.iter().map(|s| s.as_str()))
        .into_series()
        .with_name("dtype".into());
    let gcd_s = Int128Chunked::from_iter_options("gcd".into(), gcds.into_iter())
        .into_decimal_unchecked(Some(38), 0)
        .into_series();

    let struct_ca = StructChunked::from_series(
        "column_gcd".into(),
        inputs.len(),
        [column_s, dtype_s, gcd_s].iter(),
    )?;
    Ok(struct_ca.into_series())
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use polars_arrow::bitmap::Bitmap;

    // Arrow has no plain 128-bit integer; decimal128(38, 0) is the widest native type.
    const GCD_DTYPE: DataType = DataType::Decimal(Some(38), Some(0));

    fn gcd_of(s: Series) -> Option<i128> {
        series_gcd(&s).unwrap()
    }

    #[test]
    fn multiples_of_known_gcd() {
        assert_eq!(gcd_of(Series::new("a".into(), &[12i64, -18, 30])), Some(6));
    }

    #[test]
    fn coprime_values_give_one() {
        assert_eq!(gcd_of(Series::new("a".into(), &[6i32, 10, 15])), Some(1));
    }

    #[test]
    fn zeros_are_identity_and_empty_is_zero() {
        assert_eq!(gcd_of(Series::new("a".into(), &[0i64, 0, 21, 0])), Some(21));
        assert_eq!(gcd_of(Series::new("a".into(), &[0i64, 0])), Some(0));
        assert_eq!(gcd_of(Series::new_empty("a".into(), &DataType::Int64)), Some(0));
    }

    #[test]
    fn nulls_skipped_and_all_null_is_zero() {
        assert_eq!(gcd_of(Series::new("a".into(), &[Some(12i64), None, Some(18)])), Some(6));
        assert_eq!(gcd_of(Series::new("a".into(), &[None::<i64>, None])), Some(0));
    }

    #[test]
    fn signed_and_unsigned_extremes() {
        assert_eq!(gcd_of(Series::new("a".into(), &[i8::MIN])), Some(128));
        assert_eq!(gcd_of(Series::new("a".into(), &[i64::MIN])), Some(1i128 << 63));
        assert_eq!(gcd_of(Series::new("a".into(), &[i64::MIN, 1i64 << 62])), Some(1i128 << 62));
        assert_eq!(gcd_of(Series::new("a".into(), &[u64::MAX])), Some(u64::MAX as i128));
        let dec = |v: &[i128]| Int128Chunked::from_slice("a".into(), v).into_decimal_unchecked(Some(38), 0).into_series();
        assert_eq!(gcd_of(dec(&[10i128.pow(38) - 1])), Some(10i128.pow(38) - 1));
        assert_eq!(gcd_of(dec(&[-(10i128.pow(37)), 10i128.pow(36)])), Some(10i128.pow(36)));
        assert_eq!(gcd_of(Series::new("a".into(), &[12i128])), None); // Int128 is not integer-backed
    }

    #[test]
    fn step_matches_binary_gcd() {
        let pairs_64 = [
            (0u64, 0u64), (0, 7), (7, 0), (3_600, (1u64 << 52) + 7_200), (12, 18),
            (u64::MAX, 3), (3, u64::MAX), (1u64 << 63, 1u64 << 62), (1, u64::MAX),
        ];
        for (g, v) in pairs_64 {
            assert_eq!(step_u64(g, v), binary_u64(g, v), "step_u64({g}, {v})");
        }
        let pairs_128 = [
            (0u128, 0u128), (0, 5), (5, 0), (10u128.pow(20), 10u128.pow(35) + 10u128.pow(20)),
            (1u128 << 127, 1u128 << 126), (u128::MAX, 15), (1u128 << 127, 0),
        ];
        for (g, v) in pairs_128 {
            assert_eq!(step_u128(g, v), binary_u128(g, v), "step_u128({g}, {v})");
        }
    }

    fn slice_gcd(values: &[u64], validity: Option<&Bitmap>) -> u64 {
        gcd_slice(values, validity, |v: u64| v, step_u64, &AtomicBool::new(false))
    }

    #[test]
    fn masked_null_payloads_ignored() {
        // 1s sit under null slots. A correct mask ignores them: the result stays
        // 6, and they must not trigger the gcd == 1 early exit either.
        let values = vec![12u64, 1, 18, 1];
        let validity = Bitmap::from_iter([true, false, true, false]);
        assert_eq!(slice_gcd(&values, Some(&validity)), 6);
    }

    #[test]
    fn crosses_parallel_chunk_and_block_boundaries() {
        let n = 3 * CHUNK + 17;
        let mut values = vec![12u64; n];
        values[2 * CHUNK + BLOCK + 5] = 18;
        assert_eq!(slice_gcd(&values, None), 6);

        // A masked spoiler just past the first chunk boundary, and a valid 18
        // in a later block of the same chunk (bit iterator must stay aligned).
        let mut masked = vec![12u64; n];
        masked[CHUNK + 3] = 1;
        masked[CHUNK + 3 * BLOCK + 1] = 18;
        let validity = Bitmap::from_iter((0..n).map(|i| i != CHUNK + 3));
        assert_eq!(slice_gcd(&masked, Some(&validity)), 6);
    }

    #[test]
    fn early_exit_stops_scanning() {
        use std::sync::atomic::AtomicUsize;
        // A 1 in the first block: the column's GCD is final after BLOCK values.
        let n = 64 * CHUNK;
        let mut values = vec![12u64; n];
        values[0] = 1;
        let visited = AtomicUsize::new(0);
        let mag = |v: u64| {
            visited.fetch_add(1, Ordering::Relaxed);
            v
        };
        let g = gcd_slice(&values[..], None, mag, step_u64, &AtomicBool::new(false));
        assert_eq!(g, 1);
        // Chunks already running when the flag is set stop at their next block,
        // so only a small fraction of the n values is ever read.
        let seen = visited.load(Ordering::Relaxed);
        assert!(seen < n / 4, "early exit did not stop the scan: visited {seen} of {n}");
    }

    #[test]
    fn no_early_exit_while_gcd_above_one() {
        // GCD bottoms out at 2 (never 1), so every value must be read.
        use std::sync::atomic::AtomicUsize;
        let n = 4 * CHUNK;
        let mut values = vec![4u64; n];
        values[n - 1] = 2;
        let visited = AtomicUsize::new(0);
        let mag = |v: u64| {
            visited.fetch_add(1, Ordering::Relaxed);
            v
        };
        assert_eq!(gcd_slice(&values[..], None, mag, step_u64, &AtomicBool::new(false)), 2);
        assert_eq!(visited.load(Ordering::Relaxed), n);
    }

    #[test]
    fn early_exit_across_arrow_chunks() {
        // Chunk 1 reaches 1; chunk 2 (a huge multiple of 12) must not change it.
        let mut s = Series::new("a".into(), &[12i64, 7]);
        s.append(&Series::new("a".into(), vec![12i64; 2 * CHUNK])).unwrap();
        assert_eq!(s.n_chunks(), 2);
        assert_eq!(gcd_of(s), Some(1));
    }

    #[test]
    fn temporal_uses_physical_values() {
        let d = Series::new("d".into(), &[7i32, 14, 21]).cast(&DataType::Date).unwrap();
        assert_eq!(gcd_of(d), Some(7));
        let dt = Series::new("t".into(), &[3_600_000_000i64, 7_200_000_000])
            .cast(&DataType::Datetime(TimeUnit::Microseconds, None))
            .unwrap();
        assert_eq!(gcd_of(dt), Some(3_600_000_000));
    }

    #[test]
    fn non_integer_dtypes_are_none() {
        assert_eq!(gcd_of(Series::new("f".into(), &[2.0f64, 4.0])), None);
        assert_eq!(gcd_of(Series::new("s".into(), &["a", "b"])), None);
        assert_eq!(gcd_of(Series::new("b".into(), &[true, false])), None);
        // Categorical is physically u32 codes — must still be None.
        let cat = Series::new("c".into(), &["x", "y", "x"])
            .cast(&DataType::from_categories(Categories::global()))
            .unwrap();
        assert_eq!(gcd_of(cat), None);
    }

    #[test]
    fn impl_rows_follow_input_order() {
        let a = Series::new("a".into(), &[4i64, 8]);
        let b = Series::new("b".into(), &["x", "y"]);
        let c = Series::new("c".into(), &[9u8, 6]);
        let out = column_gcd_impl(&[a, b, c]).unwrap();
        let df = out.into_frame().unnest(["column_gcd"]).unwrap();
        let cols: Vec<_> = df.column("column").unwrap().str().unwrap().into_no_null_iter().collect();
        let dtypes: Vec<_> = df.column("dtype").unwrap().str().unwrap().into_no_null_iter().collect();
        let gcd = df.column("gcd").unwrap();
        assert_eq!(gcd.dtype(), &GCD_DTYPE);
        let gcds: Vec<_> = gcd.decimal().unwrap().physical().into_iter().collect();
        assert_eq!(cols, ["a", "b", "c"]);
        assert_eq!(dtypes, ["i64", "str", "u8"]);
        assert_eq!(gcds, [Some(4), None, Some(3)]);
    }
}
