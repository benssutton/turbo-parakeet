//! Float statistics: NaN / inf / fractional counts, decimal places, f32 round trip (group B).

use super::*;
use polars::prelude::*;
use polars_arrow::bitmap::Bitmap;
use rayon::prelude::*;

// ─────────────────────────────────────────────────────────────────────────────
// describe — float statistics (group B)
// ─────────────────────────────────────────────────────────────────────────────
//
// Decimal places come from the shortest round-trip representation (ryu) in the
// column's own width — f32 digits for Float32 — exponent-aware:
// max(0, fraction digits without trailing zeros − exponent).

#[derive(Default, Clone, Copy)]
pub(crate) struct FloatStats {
    pub n_nan: u64,
    pub n_inf: u64,
    pub n_fractional: u64,
    pub max_frac_digits: Option<u32>,
    /// Finite values that change under f64 → f32 → f64 (meaningless for Float32).
    pub n_f32_inexact: u64,
    /// Values equal to -0.0 (arrow-cast renders them "-0.0", a decimal "0").
    pub n_neg_zero: u64,
}

pub(crate) fn frac_digits(repr: &str) -> u32 {
    let repr = repr.trim_start_matches('-');
    let (mantissa, exp) = match repr.split_once(['e', 'E']) {
        Some((m, e)) => (m, e.parse::<i64>().unwrap_or(0)),
        None => (repr, 0),
    };
    let frac = mantissa
        .split_once('.')
        .map_or("", |(_, f)| f)
        .trim_end_matches('0');
    (frac.len() as i64 - exp).max(0) as u32
}

impl FloatStats {
    pub(crate) fn merge(self, o: Self) -> Self {
        Self {
            n_nan: self.n_nan + o.n_nan,
            n_inf: self.n_inf + o.n_inf,
            n_fractional: self.n_fractional + o.n_fractional,
            max_frac_digits: self.max_frac_digits.max(o.max_frac_digits),
            n_f32_inexact: self.n_f32_inexact + o.n_f32_inexact,
            n_neg_zero: self.n_neg_zero + o.n_neg_zero,
        }
    }

    fn add_f64(&mut self, x: f64, buf: &mut ryu::Buffer) {
        if x.is_nan() {
            self.n_nan += 1;
        } else if x.is_infinite() {
            self.n_inf += 1;
        } else {
            self.n_neg_zero += (x == 0.0 && x.is_sign_negative()) as u64;
            self.n_fractional += (x != x.trunc()) as u64;
            self.max_frac_digits = self
                .max_frac_digits
                .max(Some(frac_digits(buf.format_finite(x))));
            self.n_f32_inexact += ((x as f32) as f64 != x) as u64;
        }
    }

    fn add_f32(&mut self, x: f32, buf: &mut ryu::Buffer) {
        if x.is_nan() {
            self.n_nan += 1;
        } else if x.is_infinite() {
            self.n_inf += 1;
        } else {
            self.n_neg_zero += (x == 0.0 && x.is_sign_negative()) as u64;
            self.n_fractional += (x != x.trunc()) as u64;
            self.max_frac_digits = self
                .max_frac_digits
                .max(Some(frac_digits(buf.format_finite(x))));
        }
    }
}

/// Fold `add` over the valid values of one Arrow chunk, CHUNK values per task.
pub(crate) fn fold<T: Copy + Sync>(
    values: &[T],
    validity: Option<&Bitmap>,
    add: impl Fn(&mut FloatStats, T, &mut ryu::Buffer) + Sync,
) -> FloatStats {
    let validity = validity.filter(|bm| bm.unset_bits() > 0);
    values
        .par_chunks(CHUNK)
        .enumerate()
        .map(|(i, chunk)| {
            let (mut st, mut buf) = (FloatStats::default(), ryu::Buffer::new());
            for (j, &x) in chunk.iter().enumerate() {
                if validity.is_some_and(|bm| !bm.get_bit(i * CHUNK + j)) {
                    continue;
                }
                add(&mut st, x, &mut buf);
            }
            st
        })
        .reduce(FloatStats::default, FloatStats::merge)
}

pub(crate) fn float_stats(s: &Series) -> PolarsResult<Option<FloatStats>> {
    Ok(Some(match s.dtype() {
        DataType::Float64 => s
            .f64()?
            .downcast_iter()
            .map(|arr| fold(arr.values().as_slice(), arr.validity(), FloatStats::add_f64))
            .fold(FloatStats::default(), FloatStats::merge),
        DataType::Float32 => s
            .f32()?
            .downcast_iter()
            .map(|arr| fold(arr.values().as_slice(), arr.validity(), FloatStats::add_f32))
            .fold(FloatStats::default(), FloatStats::merge),
        _ => return Ok(None),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decimal_places_of_shortest_reprs() {
        for (repr, want) in [
            ("0.1", 1),
            ("1e-7", 7),
            ("1.5e20", 0),
            ("1.5e+20", 0),
            ("3.0", 0),
            ("-0.0", 0),
            ("123.45", 2),
            ("1.25e-3", 5),
        ] {
            assert_eq!(frac_digits(repr), want, "{repr}");
        }
    }

    #[test]
    fn f64_stats() {
        let s = Series::new(
            "x".into(),
            &[
                Some(0.1),
                Some(1e-7),
                Some(1.5e20),
                Some(3.0),
                Some(f64::INFINITY),
                Some(f64::NAN),
                None,
            ],
        );
        let st = float_stats(&s).unwrap().unwrap();
        assert_eq!(
            (st.n_nan, st.n_inf, st.n_fractional, st.max_frac_digits),
            (1, 1, 2, Some(7))
        );
        assert_eq!(st.n_f32_inexact, 3); // 0.1, 1e-7, 1.5e20
    }

    #[test]
    fn f32_uses_its_own_repr_and_non_floats_are_none() {
        let st = float_stats(&Series::new("x".into(), &[0.1f32, 0.25]))
            .unwrap()
            .unwrap();
        assert_eq!(st.max_frac_digits, Some(2));
        assert!(float_stats(&Series::new("x".into(), &[1i32]))
            .unwrap()
            .is_none());
        assert_eq!(
            float_stats(&Series::new("x".into(), &[f64::NAN]))
                .unwrap()
                .unwrap()
                .max_frac_digits,
            None
        );
    }

    #[test]
    fn float_negative_zero() {
        let st = float_stats(&Series::new("x".into(), &[0.0f64, -0.0, 1.0]))
            .unwrap()
            .unwrap();
        assert_eq!(st.n_neg_zero, 1);
    }
}
