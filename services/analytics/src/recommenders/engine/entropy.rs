//! Entropy cells: exact, the bottom-k sample estimate, the AMS estimate.

use polars::prelude::AnyValue;

use crate::common::entropy_math::entropy_from_counts_iter;
use crate::techniques::describe::Row;

/// Streaming's estimator once the distinct sample overflows.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EntropyEstimator {
    Sample,
    #[default]
    Ams,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Entropy {
    pub value: f64,
    pub low: f64,
    pub high: f64,
    pub method: &'static str,
}

/// Bounds clamped to [0, log2 n].
fn clamp(x: f64, n: u64) -> f64 {
    let top = if n > 1 { (n as f64).log2() } else { 0.0 };
    x.clamp(0.0, top)
}

impl Entropy {
    pub(crate) fn exact(value: f64) -> Self {
        Entropy {
            value,
            low: value,
            high: value,
            method: "exact",
        }
    }

    /// Exact entropy from every non-null value's count plus `n_null` (n = all rows).
    pub(crate) fn from_counts(counts: impl Iterator<Item = u64>, n_null: u64, n: u64) -> Self {
        if n == 0 {
            return Self::exact(0.0);
        }
        let h = entropy_from_counts_iter(
            counts.chain((n_null > 0).then_some(n_null)),
            (n as f64).log2(),
            n as f64,
        );
        Self::exact(h.max(0.0)) // the SIMD sum can leave -4e-16 on a constant column
    }

    /// Bottom-k sample estimate: `counts` are the k' held values' exact counts,
    /// `d` = (est_cardinality, est_low, est_high) of the non-null distinct values.
    pub(crate) fn from_sample(counts: &[u64], d: (f64, f64, f64), n_null: u64, n: u64) -> Self {
        let k = counts.len() as f64;
        if n == 0 || counts.is_empty() {
            return Self::from_counts(counts.iter().copied(), n_null, n);
        }
        let nf = n as f64;
        let term = |c: u64| {
            if c > 1 {
                c as f64 * (c as f64).log2()
            } else {
                0.0
            }
        };
        let terms: Vec<f64> = counts.iter().map(|&c| term(c)).collect();
        let mean = terms.iter().sum::<f64>() / k;
        let s2 = if counts.len() > 1 {
            terms.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / (k - 1.0)
        } else {
            0.0
        };
        let null_term = term(n_null);
        let at = |dd: f64| {
            let dd = dd.max(k);
            let h = nf.log2() - (dd * mean + null_term) / nf;
            let sd = (dd * dd * (1.0 - k / dd).max(0.0) * s2 / k).sqrt() / nf;
            (h, sd)
        };
        let (h, _) = at(d.0);
        let (mut low, mut high) = (f64::INFINITY, f64::NEG_INFINITY);
        for dd in [d.0, d.1, d.2] {
            let (hh, sd) = at(dd);
            low = low.min(hh - 3.0 * sd);
            high = high.max(hh + 3.0 * sd);
        }
        Entropy {
            value: clamp(h, n),
            low: clamp(low, n),
            high: clamp(high, n),
            method: "sample",
        }
    }

    /// AMS estimate with its standard error, interval ±3σ.
    pub(crate) fn from_ams(value: f64, se: f64, n: u64) -> Self {
        Entropy {
            value: clamp(value, n),
            low: clamp(value - 3.0 * se, n),
            high: clamp(value + 3.0 * se, n),
            method: "ams",
        }
    }
}

/// The four `entropy*` cells; all null for None.
pub(crate) fn entropy_cells(e: Option<&Entropy>) -> Row {
    match e {
        Some(e) => vec![
            AnyValue::Float64(e.value),
            AnyValue::Float64(e.low),
            AnyValue::Float64(e.high),
            AnyValue::StringOwned(e.method.into()),
        ],
        None => vec![AnyValue::Null; 4],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_from_counts() {
        let e = Entropy::from_counts([2u64, 1].into_iter(), 2, 5);
        let want = -(2.0 * 0.4f64 * 0.4f64.log2() + 0.2 * 0.2f64.log2());
        assert!((e.value - want).abs() < 1e-12);
        assert_eq!((e.low, e.high, e.method), (e.value, e.value, "exact"));
        assert_eq!(Entropy::from_counts(std::iter::empty(), 0, 0).value, 0.0);
        assert_eq!(Entropy::from_counts([10u64].into_iter(), 0, 10).value, 0.0);
    }

    #[test]
    fn sample_with_equal_counts_is_exact_and_tight() {
        // D = 1000 values of count 4 (n = 4000), sampled 100 of them: T = 1000·4·2 = 8000,
        // H = log2(4000) − 8000/4000 = log2(1000); every term equal, so s² = 0.
        let e = Entropy::from_sample(&[4; 100], (1000.0, 1000.0, 1000.0), 0, 4000);
        assert!((e.value - 1000f64.log2()).abs() < 1e-9);
        assert!((e.high - e.low).abs() < 1e-9);
        assert_eq!(e.method, "sample");
    }

    #[test]
    fn sample_interval_widens_with_spread_and_d_uncertainty() {
        let counts: Vec<u64> = (1..=100).collect();
        let e = Entropy::from_sample(&counts, (1000.0, 900.0, 1100.0), 10, 60_000);
        assert!(e.low < e.value && e.value < e.high);
        assert!(e.high <= 60_000f64.log2() && e.low >= 0.0);
    }

    #[test]
    fn ams_interval_is_three_sigma_clamped() {
        let e = Entropy::from_ams(3.0, 0.1, 1 << 10);
        assert!((e.low - 2.7).abs() < 1e-12 && (e.high - 3.3).abs() < 1e-12);
        assert_eq!(Entropy::from_ams(0.1, 1.0, 16).low, 0.0);
    }
}
