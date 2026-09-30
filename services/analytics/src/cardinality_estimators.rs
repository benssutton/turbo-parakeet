// ─────────────────────────────────────────────────────────────────────────────
// cardinality_estimators — number of distinct values in a population
// ─────────────────────────────────────────────────────────────────────────────
//
// Ports of analytics/describe/estimators.py (the Python reference that feeds
// Describe's conclusions); recommend.rs uses them to size dictionaries. Picked by
// rule, never averaged. 95% closed-form intervals (z = 1.96).

pub(crate) const Z: f64 = 1.96;

/// Bias-corrected Chao1 with Chao's (1987) log-normal interval; variance as in the
/// EstimateS user guide. (estimate, low, high).
pub(crate) fn chao1(d: u64, f1: u64, f2: u64) -> (f64, f64, f64) {
    let (d, f1, f2) = (d as f64, f1 as f64, f2 as f64);
    let s = d + f1 * (f1 - 1.0) / (2.0 * (f2 + 1.0));
    let t = s - d;
    if t <= 0.0 {
        return (s, d, d);
    }
    let var = if f2 > 0.0 {
        f1 * (f1 - 1.0) / (2.0 * (f2 + 1.0))
            + f1 * (2.0 * f1 - 1.0).powi(2) / (4.0 * (f2 + 1.0).powi(2))
            + f1.powi(2) * f2 * (f1 - 1.0).powi(2) / (4.0 * (f2 + 1.0).powi(4))
    } else {
        f1 * (f1 - 1.0) / 2.0 + f1 * (2.0 * f1 - 1.0).powi(2) / 4.0 - f1.powi(4) / (4.0 * s)
    };
    let k = (Z * (1.0 + var.max(0.0) / (t * t)).ln().sqrt()).exp();
    (s, d + t / k, d + t * k)
}

/// Schnabel (multi-sample Lincoln–Petersen) over the three split subsets;
/// `history[k-1]` = distinct values whose subset mask is k. None unless n > 0,
/// d/n < 0.5 and there is at least one recapture.
pub(crate) fn schnabel(history: &[u64; 7], d: u64, n: u64) -> Option<(f64, f64, f64)> {
    if n == 0 || d as f64 / n as f64 >= 0.5 {
        return None;
    }
    let h = |masks: &[usize]| masks.iter().map(|&m| history[m - 1]).sum::<u64>() as f64;
    let (s1, s2, s3) = (h(&[1, 3, 5, 7]), h(&[2, 3, 6, 7]), h(&[4, 5, 6, 7]));
    let union12 = h(&[1, 2, 3, 5, 6, 7]);
    let r = h(&[3, 7]) + h(&[5, 6, 7]);
    if r < 1.0 {
        return None;
    }
    let a = s2 * s1 + s3 * union12;
    let r_lo = r * (1.0 - 1.0 / (9.0 * r) - Z / (3.0 * r.sqrt())).powi(3);
    let r_hi = (r + 1.0) * (1.0 - 1.0 / (9.0 * (r + 1.0)) + Z / (3.0 * (r + 1.0).sqrt())).powi(3);
    Some((a / (r + 1.0), a / r_hi, a / r_lo))
}

/// Haas–Stokes Duj1: a sample of n non-null values at sampling fraction q.
pub(crate) fn duj1(d: u64, f1: u64, n: u64, q: f64) -> f64 {
    if n == 0 {
        0.0
    } else {
        d as f64 / (1.0 - (1.0 - q) * f1 as f64 / n as f64)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Method {
    Exact,
    Duj1,
    Schnabel,
    Chao1,
    /// Streaming: distinct tracking stopped past `categorical_threshold`; no estimate.
    Overflowed,
}

impl Method {
    /// Lower-case name, as the dictionary candidate's evidence reports it.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Method::Exact => "exact",
            Method::Duj1 => "duj1",
            Method::Schnabel => "schnabel",
            Method::Chao1 => "chao1",
            Method::Overflowed => "overflowed",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Estimate {
    pub est_cardinality: f64,
    pub est_low: Option<f64>,
    pub est_high: Option<f64>,
    pub method: Method,
}

/// The estimate picked by rule: q == 1 → exact; q < 1 → Duj1; Schnabel valid →
/// Schnabel; else Chao1. q = frame rows / population rows (None: unknown).
pub(crate) fn estimate(
    d: u64,
    n: u64,
    f1: u64,
    f2: u64,
    history: &[u64; 7],
    q: Option<f64>,
) -> Estimate {
    let (c, c_lo, c_hi) = chao1(d, f1, f2);
    match (q, schnabel(history, d, n)) {
        (Some(1.0), _) => {
            let d = d as f64;
            Estimate {
                est_cardinality: d,
                est_low: Some(d),
                est_high: Some(d),
                method: Method::Exact,
            }
        }
        (Some(q), _) => Estimate {
            est_cardinality: duj1(d, f1, n, q),
            est_low: None,
            est_high: None,
            method: Method::Duj1,
        },
        (None, Some((s, lo, hi))) => Estimate {
            est_cardinality: s,
            est_low: Some(lo),
            est_high: Some(hi),
            method: Method::Schnabel,
        },
        (None, None) => Estimate {
            est_cardinality: c,
            est_low: Some(c_lo),
            est_high: Some(c_hi),
            method: Method::Chao1,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: (f64, f64, f64), b: (f64, f64, f64)) -> bool {
        [(a.0, b.0), (a.1, b.1), (a.2, b.2)]
            .iter()
            .all(|(x, y)| (x - y).abs() <= 1e-9 * y.abs().max(1.0))
    }

    #[test]
    fn chao1_cases() {
        assert!(close(
            chao1(10, 4, 2),
            (12.0, 10.249903382590167, 26.00618590489368)
        ));
        assert!(close(
            chao1(5, 3, 0),
            (8.0, 5.369121767830802, 29.38219792045809)
        ));
        assert_eq!(chao1(7, 0, 3), (7.0, 7.0, 7.0));
        assert_eq!(chao1(7, 1, 0), (7.0, 7.0, 7.0));
    }

    #[test]
    fn schnabel_cases() {
        assert!(close(
            schnabel(&[0, 0, 0, 0, 0, 0, 10], 10, 30_000).unwrap(),
            (9.523809523809524, 6.474567576908709, 16.378255262343956)
        ));
        assert!(close(
            schnabel(&[5, 5, 5, 2, 2, 2, 1], 22, 1_000).unwrap(),
            (25.75, 15.69849888622768, 56.349688010901595)
        ));
        assert!(schnabel(&[5, 5, 5, 2, 2, 2, 1], 22, 44).is_none());
        assert!(schnabel(&[3, 3, 0, 3, 0, 0, 0], 9, 100).is_none());
        assert!(schnabel(&[0; 7], 0, 0).is_none());
    }

    #[test]
    fn overflowed_method_name() {
        assert_eq!(Method::Overflowed.name(), "overflowed");
    }

    #[test]
    fn duj1_cases() {
        assert!((duj1(10, 4, 40, 0.5) - 10.526315789473685).abs() < 1e-12);
        assert_eq!(duj1(0, 0, 0, 0.5), 0.0);
    }

    #[test]
    fn estimate_picks_exact_then_duj1_then_schnabel_then_chao1() {
        let h = [0, 0, 0, 0, 0, 0, 10];
        let exact = estimate(10, 40, 4, 2, &h, Some(1.0));
        assert_eq!(
            (exact.method, exact.est_cardinality, exact.est_high),
            (Method::Exact, 10.0, Some(10.0))
        );
        let duj = estimate(10, 40, 4, 2, &h, Some(0.5));
        assert_eq!((duj.method, duj.est_high), (Method::Duj1, None));
        assert!((duj.est_cardinality - 10.526315789473685).abs() < 1e-12);
        assert_eq!(estimate(10, 40, 4, 2, &h, None).method, Method::Schnabel);
        let chao = estimate(10, 15, 4, 2, &h, None);
        assert_eq!((chao.method, chao.est_cardinality), (Method::Chao1, 12.0));
    }
}
