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

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Method {
    /// The exact distinct count (cardinality ratio >= 0.5, or nothing to estimate).
    Observed,
    /// The HyperLogLog count (streaming's sampling phase, ratio >= 0.5).
    Hll,
    Schnabel,
    Chao1,
}

impl Method {
    /// Lower-case name, as `est_method` and the dictionary candidate's evidence report it.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Method::Observed => "observed",
            Method::Hll => "hll",
            Method::Schnabel => "schnabel",
            Method::Chao1 => "chao1",
        }
    }
}

/// A level's distinct count: exact, or a HyperLogLog estimate with its relative
/// standard error and `seen`, the distinct values proven to exist (a lower bound).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Count {
    Exact(u64),
    Hll {
        estimate: f64,
        std_error: f64,
        seen: u64,
    },
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Estimate {
    pub est_cardinality: f64,
    pub est_low: Option<f64>,
    pub est_high: Option<f64>,
    pub method: Method,
}

/// The estimate picked by rule (spec 2026-10-01 s4), with `estimates_agree`.
/// n = non-null values; d = the count. d/n >= 0.5 -> the count itself (observed, or
/// HLL +- 3 sigma). Below: Schnabel when valid, else Chao1, floored at d (the estimate) and
/// at max(d - 3 sigma, seen) for HLL (the low end): observed values bound the population
/// from below.
pub(crate) fn pick_estimate(
    count: Count,
    n: u64,
    f1: u64,
    f2: u64,
    history: &[u64; 7],
) -> (Estimate, Option<bool>) {
    let (d, floor, high, method) = match count {
        Count::Exact(d) => (d as f64, d as f64, d as f64, Method::Observed),
        Count::Hll {
            estimate,
            std_error,
            seen,
        } => (
            estimate,
            (estimate * (1.0 - 3.0 * std_error)).max(seen as f64),
            estimate * (1.0 + 3.0 * std_error),
            Method::Hll,
        ),
    };
    if n == 0 {
        let zero = Estimate {
            est_cardinality: 0.0,
            est_low: Some(0.0),
            est_high: Some(0.0),
            method: Method::Observed,
        };
        return (zero, None);
    }
    if d / n as f64 >= 0.5 {
        let e = Estimate {
            est_cardinality: d,
            est_low: Some(floor),
            est_high: Some(high),
            method,
        };
        return (e, None);
    }
    // An HLL d just under n/2 can round to n/2, where Schnabel's own validity check
    // rejects it and Chao1 is used: harmless at the boundary.
    let du = d.round() as u64;
    let (c, c_lo, c_hi) = chao1(du, f1, f2);
    let sch = schnabel(history, du, n);
    let (e, lo, hi, method) = match sch {
        Some((s, s_lo, s_hi)) => (s, s_lo, s_hi, Method::Schnabel),
        None => (c, c_lo, c_hi, Method::Chao1),
    };
    let est = e.max(d);
    (
        Estimate {
            est_cardinality: est,
            est_low: Some(lo.max(floor)),
            est_high: Some(hi.max(est)),
            method,
        },
        sch.map(|(_, s_lo, s_hi)| c_lo <= s_hi && s_lo <= c_hi),
    )
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

    const H10: [u64; 7] = [0, 0, 0, 0, 0, 0, 10];

    #[test]
    fn no_values_is_observed_zero() {
        let (e, agree) = pick_estimate(Count::Exact(0), 0, 0, 0, &[0; 7]);
        assert_eq!(
            (e.method, e.est_cardinality, e.est_low, e.est_high),
            (Method::Observed, 0.0, Some(0.0), Some(0.0))
        );
        assert_eq!(agree, None);
    }

    #[test]
    fn high_ratio_is_the_count() {
        let (e, _) = pick_estimate(Count::Exact(10), 15, 4, 2, &H10);
        assert_eq!(
            (e.method, e.est_cardinality, e.est_low, e.est_high),
            (Method::Observed, 10.0, Some(10.0), Some(10.0))
        );
        let (h, agree) = pick_estimate(
            Count::Hll {
                estimate: 1_000.0,
                std_error: 0.01,
                seen: 0,
            },
            1_500,
            0,
            0,
            &[0; 7],
        );
        assert_eq!(
            (h.method, h.est_cardinality, agree),
            (Method::Hll, 1_000.0, None)
        );
        assert!(
            (h.est_low.unwrap() - 970.0).abs() < 1e-9
                && (h.est_high.unwrap() - 1_030.0).abs() < 1e-9
        );
    }

    #[test]
    fn low_ratio_is_schnabel_floored_at_the_count() {
        // Schnabel 9.52 [6.47, 16.38] < d = 10: the estimate and low end are floored at 10.
        let (e, agree) = pick_estimate(Count::Exact(10), 40, 4, 2, &H10);
        assert_eq!(
            (e.method, e.est_cardinality, e.est_low),
            (Method::Schnabel, 10.0, Some(10.0))
        );
        assert!((e.est_high.unwrap() - 16.378255262343956).abs() < 1e-9);
        assert_eq!(agree, Some(true));
    }

    #[test]
    fn low_ratio_without_recaptures_is_chao1() {
        let h = [3, 3, 0, 3, 0, 0, 0];
        let (e, agree) = pick_estimate(Count::Exact(9), 100, 4, 2, &h);
        let (c, lo, hi) = chao1(9, 4, 2);
        assert_eq!(
            (e.method, e.est_cardinality, e.est_low, e.est_high),
            (Method::Chao1, c, Some(lo.max(9.0)), Some(hi))
        );
        assert_eq!(agree, None);
    }

    #[test]
    fn hll_floor_is_three_standard_errors_below() {
        // Schnabel 9.52 [6.47, 16.38]: the estimate floors at 10, the low end at 10 - 3 * 0.1 = 9.7.
        let (e, _) = pick_estimate(
            Count::Hll {
                estimate: 10.0,
                std_error: 0.01,
                seen: 0,
            },
            40,
            4,
            2,
            &H10,
        );
        assert_eq!((e.method, e.est_cardinality), (Method::Schnabel, 10.0));
        assert!((e.est_low.unwrap() - 9.7).abs() < 1e-9);
        assert!((e.est_high.unwrap() - 16.378255262343956).abs() < 1e-9);
    }

    #[test]
    fn hll_low_end_is_floored_at_the_values_seen() {
        let hll = |seen| Count::Hll {
            estimate: 1_000.0,
            std_error: 0.05,
            seen,
        };
        // Ratio ≥ 0.5: 1000 · (1 − 0.15) = 850 < 900 seen.
        let (e, _) = pick_estimate(hll(900), 1_500, 0, 0, &[0; 7]);
        assert_eq!((e.method, e.est_low), (Method::Hll, Some(900.0)));
        let (e, _) = pick_estimate(hll(0), 1_500, 0, 0, &[0; 7]);
        assert!((e.est_low.unwrap() - 850.0).abs() < 1e-9);
        // Ratio < 0.5: Schnabel 9.52 [6.47, 16.38]; 10 · (1 − 0.15) = 8.5 < 10 seen.
        let small = |seen| Count::Hll {
            estimate: 10.0,
            std_error: 0.05,
            seen,
        };
        let (e, _) = pick_estimate(small(10), 40, 4, 2, &H10);
        assert_eq!((e.method, e.est_low), (Method::Schnabel, Some(10.0)));
        let (e, _) = pick_estimate(small(0), 40, 4, 2, &H10);
        assert!((e.est_low.unwrap() - 8.5).abs() < 1e-9);
    }

    #[test]
    fn method_names() {
        let names: Vec<_> = [
            Method::Observed,
            Method::Hll,
            Method::Schnabel,
            Method::Chao1,
        ]
        .iter()
        .map(|m| m.name())
        .collect();
        assert_eq!(names, ["observed", "hll", "schnabel", "chao1"]);
    }
}
