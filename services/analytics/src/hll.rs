//! HyperLogLog distinct counting (Flajolet et al. 2007) with Ertl's improved estimator
//! ("New cardinality estimation algorithms for HyperLogLog sketches", 2017), which needs
//! no bias tables or range switches. Callers pass one well-mixed 64-bit hash per value
//! (`hash_key`). No Polars or Arrow types: a later entry point can wrap it as is.

use std::hash::BuildHasher;

/// A value key (`shared::encode_series`) spread over 64 bits. Integer keys are raw
/// values, so they are always mixed; foldhash's quality variant finishes with a full
/// avalanche. Not stable across foldhash versions: never persist a sketch.
#[inline]
pub(crate) fn hash_key(key: u64) -> u64 {
    foldhash::quality::FixedState::default().hash_one(key)
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Hll {
    p: u8,
    registers: Vec<u8>,
}

impl Hll {
    /// 2^p one-byte registers; p in 4..=18.
    pub(crate) fn new(p: u8) -> Self {
        assert!(
            (4..=18).contains(&p),
            "HyperLogLog precision {p} outside 4..=18"
        );
        Hll {
            p,
            registers: vec![0; 1 << p],
        }
    }

    /// Bucket = the top p bits; rank = leading zeros of the other 64 − p bits + 1.
    #[inline]
    pub(crate) fn insert(&mut self, hash: u64) {
        let i = (hash >> (64 - self.p)) as usize;
        let w = hash << self.p;
        let rank = if w == 0 {
            65 - self.p as u32
        } else {
            w.leading_zeros() + 1
        } as u8;
        let r = &mut self.registers[i];
        if rank > *r {
            *r = rank;
        }
    }

    /// Ertl's improved estimator: α∞·m² / z, with z built from the register histogram
    /// (σ corrects for empty registers, τ for saturated ones). 0 for an empty sketch.
    pub(crate) fn estimate(&self) -> f64 {
        let m = self.registers.len() as f64;
        let q = 64 - self.p as usize;
        let mut c = vec![0u64; q + 2];
        for &r in &self.registers {
            c[r as usize] += 1;
        }
        if c[0] == self.registers.len() as u64 {
            return 0.0;
        }
        let mut z = m * tau(1.0 - c[q + 1] as f64 / m);
        for k in (1..=q).rev() {
            z = 0.5 * (z + c[k] as f64);
        }
        z += m * sigma(c[0] as f64 / m);
        m * m / (2.0 * std::f64::consts::LN_2) / z
    }

    /// Relative standard error of `estimate`: 1.04 / √m.
    pub(crate) fn std_error(&self) -> f64 {
        1.04 / (self.registers.len() as f64).sqrt()
    }
}

fn sigma(mut x: f64) -> f64 {
    if x == 1.0 {
        return f64::INFINITY;
    }
    let (mut y, mut z) = (1.0, x);
    loop {
        x *= x;
        let previous = z;
        z += x * y;
        y += y;
        if z == previous {
            return z;
        }
    }
}

fn tau(mut x: f64) -> f64 {
    if x == 0.0 || x == 1.0 {
        return 0.0;
    }
    let (mut y, mut z) = (1.0, 1.0 - x);
    loop {
        x = x.sqrt();
        let previous = z;
        y *= 0.5;
        z -= (1.0 - x).powi(2) * y;
        if z == previous {
            return z / 3.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sketch(n: u64, seed: u64) -> Hll {
        let mut h = Hll::new(14);
        for v in 0..n {
            h.insert(hash_key(v + (seed << 40)));
        }
        h
    }

    #[test]
    fn empty_is_zero() {
        assert_eq!(Hll::new(14).estimate(), 0.0);
    }

    #[test]
    fn estimates_within_three_standard_errors() {
        for n in [1u64, 100, 10_000, 1_000_000] {
            for seed in 0..3 {
                let h = sketch(n, seed);
                let e = h.estimate();
                let tolerance = (3.0 * h.std_error() * n as f64).max(1.0);
                assert!(
                    (e - n as f64).abs() <= tolerance,
                    "n={n} seed={seed} estimate={e}"
                );
            }
        }
    }

    #[test]
    fn duplicates_do_not_count() {
        let mut h = Hll::new(14);
        for _ in 0..10 {
            for v in 0..1_000u64 {
                h.insert(hash_key(v));
            }
        }
        assert!((h.estimate() - 1_000.0).abs() < 30.0, "{}", h.estimate());
    }

    #[test]
    #[should_panic(expected = "outside 4..=18")]
    fn precision_is_bounded() {
        Hll::new(3);
    }

    #[test]
    #[should_panic(expected = "outside 4..=18")]
    fn precision_above_bound_panics() {
        Hll::new(19);
    }

    #[test]
    fn precision_bounds_construct() {
        Hll::new(4);
        Hll::new(18);
    }

    #[test]
    fn saturated_registers_are_corrected_monotonically() {
        let build = |top: u8| {
            let mut h = Hll::new(4);
            for (i, r) in h.registers.iter_mut().enumerate() {
                *r = if i < 4 { top } else { 3 };
            }
            h
        };
        let q = 64 - 4;
        let saturated = build(q + 1).estimate();
        let lower = build(10).estimate();
        assert!(saturated.is_finite() && saturated > 0.0, "{saturated}");
        assert!(saturated > lower, "{saturated} <= {lower}");
    }

    #[test]
    fn tau_boundaries() {
        assert_eq!(tau(0.0), 0.0);
        assert_eq!(tau(1.0), 0.0);
        let t = tau(0.5);
        assert!(t > 0.0 && t < 1.0 / 3.0, "{t}");
    }
}
