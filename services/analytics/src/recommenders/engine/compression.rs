//! Analytic pruning, then emulation: GCD and compression choices (spec 2026-10-19 §4).
#![cfg_attr(not(test), allow(dead_code))]

use crate::common::clickhouse_sizes::{Chain, Compression};

/// The four keywords (spec §4.3).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CompressionParams {
    pub zstd_levels: Vec<i32>,
    pub zstd_level_tolerance: f64,
    pub zstd_min_saving: f64,
    pub gcd_min_saving: f64,
}

impl Default for CompressionParams {
    fn default() -> Self {
        CompressionParams {
            zstd_levels: vec![1, 3, 9],
            zstd_level_tolerance: 0.01,
            zstd_min_saving: 0.10,
            gcd_min_saving: 0.01,
        }
    }
}

/// Bytes holding the magnitude `m` (plus a sign bit when `signed`); 0 → 1.
pub(crate) fn bytes_needed(m: u128, signed: bool) -> u32 {
    let bits = 128 - m.leading_zeros() + signed as u32;
    bits.div_ceil(8).max(1)
}

/// Spec §4.1: GCD is a candidate only if dividing by `g` frees a byte at the maximum
/// magnitude. Returns (survives, evidence).
pub(crate) fn gcd_survives(g: u128, max_abs: u128, signed: bool) -> (bool, String) {
    if g <= 1 {
        return (false, format!("gcd {g}"));
    }
    let (before, after) = (
        bytes_needed(max_abs, signed),
        bytes_needed(max_abs / g, signed),
    );
    (
        after < before,
        format!("gcd {g}: {before} → {after} bytes at max |value| {max_abs}"),
    )
}

/// Shannon entropy in bits per byte of a byte histogram.
pub(crate) fn byte_entropy(h: &[u64; 256]) -> f64 {
    let n: u64 = h.iter().sum();
    if n == 0 {
        return 0.0;
    }
    let nf = n as f64;
    h.iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / nf;
            -p * p.log2()
        })
        .sum()
}

/// Spec §4.2's shortcut: ZSTD cannot save `min_saving` over LZ4. Returns (fires, evidence).
pub(crate) fn incompressible(
    h: &[u64; 256],
    distinct_ratio: f64,
    min_saving: f64,
) -> (bool, String) {
    let hb = byte_entropy(h);
    let bound = 1.0 - hb / 8.0;
    (
        bound < min_saving && distinct_ratio >= 0.5,
        format!(
            "byte entropy {hb:.3} bits (ZSTD gain ≤ {:.1}%), distinct ratio {distinct_ratio:.3}",
            bound * 100.0
        ),
    )
}

/// Spec §4.2: the lowest level within tolerance of the best ZSTD size, then ZSTD over LZ4
/// only if it saves `min_saving`. `zstd`: (level, bytes), any order, may be empty.
pub(crate) fn choose_compression(
    lz4: u64,
    zstd: &[(i32, u64)],
    p: &CompressionParams,
) -> (Compression, u64) {
    let Some(best) = zstd.iter().map(|&(_, b)| b).min() else {
        return (Compression::Lz4, lz4);
    };
    let limit = best as f64 * (1.0 + p.zstd_level_tolerance);
    let (level, bytes) = zstd
        .iter()
        .filter(|&&(_, b)| b as f64 <= limit)
        .min_by_key(|&&(l, _)| l)
        .copied()
        .expect("the best level is within its own tolerance");
    if bytes as f64 <= (1.0 - p.zstd_min_saving) * lz4 as f64 {
        (Compression::Zstd(level), bytes)
    } else {
        (Compression::Lz4, lz4)
    }
}

/// One candidate row of `clickhouse_candidates` (and the Arrow IPC evidence).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CodecCandidate {
    pub codecs: String,
    pub predicted_bytes: Option<u64>,
    pub outcome: &'static str, // "chosen" | "rejected" | "pruned"
    pub evidence: String,
}

pub(crate) struct Decision {
    pub chain: Chain,
    pub bytes: u64,
    pub candidates: Vec<CodecCandidate>,
}

/// One considered chain family (GCD or not): its measured rows and its pick.
struct Family {
    gcd: bool,
    /// (compression, bytes) of every measured chain.
    measured: Vec<(Compression, u64)>,
    pick: (Compression, u64),
    /// Best ZSTD size among the measured levels.
    zstd_best: Option<u64>,
}

fn pruned(codecs: String, evidence: String) -> CodecCandidate {
    CodecCandidate {
        codecs,
        predicted_bytes: None,
        outcome: "pruned",
        evidence,
    }
}

/// Why a measured chain (not the family's pick) lost.
fn lost_to_pick(c: Compression, b: u64, f: &Family, p: &CompressionParams) -> String {
    let saving = p.zstd_min_saving * 100.0;
    match (c, f.pick.0) {
        (Compression::Lz4, Compression::Zstd(l)) => {
            format!("LZ4 {b}: ZSTD({l}) {} saves ≥ {saving}%", f.pick.1)
        }
        (Compression::Zstd(l), Compression::Lz4) => {
            format!("ZSTD({l}) {b} saves < {saving}% over LZ4 {}", f.pick.1)
        }
        (Compression::Zstd(l), Compression::Zstd(k)) => {
            let best = f.zstd_best.unwrap_or(b);
            let tol = p.zstd_level_tolerance * 100.0;
            if b as f64 > best as f64 * (1.0 + p.zstd_level_tolerance) {
                format!("ZSTD({l}) {b} more than {tol}% above the best ZSTD {best}")
            } else {
                format!(
                    "ZSTD({l}) {b} within {tol}% of ZSTD({k}) {}: lower level wins",
                    f.pick.1
                )
            }
        }
        _ => "not the family's pick".to_string(),
    }
}

/// Spec §4.1–4.2. `measure(chain)` emulates one chain → (bytes, histogram of the
/// uncompressed bytes); `gcd` is Some((g, max_abs, signed)) when GCD applies to the column,
/// None otherwise or when unknown (`gcd_note` then says why, or is "" when GCD does not
/// apply at all — then no GCD row is emitted).
pub(crate) fn decide(
    mut measure: impl FnMut(Chain) -> Result<(u64, [u64; 256]), String>,
    gcd: Option<(u128, u128, bool)>,
    gcd_note: &str,
    distinct_ratio: f64,
    p: &CompressionParams,
) -> Result<Decision, String> {
    let mut pruned_rows = Vec::new();
    let mut flags = vec![false];
    match gcd {
        Some((g, max_abs, signed)) => {
            let (ok, evidence) = gcd_survives(g, max_abs, signed);
            if ok {
                flags.push(true);
            } else {
                pruned_rows.push(pruned("GCD".to_string(), evidence));
            }
        }
        None if !gcd_note.is_empty() => {
            pruned_rows.push(pruned("GCD".to_string(), gcd_note.to_string()));
        }
        None => {}
    }

    let mut families = Vec::new();
    for gcd in flags {
        let chain = |compression| Chain { gcd, compression };
        let (lz4, hist) = measure(chain(Compression::Lz4))?;
        let mut measured = vec![(Compression::Lz4, lz4)];
        let (fires, shortcut) = incompressible(&hist, distinct_ratio, p.zstd_min_saving);
        let (pick, zstd_best);
        if fires {
            for &l in &p.zstd_levels {
                pruned_rows.push(pruned(chain(Compression::Zstd(l)).ddl(), shortcut.clone()));
            }
            pick = (Compression::Lz4, lz4);
            zstd_best = None;
        } else {
            let mut zstd = Vec::with_capacity(p.zstd_levels.len());
            for &l in &p.zstd_levels {
                let (bytes, _) = measure(chain(Compression::Zstd(l)))?;
                zstd.push((l, bytes));
                measured.push((Compression::Zstd(l), bytes));
            }
            pick = choose_compression(lz4, &zstd, p);
            zstd_best = zstd.iter().map(|&(_, b)| b).min();
        }
        families.push(Family {
            gcd,
            measured,
            pick,
            zstd_best,
        });
    }

    // GCD wins iff it saves gcd_min_saving over the best non-GCD chain.
    let plain = &families[0];
    let winner_gcd = families
        .get(1)
        .is_some_and(|g| g.pick.1 as f64 <= (1.0 - p.gcd_min_saving) * plain.pick.1 as f64);
    let win = families
        .iter()
        .find(|f| f.gcd == winner_gcd)
        .expect("the winning family was measured");
    let chain = Chain {
        gcd: win.gcd,
        compression: win.pick.0,
    };
    let bytes = win.pick.1;

    let mut rows = Vec::new();
    for f in &families {
        for &(c, b) in &f.measured {
            let codecs = Chain {
                gcd: f.gcd,
                compression: c,
            }
            .ddl();
            let is_pick = c == f.pick.0;
            let (outcome, evidence) = if is_pick && f.gcd == winner_gcd {
                ("chosen", "smallest after pruning".to_string())
            } else if is_pick && f.gcd {
                (
                    "rejected",
                    format!(
                        "GCD saves < {}% over no GCD ({b} vs {})",
                        p.gcd_min_saving * 100.0,
                        plain.pick.1
                    ),
                )
            } else if is_pick {
                (
                    "rejected",
                    format!("GCD saves ≥ {}%: {bytes} vs {b}", p.gcd_min_saving * 100.0),
                )
            } else {
                ("rejected", lost_to_pick(c, b, f, p))
            };
            rows.push(CodecCandidate {
                codecs,
                predicted_bytes: Some(b),
                outcome,
                evidence,
            });
        }
    }
    rows.sort_by_key(|r| r.predicted_bytes);
    rows.extend(pruned_rows);
    Ok(Decision {
        chain,
        bytes,
        candidates: rows,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    const UNIFORM: [u64; 256] = [1; 256];

    fn skewed() -> [u64; 256] {
        let mut h = [0u64; 256];
        h[0] = 900;
        for c in h.iter_mut().skip(1).take(100) {
            *c = 1;
        }
        h
    }

    #[test]
    fn bytes_needed_cases() {
        assert_eq!(bytes_needed(0, false), 1);
        assert_eq!(bytes_needed(255, false), 1);
        assert_eq!(bytes_needed(256, false), 2);
        assert_eq!(bytes_needed(127, true), 1);
        assert_eq!(bytes_needed(128, true), 2);
        assert_eq!(bytes_needed(u64::MAX as u128, false), 8);
    }

    #[test]
    fn gcd_survives_cases() {
        assert!(!gcd_survives(1, 1 << 40, false).0);
        assert!(gcd_survives(2, 1 << 32, false).0);
        assert!(!gcd_survives(2, (1 << 32) - 2, false).0);
        assert!(!gcd_survives(3, 3 << 40, false).0);
        assert!(gcd_survives(4, 1020, false).0);
        assert!(gcd_survives(256, 65_536, false).0);
        assert!(gcd_survives(2, 128, true).0);
    }

    #[test]
    fn entropy_and_shortcut() {
        assert!((byte_entropy(&UNIFORM) - 8.0).abs() < 1e-12);
        let mut one = [0u64; 256];
        one[7] = 100;
        assert_eq!(byte_entropy(&one), 0.0);
        assert!(incompressible(&UNIFORM, 0.9, 0.10).0);
        assert!(!incompressible(&UNIFORM, 0.3, 0.10).0);
        assert!(!incompressible(&skewed(), 0.9, 0.10).0);
    }

    #[test]
    fn choose_compression_cases() {
        let p = CompressionParams::default();
        let z = [(1, 1000), (3, 995), (9, 990)];
        assert_eq!(
            choose_compression(2000, &z, &p),
            (Compression::Zstd(3), 995)
        );
        assert_eq!(choose_compression(1050, &z, &p), (Compression::Lz4, 1050));
        assert_eq!(choose_compression(777, &[], &p), (Compression::Lz4, 777));
    }

    type Table = HashMap<Chain, (u64, [u64; 256])>;

    fn table(gcd: bool, lz4: u64, z: [u64; 3], h: [u64; 256], t: &mut Table) {
        let c = |compression| Chain { gcd, compression };
        t.insert(c(Compression::Lz4), (lz4, h));
        for (l, b) in [1, 3, 9].into_iter().zip(z) {
            t.insert(c(Compression::Zstd(l)), (b, h));
        }
    }

    fn run(
        t: &Table,
        gcd: Option<(u128, u128, bool)>,
        note: &str,
        ratio: f64,
        p: &CompressionParams,
    ) -> (Decision, Vec<Chain>) {
        let mut calls = Vec::new();
        let d = decide(
            |c| {
                calls.push(c);
                t.get(&c)
                    .copied()
                    .ok_or_else(|| format!("unmeasured {c:?}"))
            },
            gcd,
            note,
            ratio,
            p,
        )
        .unwrap();
        let chosen = d
            .candidates
            .iter()
            .filter(|r| r.outcome == "chosen")
            .count();
        assert_eq!(chosen, 1);
        (d, calls)
    }

    #[test]
    fn decide_gcd_pruned_by_byte_rule() {
        let mut t = Table::new();
        table(false, 2000, [1000, 990, 985], skewed(), &mut t);
        let p = CompressionParams::default();
        let (d, calls) = run(&t, Some((3, 3 << 40, false)), "", 0.1, &p);
        assert!(calls.iter().all(|c| !c.gcd));
        let g: Vec<_> = d.candidates.iter().filter(|r| r.codecs == "GCD").collect();
        assert_eq!(g.len(), 1);
        assert_eq!(g[0].outcome, "pruned");
        assert_eq!(g[0].predicted_bytes, None);
        assert_eq!(d.chain.compression, Compression::Zstd(3));
        assert_eq!(d.bytes, 990);
        assert_eq!(d.candidates.last().unwrap().outcome, "pruned");
    }

    #[test]
    fn decide_gcd_saves_too_little_or_enough() {
        let p = CompressionParams::default();
        let mut t = Table::new();
        table(false, 2000, [1000, 990, 985], skewed(), &mut t);
        table(true, 2000, [1000, 985, 980], skewed(), &mut t);
        let (d, _) = run(&t, Some((4, 1020, false)), "", 0.1, &p);
        assert!(!d.chain.gcd);
        assert!(d
            .candidates
            .iter()
            .any(|r| r.codecs == "GCD, ZSTD(3)" && r.outcome == "rejected"));

        let mut t = Table::new();
        table(false, 2000, [1000, 990, 985], skewed(), &mut t);
        table(true, 2000, [950, 940, 935], skewed(), &mut t);
        let (d, _) = run(&t, Some((4, 1020, false)), "", 0.1, &p);
        assert!(d.chain.gcd);
        assert_eq!(d.bytes, 940);
        assert_eq!(d.candidates[0].codecs, "GCD, ZSTD(9)");
        assert!(d
            .candidates
            .iter()
            .any(|r| r.codecs == "GCD, ZSTD(3)" && r.outcome == "chosen"));
    }

    #[test]
    fn decide_shortcut_prunes_zstd() {
        let mut t = Table::new();
        table(false, 1000, [900, 900, 900], UNIFORM, &mut t);
        let (d, calls) = run(&t, None, "", 0.9, &CompressionParams::default());
        assert_eq!(calls.len(), 1);
        assert_eq!(d.chain.compression, Compression::Lz4);
        assert_eq!(
            d.candidates
                .iter()
                .filter(|r| r.outcome == "pruned")
                .count(),
            3
        );
    }

    #[test]
    fn decide_without_levels_and_gcd_notes() {
        let mut t = Table::new();
        table(false, 1000, [900, 900, 900], skewed(), &mut t);
        let p = CompressionParams {
            zstd_levels: vec![],
            ..Default::default()
        };
        let (d, _) = run(&t, None, "gcd unknown", 0.1, &p);
        assert_eq!(d.chain.compression, Compression::Lz4);
        assert_eq!(d.candidates.len(), 2);
        assert_eq!(d.candidates[1].codecs, "GCD");
        assert_eq!(d.candidates[1].evidence, "gcd unknown");
        let (d, _) = run(&t, None, "", 0.1, &p);
        assert_eq!(d.candidates.len(), 1);
    }
}
