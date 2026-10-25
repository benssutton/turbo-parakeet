//! Analytic pruning, then emulation: transform (GCD, Delta, DoubleDelta) and compression
//! choices.

use crate::common::clickhouse_sizes::{Chain, Compression, Transform};
use crate::techniques::delta::DeltaStats;

/// The four keywords.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CompressionParams {
    pub zstd_levels: Vec<i32>,
    pub zstd_level_tolerance: f64,
    pub zstd_min_saving: f64,
    pub transform_min_saving: f64,
}

impl Default for CompressionParams {
    fn default() -> Self {
        CompressionParams {
            zstd_levels: vec![1, 3, 9],
            zstd_level_tolerance: 0.01,
            zstd_min_saving: 0.10,
            transform_min_saving: 0.01,
        }
    }
}

/// Bytes holding the magnitude `m` (plus a sign bit when `signed`); 0 → 1.
pub(crate) fn bytes_needed(m: u128, signed: bool) -> u32 {
    let bits = 128 - m.leading_zeros() + signed as u32;
    bits.div_ceil(8).max(1)
}

/// GCD is a candidate only if dividing by `g` frees a byte at the maximum
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

/// Delta survives iff the mean delta bytes < the mean value bytes.
pub(crate) fn delta_survives(s: &DeltaStats) -> (bool, String) {
    let (v, d) = (s.value_bytes_mean(), s.delta_bytes_mean());
    if d.is_nan() {
        return (false, "too few values".to_string());
    }
    (d < v, format!("delta {d:.2} B vs value {v:.2} B"))
}

/// DoubleDelta survives iff the mean double-delta bytes < min(mean value bytes, mean delta bytes).
pub(crate) fn double_delta_survives(s: &DeltaStats) -> (bool, String) {
    let (v, d, dd) = (
        s.value_bytes_mean(),
        s.delta_bytes_mean(),
        s.double_delta_bytes_mean(),
    );
    if dd.is_nan() {
        return (false, "too few values".to_string());
    }
    (
        dd < v.min(d),
        format!("double delta {dd:.2} B vs min(value {v:.2} B, delta {d:.2} B)"),
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

/// The near-incompressible shortcut: ZSTD cannot save `min_saving` over LZ4. Returns (fires,
/// evidence).
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

/// The lowest level within tolerance of the best ZSTD size, then ZSTD over LZ4
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

/// One considered chain family (one transform): its measured rows and its pick.
struct Family {
    transform: Transform,
    /// (compression, bytes) of every measured chain.
    measured: Vec<(Compression, u64)>,
    pick: (Compression, u64),
    /// Best ZSTD size among the measured levels.
    zstd_best: Option<u64>,
}

/// What `decide` may transform: the column's statistics, where known.
pub(crate) struct Transforms<'a> {
    /// Some((g, max_abs, signed)) when GCD applies (the byte-width rule).
    pub gcd: Option<(u128, u128, bool)>,
    /// Why GCD is missing when unknown; "" when GCD does not apply at all.
    pub gcd_note: &'a str,
    /// Delta statistics when Delta / DoubleDelta apply (integer stream, width 1/2/4/8).
    pub delta: Option<DeltaStats>,
}

fn transform_name(t: Transform) -> &'static str {
    match t {
        Transform::None => "",
        Transform::Gcd => "GCD",
        Transform::Delta => "Delta",
        Transform::DoubleDelta => "DoubleDelta",
    }
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

/// `measure(chain)` emulates one chain → (bytes, histogram of the uncompressed bytes).
/// Families: no transform plus each transform that survives its pruning rule; each is measured
/// (LZ4, then ZSTD levels unless the shortcut fires on its histogram). The winner is the
/// transform family with the smallest pick that saves `transform_min_saving` over the plain
/// pick (ties: GCD, Delta, DoubleDelta).
pub(crate) fn decide(
    mut measure: impl FnMut(Chain) -> Result<(u64, [u64; 256]), String>,
    tr: Transforms,
    distinct_ratio: f64,
    p: &CompressionParams,
) -> Result<Decision, String> {
    let mut pruned_rows = Vec::new();
    let flags = surviving_transforms(&tr, &mut pruned_rows);

    let mut families = Vec::new();
    for transform in flags {
        families.push(measure_family(
            transform,
            &mut measure,
            distinct_ratio,
            p,
            &mut pruned_rows,
        )?);
    }

    let plain = &families[0];
    let mut win = plain;
    for f in &families[1..] {
        if qualifies(f, plain, p) && (win.transform == Transform::None || f.pick.1 < win.pick.1) {
            win = f;
        }
    }
    let chain = Chain {
        transform: win.transform,
        compression: win.pick.0,
    };
    let bytes = win.pick.1;

    let mut rows = candidate_rows(&families, win, p);
    rows.sort_by_key(|r| r.predicted_bytes);
    rows.extend(pruned_rows);
    Ok(Decision {
        chain,
        bytes,
        candidates: rows,
    })
}

/// The transforms worth measuring: none, plus each of GCD / Delta / DoubleDelta that survives
/// its pruning rule; the others become `pruned` rows.
fn surviving_transforms(tr: &Transforms, pruned_rows: &mut Vec<CodecCandidate>) -> Vec<Transform> {
    let mut flags = vec![Transform::None];
    match tr.gcd {
        Some((g, max_abs, signed)) => {
            let (ok, evidence) = gcd_survives(g, max_abs, signed);
            if ok {
                flags.push(Transform::Gcd);
            } else {
                pruned_rows.push(pruned("GCD".to_string(), evidence));
            }
        }
        None if !tr.gcd_note.is_empty() => {
            pruned_rows.push(pruned("GCD".to_string(), tr.gcd_note.to_string()));
        }
        None => {}
    }
    if let Some(s) = &tr.delta {
        for (t, (ok, evidence)) in [
            (Transform::Delta, delta_survives(s)),
            (Transform::DoubleDelta, double_delta_survives(s)),
        ] {
            if ok {
                flags.push(t);
            } else {
                pruned_rows.push(pruned(transform_name(t).to_string(), evidence));
            }
        }
    }
    flags
}

/// Measures one transform's chains: LZ4, then the ZSTD levels unless the shortcut fires on the
/// LZ4 histogram (the skipped levels become `pruned` rows).
fn measure_family(
    transform: Transform,
    measure: &mut impl FnMut(Chain) -> Result<(u64, [u64; 256]), String>,
    distinct_ratio: f64,
    p: &CompressionParams,
    pruned_rows: &mut Vec<CodecCandidate>,
) -> Result<Family, String> {
    let chain = |compression| Chain {
        transform,
        compression,
    };
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
    Ok(Family {
        transform,
        measured,
        pick,
        zstd_best,
    })
}

/// Whether a family's pick saves `transform_min_saving` over the plain pick.
fn qualifies(f: &Family, plain: &Family, p: &CompressionParams) -> bool {
    f.pick.1 as f64 <= (1.0 - p.transform_min_saving) * plain.pick.1 as f64
}

/// One row per measured chain of every family, with the outcome against the winning family.
fn candidate_rows(families: &[Family], win: &Family, p: &CompressionParams) -> Vec<CodecCandidate> {
    let mut rows = Vec::new();
    for f in families {
        for &(c, b) in &f.measured {
            let codecs = Chain {
                transform: f.transform,
                compression: c,
            }
            .ddl();
            let (outcome, evidence) = chain_outcome(c, b, f, win, &families[0], p);
            rows.push(CodecCandidate {
                codecs,
                predicted_bytes: Some(b),
                outcome,
                evidence,
            });
        }
    }
    rows
}

/// Outcome and evidence of measured chain `(c, b)` of family `f`; `win` is the winning family,
/// `plain` the no-transform one.
fn chain_outcome(
    c: Compression,
    b: u64,
    f: &Family,
    win: &Family,
    plain: &Family,
    p: &CompressionParams,
) -> (&'static str, String) {
    let bytes = win.pick.1;
    let pct = p.transform_min_saving * 100.0;
    let is_pick = c == f.pick.0;
    if is_pick && f.transform == win.transform {
        ("chosen", "smallest after pruning".to_string())
    } else if is_pick && f.transform == Transform::None {
        (
            "rejected",
            format!(
                "{} saves ≥ {pct}% over no transform: {bytes} vs {b}",
                transform_name(win.transform)
            ),
        )
    } else if is_pick && !qualifies(f, plain, p) {
        (
            "rejected",
            format!(
                "{} saves < {pct}% over no transform ({b} vs {})",
                transform_name(f.transform),
                plain.pick.1
            ),
        )
    } else if is_pick {
        (
            "rejected",
            format!("{} smaller ({bytes} vs {b})", transform_name(win.transform)),
        )
    } else {
        ("rejected", lost_to_pick(c, b, f, p))
    }
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
        table_t(Transform::from_gcd(gcd), lz4, z, h, t);
    }

    fn table_t(transform: Transform, lz4: u64, z: [u64; 3], h: [u64; 256], t: &mut Table) {
        let c = |compression| Chain {
            transform,
            compression,
        };
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
        let tr = Transforms {
            gcd,
            gcd_note: note,
            delta: None,
        };
        run_tr(t, tr, ratio, p)
    }

    fn run_tr(
        t: &Table,
        tr: Transforms,
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
            tr,
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

    fn stats(n: u64, value: u64, delta: u64, double: u64) -> DeltaStats {
        DeltaStats {
            n_values: n,
            n_deltas: n - 1,
            n_double_deltas: n - 2,
            value_bytes: value * n,
            delta_bytes: delta * (n - 1),
            double_delta_bytes: double * (n - 2),
            ..Default::default()
        }
    }

    #[test]
    fn decide_gcd_pruned_by_byte_rule() {
        let mut t = Table::new();
        table(false, 2000, [1000, 990, 985], skewed(), &mut t);
        let p = CompressionParams::default();
        let (d, calls) = run(&t, Some((3, 3 << 40, false)), "", 0.1, &p);
        assert!(calls.iter().all(|c| c.transform != Transform::Gcd));
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
        assert!(d.chain.transform != Transform::Gcd);
        assert!(d
            .candidates
            .iter()
            .any(|r| r.codecs == "GCD, ZSTD(3)" && r.outcome == "rejected"));

        let mut t = Table::new();
        table(false, 2000, [1000, 990, 985], skewed(), &mut t);
        table(true, 2000, [950, 940, 935], skewed(), &mut t);
        let (d, _) = run(&t, Some((4, 1020, false)), "", 0.1, &p);
        assert!(d.chain.transform == Transform::Gcd);
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

    #[test]
    fn delta_survival_rules() {
        let s = stats(100, 8, 2, 1);
        assert!(delta_survives(&s).0);
        assert!(double_delta_survives(&s).0);
        let s = stats(100, 8, 8, 1);
        assert!(!delta_survives(&s).0);
        assert!(double_delta_survives(&s).0);
        let s = stats(100, 4, 1, 1);
        assert!(delta_survives(&s).0);
        let (ok, ev) = double_delta_survives(&s);
        assert!(!ok);
        assert_eq!(ev, "double delta 1.00 B vs min(value 4.00 B, delta 1.00 B)");
        assert_eq!(
            delta_survives(&stats(100, 8, 2, 1)).1,
            "delta 2.00 B vs value 8.00 B"
        );
        let one = DeltaStats {
            n_values: 1,
            value_bytes: 8,
            ..Default::default()
        };
        assert_eq!(delta_survives(&one), (false, "too few values".to_string()));
        assert_eq!(
            double_delta_survives(&stats(2, 8, 1, 1)),
            (false, "too few values".to_string())
        );
    }

    fn delta_run(plain: u64, delta: u64, double: u64, s: DeltaStats) -> (Decision, Vec<Chain>) {
        let mut t = Table::new();
        table_t(Transform::None, plain, [plain; 3], skewed(), &mut t);
        table_t(Transform::Delta, delta, [delta; 3], skewed(), &mut t);
        table_t(
            Transform::DoubleDelta,
            double,
            [double; 3],
            skewed(),
            &mut t,
        );
        let tr = Transforms {
            gcd: None,
            gcd_note: "",
            delta: Some(s),
        };
        run_tr(&t, tr, 0.1, &CompressionParams::default())
    }

    #[test]
    fn decide_delta_family_saving_threshold() {
        let (d, _) = delta_run(1000, 950, 0, stats(100, 8, 2, 8));
        assert_eq!(d.chain.transform, Transform::Delta);
        assert_eq!(d.bytes, 950);
        let (d, _) = delta_run(1000, 995, 0, stats(100, 8, 2, 8));
        assert_eq!(d.chain.transform, Transform::None);
        assert!(d.candidates.iter().any(|r| r.codecs == "Delta, LZ4"
            && r.outcome == "rejected"
            && r.evidence
                .contains("Delta saves < 1% over no transform (995 vs 1000)")));
    }

    #[test]
    fn decide_double_delta_beats_delta_when_smaller() {
        let (d, _) = delta_run(1000, 500, 100, stats(100, 8, 2, 1));
        assert_eq!(d.chain.transform, Transform::DoubleDelta);
        assert_eq!(d.bytes, 100);
        assert!(d.candidates.iter().any(|r| r.codecs == "Delta, LZ4"
            && r.outcome == "rejected"
            && r.evidence == "DoubleDelta smaller (100 vs 500)"));
        // a tie goes to the earlier family (Delta)
        let (d, _) = delta_run(1000, 100, 100, stats(100, 8, 2, 1));
        assert_eq!(d.chain.transform, Transform::Delta);
    }

    #[test]
    fn decide_prunes_delta_families_without_measuring() {
        let mut t = Table::new();
        table_t(Transform::None, 1000, [900, 900, 900], skewed(), &mut t);
        let tr = Transforms {
            gcd: None,
            gcd_note: "",
            delta: Some(stats(100, 4, 4, 4)),
        };
        let (d, calls) = run_tr(&t, tr, 0.1, &CompressionParams::default());
        assert!(calls.iter().all(|c| c.transform == Transform::None));
        for name in ["Delta", "DoubleDelta"] {
            let r = d.candidates.iter().find(|r| r.codecs == name).unwrap();
            assert_eq!(r.outcome, "pruned");
            assert_eq!(r.predicted_bytes, None);
        }
    }
}
