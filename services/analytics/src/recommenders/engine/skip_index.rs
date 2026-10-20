//! Skip-index decisions (spec 2026-10-20 §4): gate, set overflow, bloom fallback and economics,
//! granularity within a kind, choice across kinds.

use std::cmp::Ordering;

/// Candidates this close (in `skip_rows`) are equivalent: the coarser / smaller one wins.
const TIE: f64 = 0.02;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SkipParams {
    pub granularities: Vec<u32>,
    pub set_max_sizes: Vec<u64>,
    pub bloom_fprs: Vec<f64>,
    pub min_skip: f64,
    pub min_gain: f64,
    pub max_overflow: f64,
}

impl Default for SkipParams {
    fn default() -> Self {
        Self {
            granularities: vec![1, 2, 4, 8],
            set_max_sizes: vec![125, 250],
            bloom_fprs: vec![0.025],
            min_skip: 0.3,
            min_gain: 0.1,
            max_overflow: 0.1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Kind {
    MinMax,
    Set(u64),
    Bloom(f64),
}

impl Kind {
    /// "minmax", "set(125)", "bloom_filter(0.025)".
    pub(crate) fn ddl(&self) -> String {
        match self {
            Kind::MinMax => "minmax".to_string(),
            Kind::Set(n) => format!("set({n})"),
            Kind::Bloom(f) => format!("bloom_filter({f})"),
        }
    }

    /// Tie-break order: minmax < set < bloom.
    fn rank(&self) -> u8 {
        match self {
            Kind::MinMax => 0,
            Kind::Set(_) => 1,
            Kind::Bloom(_) => 2,
        }
    }
}

/// One evaluated candidate's measurements.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Measured {
    pub kind: Kind,
    pub granularity: u32,
    pub skip_rows: f64,
    pub skip_values: f64,
    pub expected_skip_rows: f64,
    pub bytes: Option<u64>,
    /// Set only: (window_distinct_mean, window_distinct_max, overflow_fraction).
    pub overflow: Option<(f64, u64, f64)>,
}

/// One `clickhouse_skip_indexes` row.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SkipCandidate {
    /// None for a pruned row without measurements (e.g. a granularity too large to evaluate).
    pub measured: Option<Measured>,
    pub kind: Kind,
    pub granularity: u32,
    pub outcome: &'static str, // "chosen" | "rejected" | "pruned"
    pub evidence: String,
}

fn row(m: Measured, outcome: &'static str, evidence: String) -> SkipCandidate {
    SkipCandidate {
        kind: m.kind,
        granularity: m.granularity,
        measured: Some(m),
        outcome,
        evidence,
    }
}

/// Thousands-separated by spaces: 9800 -> "9 800".
fn grouped(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push(' ');
        }
        out.push(c);
    }
    out
}

/// Step 1: Some(evidence) when the candidate fails the gate.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
fn gate(m: &Measured, p: &SkipParams) -> Option<String> {
    // NaN shares fail the gate.
    if !(m.skip_rows >= p.min_skip) {
        return Some(format!(
            "skip {:.2} < min_skip {:.2}",
            m.skip_rows, p.min_skip
        ));
    }
    let gain = m.skip_rows - m.expected_skip_rows;
    if !(gain >= p.min_gain) {
        return Some(format!(
            "gain {gain:.2} over random order < min_gain {:.2}",
            p.min_gain
        ));
    }
    None
}

/// Steps 1–2 for a minmax / set candidate: a finished row, or Err(measured) when it passes.
fn screen(m: Measured, p: &SkipParams) -> Result<SkipCandidate, Measured> {
    if let (Kind::Set(_), Some((mean, max, frac))) = (m.kind, m.overflow) {
        if frac >= p.max_overflow {
            let ev = format!(
                "window distinct mean {mean:.1}, max {max}; overflow {frac:.2} ≥ {:.2}",
                p.max_overflow
            );
            return Ok(row(m, "pruned", ev));
        }
    }
    match gate(&m, p) {
        Some(ev) => Ok(row(m, "rejected", ev)),
        None => Err(m),
    }
}

/// Step 3 for a bloom candidate.
fn screen_bloom(
    m: Measured,
    column_bytes: Option<u64>,
    p: &SkipParams,
) -> Result<SkipCandidate, Measured> {
    let Some(col) = column_bytes else {
        return Ok(row(m, "pruned", "no ClickHouse size".to_string()));
    };
    let Some(bytes) = m.bytes else {
        return Ok(row(m, "pruned", "no index size".to_string()));
    };
    if let Some(ev) = gate(&m, p) {
        return Ok(row(m, "rejected", ev));
    }
    let saved = m.skip_rows * col as f64;
    if saved <= bytes as f64 {
        let ev = format!(
            "saves {} B per query ≤ index {} B",
            grouped(saved as u64),
            grouped(bytes)
        );
        return Ok(row(m, "rejected", ev));
    }
    Err(m)
}

/// Spec §4 steps 1–5. `cheap`: the minmax / set measurements; `pruned`: rows already pruned by
/// the caller (ineligible kinds, too-large granularities); `bloom()` is called only when no cheap
/// candidate passed steps 1–2, and returns the bloom measurements; `column_bytes`: the column's
/// predicted ClickHouse compressed bytes (None → bloom rows pruned "no ClickHouse size").
/// Returns every row: the chosen one first, then the rest by `skip_rows` descending (pruned rows
/// without measurements last).
pub(crate) fn decide_skip(
    cheap: Vec<Measured>,
    pruned: Vec<SkipCandidate>,
    bloom: impl FnOnce() -> Vec<Measured>,
    column_bytes: Option<u64>,
    p: &SkipParams,
) -> Vec<SkipCandidate> {
    let mut rows = pruned;
    let mut passing = Vec::new();
    for m in cheap {
        match screen(m, p) {
            Ok(r) => rows.push(r),
            Err(m) => passing.push(m),
        }
    }
    if passing.is_empty() {
        for m in bloom() {
            match screen_bloom(m, column_bytes, p) {
                Ok(r) => rows.push(r),
                Err(m) => passing.push(m),
            }
        }
    }

    // Step 4: per kind, the largest granularity within TIE of the kind's best.
    let mut winners: Vec<Measured> = Vec::new();
    let mut done = vec![false; passing.len()];
    for i in 0..passing.len() {
        if done[i] {
            continue;
        }
        let kind = passing[i].kind;
        let group: Vec<usize> = (i..passing.len())
            .filter(|&j| passing[j].kind == kind)
            .collect();
        let best = group
            .iter()
            .map(|&j| passing[j].skip_rows)
            .fold(f64::MIN, f64::max);
        let win = *group
            .iter()
            .filter(|&&j| passing[j].skip_rows >= best - TIE)
            .max_by_key(|&&j| passing[j].granularity)
            .expect("group is not empty");
        for &j in &group {
            done[j] = true;
            if j != win {
                let ev = format!("coarser G {} within {TIE:.2}", passing[win].granularity);
                rows.push(row(passing[j].clone(), "rejected", ev));
            }
        }
        winners.push(passing[win].clone());
    }

    // Step 5: across kinds.
    if !winners.is_empty() {
        let best = winners.iter().map(|m| m.skip_rows).fold(f64::MIN, f64::max);
        let tied = |m: &Measured| m.skip_rows >= best - TIE;
        let pick = winners
            .iter()
            .enumerate()
            .filter(|(_, m)| tied(m))
            .min_by(|(_, a), (_, b)| {
                a.bytes
                    .unwrap_or(u64::MAX)
                    .cmp(&b.bytes.unwrap_or(u64::MAX))
                    .then(a.kind.rank().cmp(&b.kind.rank()))
            })
            .map(|(i, _)| i)
            .expect("a winner is within the tie of the best");
        let label = format!(
            "{} G{}",
            winners[pick].kind.ddl(),
            winners[pick].granularity
        );
        for (i, m) in winners.into_iter().enumerate() {
            if i == pick {
                let ev = format!(
                    "skip {:.2}, gain {:.2} over random order",
                    m.skip_rows,
                    m.skip_rows - m.expected_skip_rows
                );
                rows.push(row(m, "chosen", ev));
            } else if tied(&m) {
                rows.push(row(m, "rejected", format!("larger index than {label}")));
            } else {
                rows.push(row(m, "rejected", format!("lower skip than {label}")));
            }
        }
    }

    // Chosen first, then skip_rows descending; rows without measurements last.
    rows.sort_by(|a, b| {
        let skip = |r: &SkipCandidate| r.measured.as_ref().map(|m| m.skip_rows);
        (b.outcome == "chosen")
            .cmp(&(a.outcome == "chosen"))
            .then_with(|| match (skip(a), skip(b)) {
                (Some(x), Some(y)) => y.partial_cmp(&x).unwrap_or(Ordering::Equal),
                (Some(_), None) => Ordering::Less,
                (None, Some(_)) => Ordering::Greater,
                (None, None) => Ordering::Equal,
            })
    });
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(kind: Kind, g: u32, skip: f64, expected: f64, bytes: Option<u64>) -> Measured {
        Measured {
            kind,
            granularity: g,
            skip_rows: skip,
            skip_values: skip,
            expected_skip_rows: expected,
            bytes,
            overflow: None,
        }
    }

    fn set(n: u64, g: u32, skip: f64, bytes: u64, overflow: f64) -> Measured {
        Measured {
            overflow: Some((812.0, 8192, overflow)),
            ..m(Kind::Set(n), g, skip, 0.0, Some(bytes))
        }
    }

    fn run(cheap: Vec<Measured>, bloom: Vec<Measured>, col: Option<u64>) -> Vec<SkipCandidate> {
        decide_skip(cheap, vec![], || bloom, col, &SkipParams::default())
    }

    fn find(rows: &[SkipCandidate], kind: Kind, g: u32) -> &SkipCandidate {
        rows.iter()
            .find(|r| r.kind == kind && r.granularity == g)
            .unwrap()
    }

    fn chosen(rows: &[SkipCandidate]) -> Vec<&SkipCandidate> {
        rows.iter().filter(|r| r.outcome == "chosen").collect()
    }

    #[test]
    fn ddl_text() {
        assert_eq!(Kind::MinMax.ddl(), "minmax");
        assert_eq!(Kind::Set(125).ddl(), "set(125)");
        assert_eq!(Kind::Bloom(0.025).ddl(), "bloom_filter(0.025)");
    }

    #[test]
    fn gate_rejects_low_skip_and_low_gain() {
        let rows = run(
            vec![
                m(Kind::MinMax, 1, 0.12, 0.0, Some(10)),
                m(Kind::MinMax, 2, 0.5, 0.46, Some(10)),
            ],
            vec![],
            None,
        );
        assert!(chosen(&rows).is_empty());
        assert_eq!(
            find(&rows, Kind::MinMax, 1).evidence,
            "skip 0.12 < min_skip 0.30"
        );
        assert_eq!(
            find(&rows, Kind::MinMax, 2).evidence,
            "gain 0.04 over random order < min_gain 0.10"
        );
        assert!(rows.iter().all(|r| r.outcome == "rejected"));
    }

    #[test]
    fn nan_shares_fail_the_gate() {
        let rows = run(
            vec![
                m(Kind::MinMax, 1, f64::NAN, 0.0, Some(10)),
                m(Kind::MinMax, 2, 0.9, f64::NAN, Some(10)),
            ],
            vec![],
            None,
        );
        assert!(chosen(&rows).is_empty());
        assert!(rows.iter().all(|r| r.outcome == "rejected"));
    }

    #[test]
    fn overflowing_set_is_pruned_not_gated() {
        let rows = run(vec![set(125, 1, 0.01, 100, 0.97)], vec![], None);
        let r = &rows[0];
        assert_eq!(r.outcome, "pruned");
        assert_eq!(
            r.evidence,
            "window distinct mean 812.0, max 8192; overflow 0.97 ≥ 0.10"
        );
    }

    #[test]
    fn higher_skip_wins_across_kinds() {
        let rows = run(
            vec![
                m(Kind::MinMax, 1, 0.5, 0.0, Some(10)),
                set(125, 1, 0.9, 1000, 0.0),
            ],
            vec![],
            None,
        );
        let c = chosen(&rows);
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].kind, Kind::Set(125));
        assert_eq!(
            find(&rows, Kind::MinMax, 1).evidence,
            "lower skip than set(125) G1"
        );
    }

    #[test]
    fn tie_goes_to_smaller_index_then_minmax() {
        let rows = run(
            vec![
                m(Kind::MinMax, 1, 0.90, 0.0, Some(500)),
                set(125, 1, 0.91, 100, 0.0),
            ],
            vec![],
            None,
        );
        assert_eq!(chosen(&rows)[0].kind, Kind::Set(125));
        assert_eq!(
            find(&rows, Kind::MinMax, 1).evidence,
            "larger index than set(125) G1"
        );
        let rows = run(
            vec![
                m(Kind::MinMax, 1, 0.90, 0.0, Some(100)),
                set(125, 1, 0.91, 100, 0.0),
            ],
            vec![],
            None,
        );
        assert_eq!(chosen(&rows)[0].kind, Kind::MinMax);
    }

    #[test]
    fn granularity_prefers_coarsest_within_tie() {
        let rows = run(
            vec![
                m(Kind::MinMax, 1, 0.95, 0.0, None),
                m(Kind::MinMax, 4, 0.94, 0.0, None),
                m(Kind::MinMax, 8, 0.90, 0.0, None),
            ],
            vec![],
            None,
        );
        let c = chosen(&rows);
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].granularity, 4);
        assert_eq!(
            find(&rows, Kind::MinMax, 1).evidence,
            "coarser G 4 within 0.02"
        );
        assert_eq!(find(&rows, Kind::MinMax, 8).outcome, "rejected");
        assert_eq!(rows[0].granularity, 4);
    }

    #[test]
    fn bloom_not_called_when_cheap_passes() {
        let mut calls = 0;
        let rows = decide_skip(
            vec![m(Kind::MinMax, 1, 0.9, 0.0, None)],
            vec![],
            || {
                calls += 1;
                vec![]
            },
            Some(1_000_000),
            &SkipParams::default(),
        );
        assert_eq!(calls, 0);
        assert_eq!(chosen(&rows).len(), 1);
    }

    #[test]
    fn bloom_fallback_chosen_and_rejected_by_economics() {
        let cheap = vec![m(Kind::MinMax, 1, 0.1, 0.0, None)];
        let rows = run(
            cheap.clone(),
            vec![m(Kind::Bloom(0.025), 1, 0.8, 0.0, Some(9_800))],
            Some(1_000_000),
        );
        assert_eq!(chosen(&rows)[0].kind, Kind::Bloom(0.025));
        let rows = run(
            cheap,
            vec![m(Kind::Bloom(0.025), 1, 0.8, 0.0, Some(9_800))],
            Some(1_500),
        );
        assert!(chosen(&rows).is_empty());
        assert_eq!(
            find(&rows, Kind::Bloom(0.025), 1).evidence,
            "saves 1 200 B per query ≤ index 9 800 B"
        );
    }

    #[test]
    fn bloom_pruned_without_sizes() {
        let rows = run(
            vec![],
            vec![m(Kind::Bloom(0.025), 1, 0.8, 0.0, Some(10))],
            None,
        );
        assert_eq!(rows[0].outcome, "pruned");
        assert_eq!(rows[0].evidence, "no ClickHouse size");
        let rows = run(
            vec![],
            vec![m(Kind::Bloom(0.025), 1, 0.8, 0.0, None)],
            Some(100),
        );
        assert_eq!(rows[0].evidence, "no index size");
    }

    #[test]
    fn order_chosen_then_skip_desc_then_unmeasured() {
        let unmeasured = SkipCandidate {
            measured: None,
            kind: Kind::MinMax,
            granularity: 64,
            outcome: "pruned",
            evidence: "too large".into(),
        };
        let rows = decide_skip(
            vec![
                m(Kind::MinMax, 1, 0.2, 0.0, None),
                set(125, 1, 0.6, 10, 0.0),
                m(Kind::MinMax, 2, 0.5, 0.0, Some(5)),
                set(250, 1, 0.95, 10, 0.0),
            ],
            vec![unmeasured],
            Vec::new,
            None,
            &SkipParams::default(),
        );
        assert_eq!(rows[0].outcome, "chosen");
        assert_eq!(rows[0].kind, Kind::Set(250));
        let skips: Vec<f64> = rows[1..rows.len() - 1]
            .iter()
            .map(|r| r.measured.as_ref().unwrap().skip_rows)
            .collect();
        assert!(skips.windows(2).all(|w| w[0] >= w[1]));
        assert!(rows.last().unwrap().measured.is_none());
    }
}
