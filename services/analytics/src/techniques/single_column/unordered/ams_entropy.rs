//! Shannon entropy of a stream in O(`num_samples`) memory: the Chakrabarti–Cormode–McGregor
//! estimator (TALG 6(3), 2010). AMS sampling plus a backup sample, so a dominant item cannot
//! swamp the estimate. Spec: docs/superpowers/specs/2026-10-11-ams-entropy-design.md.

use foldhash::fast::FixedState;
use polars::prelude::*;
use rayon::prelude::*;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

use crate::common::*;

/// Misra–Gries counters for the dominant item: p̂ is low by at most (1 − p)/k (CCM Lemma 2.3),
/// which biases H by about 1/k relative.
const MG_COUNTERS: usize = 256;

/// SplitMix64's finaliser on `z + γ`: a bijection on u64.
pub(crate) fn mix64(z: u64) -> u64 {
    let mut z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Key of a null row: null is its own category, as in the exact kernels. (It collides with
/// the one value whose `mix64` equals it.)
const NULL_KEY: u64 = 0x6E75_6C6C_6E75_6C6C;

/// Seed domains, so a marginal and a combination with the same index draw different labels.
pub(crate) const MARGINAL: u64 = 1;
pub(crate) const PAIR: u64 = 2;
pub(crate) const TRIPLE: u64 = 3;

/// One key for two keys, order-sensitive; nest it for triplets.
pub(crate) fn combine(a: u64, b: u64) -> u64 {
    mix64(a ^ mix64(b))
}

/// The seed of the sketch of one column or combination, from the user seed, a domain tag and
/// the identity of the columns: independent of thread schedule and of the other requests.
pub(crate) fn sketch_seed(seed: u64, domain: u64, identity: u64) -> u64 {
    mix64(mix64(mix64(seed) ^ domain) ^ identity)
}

/// A column name as a key: deterministic across runs and platforms.
pub(crate) fn name_key(name: &str) -> u64 {
    name.bytes().fold(0, |h, b| mix64(h ^ b as u64))
}

/// Row count shared by every input column; errors on no columns, zero rows or ragged input.
pub(crate) fn row_count(inputs: &[Series], what: &str) -> PolarsResult<usize> {
    let Some(first) = inputs.first() else {
        return Err(PolarsError::ComputeError(
            format!("{what} requires at least one column").into(),
        ));
    };
    let r = first.len();
    if r == 0 {
        return Err(PolarsError::ComputeError(
            "Cannot calculate entropy on empty columns".into(),
        ));
    }
    if let Some((idx, s)) = inputs.iter().enumerate().find(|(_, s)| s.len() != r) {
        return Err(PolarsError::ShapeMismatch(
            format!(
                "All columns must have the same length: column {idx} has length {} but expected {r}",
                s.len()
            )
            .into(),
        ));
    }
    Ok(r)
}

/// Per-row sketch keys of the columns in `needed` (value-stable `encode_series` keys, null
/// folded in); other slots stay empty.
pub(crate) fn keyed_columns(
    inputs: &[Series],
    needed: impl IntoIterator<Item = usize>,
) -> PolarsResult<Vec<Vec<u64>>> {
    let needed: HashSet<usize> = needed.into_iter().collect();
    let cache = build_column_cache_par(inputs, &needed)?;
    Ok(cache
        .into_par_iter()
        .map(|mut col| {
            for (v, &null) in col.values.iter_mut().zip(&col.is_null) {
                *v = if null { NULL_KEY } else { mix64(*v) };
            }
            col.values
        })
        .collect())
}

/// The kernels' output: struct `name` of string key columns, then `entropy`.
pub(crate) fn entropy_struct(
    name: &str,
    keys: Vec<(&str, Vec<&str>)>,
    entropy: Vec<f64>,
) -> PolarsResult<Series> {
    let n = entropy.len();
    let mut fields: Vec<Series> = keys
        .into_iter()
        .map(|(field, values)| {
            StringChunked::from_iter(values)
                .into_series()
                .with_name(field.into())
        })
        .collect();
    fields.push(Float64Chunked::from_vec("entropy".into(), entropy).into_series());
    Ok(StructChunked::from_series(name.into(), n, fields.iter())?.into_series())
}

/// H(col) of every column, one sketch each: `col_name`, `entropy`.
pub(crate) fn ams_marginal_entropy_impl(
    inputs: &[Series],
    num_samples: usize,
    seed: u64,
) -> PolarsResult<Series> {
    row_count(inputs, "ams_marginal_entropy")?;
    let keys = keyed_columns(inputs, 0..inputs.len())?;
    let entropy: Vec<f64> = keys
        .par_iter()
        .zip(inputs)
        .map(|(col, s)| {
            let id = name_key(s.name());
            let mut sketch = AmsEntropy::new(num_samples, sketch_seed(seed, MARGINAL, id));
            for &k in col {
                sketch.update(k);
            }
            sketch.estimate()
        })
        .collect();
    let names = inputs.iter().map(|s| s.name().as_str()).collect();
    entropy_struct("marginal_entropy", vec![("col_name", names)], entropy)
}

/// SplitMix64 generator.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        mix64(self.0)
    }

    /// Uniform in [0, 1).
    fn unit0(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform in (0, 1].
    fn unit1(&mut self) -> f64 {
        ((self.next_u64() >> 11) + 1) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Rows before the next one whose label is below `p`: geometric, failures before success.
    fn skip(&mut self, p: f64) -> u64 {
        if p >= 1.0 {
            return 0;
        }
        if p <= 0.0 {
            return u64::MAX;
        }
        (self.unit1().ln() / (-p).ln_1p()) as u64 // `as` saturates
    }
}

/// Misra–Gries frequent-items summary with `MG_COUNTERS` counters.
#[derive(Default)]
struct MisraGries(HashMap<u64, u64, FixedState>);

impl MisraGries {
    fn update(&mut self, x: u64) {
        if let Some(c) = self.0.get_mut(&x) {
            *c += 1;
        } else if self.0.len() < MG_COUNTERS {
            self.0.insert(x, 1);
        } else {
            // Amortised O(1): total decrements never exceed total increments.
            self.0.retain(|_, c| {
                *c -= 1;
                *c > 0
            });
        }
    }

    /// The item with the largest counter and that counter, a lower bound on its count.
    fn top(&self) -> (u64, u64) {
        self.0
            .iter()
            .max_by_key(|&(k, c)| (*c, *k))
            .map_or((0, 0), |(k, c)| (*k, *c))
    }
}

/// CCM Algorithm 2: H from m, the dominant item and each estimator's
/// (primary key, primary count r0, backup count r1).
fn ccm_estimate(
    m: u64,
    mg: &MisraGries,
    samples: impl Iterator<Item = (u64, u64, Option<u64>)>,
) -> f64 {
    if m == 0 {
        return 0.0;
    }
    let mf = m as f64;
    let lambda = |x: u64| {
        if x == 0 {
            0.0
        } else {
            x as f64 * (mf / x as f64).log2()
        }
    };
    let step = |r: u64| lambda(r) - lambda(r - 1); // r ≥ 1 always
    let (top, count) = mg.top();
    let p = count as f64 / mf;
    let (mut sum, mut n) = (0.0, 0usize);
    if p <= 0.5 {
        for (_, r0, _) in samples {
            sum += step(r0);
            n += 1;
        }
        return sum / n as f64;
    }
    // The dominant item's term is computed from p̂; the rest is estimated from samples of the
    // stream without it: the primary when it is not the dominant item, else the backup.
    for (s0, r0, r1) in samples {
        n += 1;
        if s0 != top {
            sum += step(r0);
        } else if let Some(r1) = r1 {
            sum += step(r1);
        }
    }
    (1.0 - p) * sum / n as f64 + p * (1.0 / p).log2()
}

/// A sampled row: its key, its label and the key's tracked count before that row.
#[derive(Clone, Copy)]
struct Sample {
    key: u64,
    label: f64,
    offset: u64,
}

/// One AMS estimator: the smallest-label row (primary), the smallest-label row of another
/// token (backup), and the next events of its two label clocks (see `AmsEntropy`).
#[derive(Clone, Copy)]
struct Estimator {
    primary: Option<Sample>,
    backup: Option<Sample>,
    /// A clock: the row of the next label below t0.
    next_a: u64,
    /// B clock: the next label in [t0, t1) falls on the `next_b`-th row (1-based) since the
    /// primary key's tracking began whose token is not the primary key.
    next_b: u64,
}

impl Estimator {
    fn t0(&self) -> f64 {
        self.primary.map_or(1.0, |p| p.label)
    }

    fn t1(&self) -> f64 {
        self.backup.map_or(1.0, |b| b.label)
    }
}

/// A key that some estimator holds. `count` and `start` cover the rows since tracking last
/// began (the entry is dropped when `holders` reaches 0): from row `start` on, `count` rows
/// held this key and the rest held other keys.
struct Tracked {
    count: u64,
    start: u64,
    holders: u32,
    /// B clocks of the estimators whose primary is this key: (`next_b`, estimator), smallest
    /// first; entries not matching the estimator's `next_b` or primary key are stale.
    b: BinaryHeap<Reverse<(u64, u32)>>,
    /// Row at which `b` is next checked (an entry of `AmsEntropy::checks`), `u64::MAX` if none.
    check_row: u64,
}

impl Tracked {
    /// Rows of other keys from `start` up to, but not including, `row` (before `row` is counted).
    fn others(&self, row: u64) -> u64 {
        row - self.start - self.count
    }
}

fn b_valid(estimators: &[Estimator], key: u64, target: u64, e: u32) -> bool {
    let est = &estimators[e as usize];
    est.next_b == target && est.primary.is_some_and(|p| p.key == key)
}

/// The smallest valid (target, estimator) of `t.b` (the B heap of `key`), left at the top
/// after popping stale entries.
fn min_valid_b(estimators: &[Estimator], key: u64, t: &mut Tracked) -> Option<(u64, u32)> {
    while let Some(&Reverse((target, e))) = t.b.peek() {
        if b_valid(estimators, key, target, e) {
            return Some((target, e));
        }
        t.b.pop();
    }
    None
}

/// Streaming Shannon-entropy sketch (bits) with `num_samples` estimators.
///
/// Each estimator's per-row label is simulated by two independent geometric clocks: A (label
/// below t0, relevant on every row; success probability t0 per row) and B (label in
/// [t0, t1), relevant only on rows whose token is not the primary's; success probability
/// (t1 − t0)/(1 − t0) per such row, given no A event). A events are processed before B events
/// on the same row, and every event redraws both clocks. Rows of the primary's token never
/// advance the B clock, so a dominant token costs no events.
pub(crate) struct AmsEntropy {
    estimators: Vec<Estimator>,
    /// A clocks: (row, estimator), smallest first; entries not matching `next_a` are stale.
    a_events: BinaryHeap<Reverse<(u64, u32)>>,
    /// (row, key): when to check that key's B heap; valid while `tracked[key].check == row`.
    checks: BinaryHeap<Reverse<(u64, u64)>>,
    tracked: HashMap<u64, Tracked, FixedState>,
    mg: MisraGries,
    rng: Rng,
    m: u64,
    #[cfg(test)]
    fired: u64,
    /// Valid key checks processed (test builds): at most one wasted per row plus one per event.
    #[cfg(test)]
    checked: u64,
}

impl AmsEntropy {
    pub(crate) fn new(num_samples: usize, seed: u64) -> Self {
        assert!(
            num_samples >= 1 && num_samples <= u32::MAX as usize,
            "num_samples must be in 1..=u32::MAX, got {num_samples}"
        );
        let blank = Estimator {
            primary: None,
            backup: None,
            next_a: 0,
            next_b: u64::MAX,
        };
        Self {
            estimators: vec![blank; num_samples],
            a_events: (0..num_samples as u32).map(|e| Reverse((0, e))).collect(),
            checks: BinaryHeap::new(),
            tracked: HashMap::default(),
            mg: MisraGries::default(),
            rng: Rng(seed),
            m: 0,
            #[cfg(test)]
            fired: 0,
            #[cfg(test)]
            checked: 0,
        }
    }

    pub(crate) fn update(&mut self, x: u64) {
        let row = self.m;
        self.m += 1;
        self.mg.update(x);
        while let Some(&Reverse((at, e))) = self.a_events.peek() {
            if at > row {
                break;
            }
            self.a_events.pop();
            if self.estimators[e as usize].next_a == row {
                self.a_event(e as usize, row, x);
            }
        }
        while let Some(&Reverse((at, key))) = self.checks.peek() {
            if at > row {
                break;
            }
            self.checks.pop();
            if self.tracked.get(&key).is_some_and(|t| t.check_row == row) {
                #[cfg(test)]
                {
                    self.checked += 1;
                }
                self.check(key, row, x);
            }
        }
        if let Some(t) = self.tracked.get_mut(&x) {
            t.count += 1;
        }
    }

    /// Entropy estimate in bits; 0 for an empty stream.
    pub(crate) fn estimate(&self) -> f64 {
        let r = |s: &Sample| self.tracked[&s.key].count - s.offset;
        ccm_estimate(
            self.m,
            &self.mg,
            self.estimators.iter().filter_map(|e| {
                let p = e.primary?;
                Some((p.key, r(&p), e.backup.as_ref().map(r)))
            }),
        )
    }

    /// A label below t0 on this row, whose token is `x`: a new primary.
    fn a_event(&mut self, e: usize, row: u64, x: u64) {
        #[cfg(test)]
        {
            self.fired += 1;
        }
        let est = self.estimators[e];
        let label = self.rng.unit0() * est.t0();
        match est.primary {
            None => {
                let offset = self.hold(x, row);
                self.estimators[e].primary = Some(Sample {
                    key: x,
                    label,
                    offset,
                });
            }
            Some(p) if p.key == x => {
                let offset = self.tracked[&x].count;
                self.estimators[e].primary = Some(Sample { label, offset, ..p });
            }
            Some(p) => {
                if let Some(b) = est.backup {
                    self.release(b.key);
                }
                let offset = self.hold(x, row);
                self.estimators[e].backup = Some(p);
                self.estimators[e].primary = Some(Sample {
                    key: x,
                    label,
                    offset,
                });
            }
        }
        self.schedule(e, row, x);
    }

    /// A label in [t0, t1) on this row, whose token `x` is not the primary's: a new backup.
    fn b_event(&mut self, e: usize, row: u64, x: u64) {
        #[cfg(test)]
        {
            self.fired += 1;
        }
        let est = self.estimators[e];
        let (t0, t1) = (est.t0(), est.t1());
        // May round up to t1: harmless, the B clock keeps its rate (one extra event).
        let label = t0 + self.rng.unit0() * (t1 - t0);
        if let Some(b) = est.backup {
            self.release(b.key);
        }
        let offset = self.hold(x, row);
        self.estimators[e].backup = Some(Sample {
            key: x,
            label,
            offset,
        });
        self.schedule(e, row, x);
    }

    /// Fire the B events due on this row among the estimators whose primary is `key`, then
    /// schedule the key's next check.
    fn check(&mut self, key: u64, row: u64, x: u64) {
        if x != key {
            let due = self.tracked[&key].others(row) + 1;
            while let Some(e) = self.pop_due(key, due) {
                self.b_event(e, row, x);
            }
        }
        self.refresh_check(key, row, x);
    }

    /// Pop the next valid B entry of `key` if its target is `due`.
    fn pop_due(&mut self, key: u64, due: u64) -> Option<usize> {
        let t = self.tracked.get_mut(&key)?;
        let (target, e) = min_valid_b(&self.estimators, key, t)?;
        debug_assert!(target >= due, "missed B event");
        if target != due {
            return None;
        }
        t.b.pop();
        Some(e as usize)
    }

    /// Point `key`'s check at the earliest row its smallest valid B target can fire: every row
    /// after this one (token `x`) holding another key. Wasted checks are paid for by rows of
    /// `key` in between, so there are at most m of them plus one per event.
    fn refresh_check(&mut self, key: u64, row: u64, x: u64) {
        let Some(t) = self.tracked.get_mut(&key) else {
            return;
        };
        // Invariant: after row R, every valid B target of `key` exceeds the number of non-`key`
        // rows through R.
        let min = min_valid_b(&self.estimators, key, t).map(|(target, _)| target);
        let at = min.map_or(u64::MAX, |target| {
            let after = t.others(row) + u64::from(x != key);
            // target == after: another B event is due on this row, during `check` (whose
            // `check` row this already is, so nothing is pushed).
            debug_assert!(target >= after, "missed B event");
            row.saturating_add(target - after)
        });
        if at != t.check_row {
            t.check_row = at;
            if at != u64::MAX {
                self.checks.push(Reverse((at, key)));
            }
        }
    }

    /// Start (or share) tracking `key` on `row`, before it is counted; its count so far is the
    /// new sample's offset.
    fn hold(&mut self, key: u64, row: u64) -> u64 {
        let t = self.tracked.entry(key).or_insert_with(|| Tracked {
            count: 0,
            start: row,
            holders: 0,
            b: BinaryHeap::new(),
            check_row: u64::MAX,
        });
        t.holders += 1;
        t.count
    }

    /// Drop a hold. With no holders left, no estimator has `key` as its primary, so the key's
    /// B heap holds only stale entries and goes with it.
    fn release(&mut self, key: u64) {
        if let Some(t) = self.tracked.get_mut(&key) {
            t.holders -= 1;
            if t.holders == 0 {
                self.tracked.remove(&key);
            }
        }
    }

    /// Redraw both clocks of estimator `e` after its event on `row` (token `x`).
    fn schedule(&mut self, e: usize, row: u64, x: u64) {
        let est = self.estimators[e];
        let p = est.primary.expect("an event leaves a primary");
        let (t0, t1) = (est.t0(), est.t1());
        let next_a = row.saturating_add(1).saturating_add(self.rng.skip(t0));
        self.estimators[e].next_a = next_a;
        if next_a != u64::MAX {
            self.a_events.push(Reverse((next_a, e as u32)));
        }
        // No backup: t1 = 1, so the B clock fires on the first row of another token.
        let skip = self.rng.skip((t1 - t0) / (1.0 - t0));
        let t = self.tracked.get_mut(&p.key).expect("primaries are tracked");
        let after = t.others(row) + u64::from(x != p.key);
        let next_b = after.saturating_add(1).saturating_add(skip);
        self.estimators[e].next_b = next_b;
        if next_b != u64::MAX {
            t.b.push(Reverse((next_b, e as u32)));
            self.refresh_check(p.key, row, x);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sketch(keys: &[u64], num_samples: usize, seed: u64) -> f64 {
        let mut s = AmsEntropy::new(num_samples, seed);
        for &k in keys {
            s.update(k);
        }
        s.estimate()
    }

    #[test]
    fn all_unique_stream_is_log2_m() {
        let keys: Vec<u64> = (0..1000).collect();
        let h = sketch(&keys, 256, 0);
        assert!((h - 1000f64.log2()).abs() < 1e-12, "{h}");
    }

    #[test]
    fn constant_stream_is_exactly_zero() {
        assert_eq!(sketch(&[7u64; 1000], 256, 0), 0.0);
    }

    #[test]
    fn empty_and_single_row_streams_are_zero() {
        assert_eq!(sketch(&[], 16, 0), 0.0);
        assert_eq!(sketch(&[42], 16, 0), 0.0);
    }

    #[test]
    fn same_seed_same_estimate() {
        let keys: Vec<u64> = (0..5000u64).map(|i| mix64(i) % 97).collect();
        assert_eq!(sketch(&keys, 128, 9), sketch(&keys, 128, 9));
        assert_ne!(sketch(&keys, 128, 9), sketch(&keys, 128, 10));
    }

    /// (events fired, key checks processed, rows) of a run with `num_samples` estimators.
    fn cost(keys: impl Iterator<Item = u64>, num_samples: usize) -> (u64, u64, u64) {
        let mut s = AmsEntropy::new(num_samples, 0);
        for k in keys {
            s.update(k);
        }
        eprintln!("cost: fired {} checked {} rows {}", s.fired, s.checked, s.m);
        (s.fired, s.checked, s.m)
    }

    fn fired(keys: impl Iterator<Item = u64>, num_samples: usize) -> u64 {
        cost(keys, num_samples).0
    }

    #[test]
    fn constant_stream_does_not_fire_every_estimator_every_row() {
        // A backup-less estimator's B clock counts only rows of other tokens, so it never
        // fires; the A clock fires on new minima: about 64 × (1 + ln 100K) ≈ 800 events.
        let fired = fired(std::iter::repeat_n(3, 100_000), 64);
        assert!(fired < 64 * 30, "fired {fired} events");
    }

    #[test]
    fn near_constant_stream_with_an_outlier_is_cheap() {
        // Labels in [t0, t1) on rows of the primary's token are no-ops; they must not be events.
        let keys = std::iter::repeat_n(0, 1000)
            .chain([1])
            .chain(std::iter::repeat_n(0, 100_000));
        let (fired, checked, rows) = cost(keys, 64);
        assert!(fired < 64 * 30, "fired {fired} events");
        assert!(
            checked <= rows + fired,
            "{checked} checks, {rows} rows, {fired} events"
        );
    }

    #[test]
    fn sparse_tail_stream_is_cheap() {
        let keys = (0..100_000u64).map(|i| {
            if mix64(i).is_multiple_of(1000) {
                1 + i
            } else {
                0
            }
        });
        let (fired, checked, rows) = cost(keys, 64);
        assert!(fired < 64 * 60, "fired {fired} events");
        assert!(
            checked <= rows + fired,
            "{checked} checks, {rows} rows, {fired} events"
        );
    }

    fn exact_entropy(keys: &[u64]) -> f64 {
        let mut counts: HashMap<u64, u64> = HashMap::new();
        for &k in keys {
            *counts.entry(k).or_default() += 1;
        }
        let m = keys.len() as f64;
        counts
            .values()
            .map(|&c| c as f64 / m * (m / c as f64).log2())
            .sum()
    }

    /// Pareto-rounded tokens: P(token >= k) = k^-1.1, a heavy head (~53% are 1) and long tail.
    fn zipf(n: usize, seed: u64) -> Vec<u64> {
        let mut rng = Rng(seed);
        (0..n)
            .map(|_| rng.unit1().powf(-1.0 / 1.1) as u64)
            .collect()
    }

    /// 95% one value, the rest spread over 1,000 others: exercises the backup sample.
    fn skewed(n: usize, seed: u64) -> Vec<u64> {
        let mut rng = Rng(seed);
        (0..n)
            .map(|_| {
                if rng.unit0() < 0.95 {
                    0
                } else {
                    1 + rng.next_u64() % 1000
                }
            })
            .collect()
    }

    fn mean_and_se(xs: &[f64]) -> (f64, f64) {
        let n = xs.len() as f64;
        let mean = xs.iter().sum::<f64>() / n;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        (mean, (var / n).sqrt())
    }

    /// CCM Algorithm 1 as written: a fresh label for every estimator on every row, O(c) per row.
    fn naive(keys: &[u64], c: usize, seed: u64) -> f64 {
        let mut rng = Rng(seed);
        let mut mg = MisraGries::default();
        // (key, label, r) per estimator; label 1.0 marks an empty slot.
        let mut p = vec![(0u64, 1.0f64, 0u64); c];
        let mut b = vec![(0u64, 1.0f64, 0u64); c];
        for &x in keys {
            mg.update(x);
            for (p, b) in p.iter_mut().zip(b.iter_mut()) {
                let t = rng.unit0();
                if t < p.1 {
                    if p.1 < 1.0 && x != p.0 {
                        *b = *p;
                    }
                    *p = (x, t, 1);
                } else if x == p.0 {
                    p.2 += 1;
                } else if t < b.1 {
                    *b = (x, t, 1);
                } else if b.1 < 1.0 && x == b.0 {
                    b.2 += 1;
                }
            }
        }
        ccm_estimate(
            keys.len() as u64,
            &mg,
            p.iter()
                .zip(&b)
                .map(|(p, b)| (p.0, p.2, (b.1 < 1.0).then_some(b.2))),
        )
    }

    #[test]
    fn mean_estimate_is_within_one_percent_of_exact() {
        for keys in [zipf(10_000, 1), skewed(10_000, 2)] {
            let h = exact_entropy(&keys);
            let est: Vec<f64> = (0..100).map(|seed| sketch(&keys, 4096, seed)).collect();
            let (mean, se) = mean_and_se(&est);
            eprintln!("accuracy: mean {mean} se {se} exact {h}");
            assert!(
                (mean - h).abs() <= 0.01 * h,
                "mean {mean} (se {se}) vs exact {h}"
            );
        }
    }

    #[test]
    fn two_clock_matches_naive_algorithm_1() {
        for keys in [zipf(2_000, 3), skewed(2_000, 4)] {
            let fast: Vec<f64> = (0..400).map(|s| sketch(&keys, 64, s)).collect();
            let slow: Vec<f64> = (0..400).map(|s| naive(&keys, 64, 10_000 + s)).collect();
            let ((m1, se1), (m2, se2)) = (mean_and_se(&fast), mean_and_se(&slow));
            // se = sd / sqrt(n), so the ratio of standard errors is the ratio of deviations.
            let ratio = se1 / se2;
            eprintln!(
                "equivalence: two-clock {m1} +- {se1}, naive {m2} +- {se2}, sd ratio {ratio}"
            );
            assert!(
                (m1 - m2).abs() <= 4.0 * (se1 * se1 + se2 * se2).sqrt(),
                "two-clock {m1} +- {se1}, naive {m2} +- {se2}"
            );
            // A 4,000-seed release run measured sd ratios of 0.99 (zipf) and 0.98 (skewed); 400 seeds keep debug runs fast.
            assert!(
                (0.8..=1.25).contains(&ratio),
                "sd ratio {ratio}: two-clock sd {}, naive sd {}",
                se1 * (fast.len() as f64).sqrt(),
                se2 * (slow.len() as f64).sqrt()
            );
        }
    }

    #[test]
    fn two_clock_scheme_matches_exact_on_small_streams() {
        // Dominant tokens (p > ½) exercise A and B events, backups and the p̂ > ½ path; the
        // estimator is unbiased, so the mean over seeds is within 4 standard errors of H.
        let streams: [&[u64]; 2] = [
            &[0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 3, 0, 1, 0, 0],
            &[9, 9, 9, 9, 4, 9, 9, 4, 9, 7, 9, 9, 9, 8, 9, 9, 9, 9, 9, 6],
        ];
        for keys in streams {
            let n = 2000;
            let hs: Vec<f64> = (0..n).map(|seed| sketch(keys, 8, seed)).collect();
            let mean = hs.iter().sum::<f64>() / n as f64;
            let var = hs.iter().map(|h| (h - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
            let se = (var / n as f64).sqrt();
            let exact = exact_entropy(keys);
            assert!(
                (mean - exact).abs() < 4.0 * se,
                "mean {mean} vs exact {exact} (se {se})"
            );
        }
    }

    #[test]
    fn misra_gries_finds_the_dominant_item() {
        let mut mg = MisraGries::default();
        for i in 0..10_000u64 {
            mg.update(if i % 4 == 0 { i } else { 5 });
        }
        let (top, count) = mg.top();
        assert_eq!(top, 5);
        // Lower bound, short by at most (m − m_top) / k = 2500 / 256.
        assert!((7_490..=7_500).contains(&count), "{count}");
    }

    fn entropies(out: Series, name: &str) -> Vec<f64> {
        let df = out.into_frame().unnest([name]).unwrap();
        df.column("entropy")
            .unwrap()
            .f64()
            .unwrap()
            .into_no_null_iter()
            .collect()
    }

    #[test]
    fn marginal_kernel_unique_constant_and_null_columns() {
        let n = 1000i64;
        let u = Series::new("u".into(), (0..n).collect::<Vec<_>>());
        let k = Series::new("k".into(), vec![7i64; n as usize]);
        let nulls = Series::new("z".into(), vec![None::<i64>; n as usize]);
        let half = Series::new(
            "h".into(),
            (0..n)
                .map(|i| (i % 2 == 0).then_some(1i64))
                .collect::<Vec<_>>(),
        );
        let out = ams_marginal_entropy_impl(&[u, k, nulls, half], 4096, 0).unwrap();
        let h = entropies(out, "marginal_entropy");
        assert!((h[0] - (n as f64).log2()).abs() < 1e-12);
        assert_eq!(h[1], 0.0);
        assert_eq!(h[2], 0.0); // all null: one category
                               // Per-estimator sd ~ log2(e) = 1.44 bits, so se = 1.44/sqrt(4096) = 0.023; 0.1 is ~4.4 se.
                               // null vs 1, half each:
        assert!((h[3] - 1.0).abs() < 0.1, "{}", h[3]);
    }

    #[test]
    fn marginal_estimate_does_not_depend_on_other_columns() {
        let x = Series::new(
            "x".into(),
            (0..2000i64).map(|i| (i * 7919) % 13).collect::<Vec<_>>(),
        );
        let y = Series::new("y".into(), (0..2000i64).map(|i| i % 5).collect::<Vec<_>>());
        let alone = entropies(
            ams_marginal_entropy_impl(std::slice::from_ref(&x), 64, 0).unwrap(),
            "marginal_entropy",
        );
        let with = entropies(
            ams_marginal_entropy_impl(&[y, x], 64, 0).unwrap(),
            "marginal_entropy",
        );
        assert_eq!(alone[0], with[1]);
    }

    #[test]
    fn marginal_kernel_rejects_empty_and_ragged_input() {
        assert!(ams_marginal_entropy_impl(&[], 16, 0).is_err());
        let empty = Series::new("a".into(), Vec::<i64>::new());
        assert!(ams_marginal_entropy_impl(&[empty], 16, 0).is_err());
        let a = Series::new("a".into(), &[1i64, 2, 3]);
        let b = Series::new("b".into(), &[1i64, 2]);
        assert!(ams_marginal_entropy_impl(&[a, b], 16, 0).is_err());
    }
}
