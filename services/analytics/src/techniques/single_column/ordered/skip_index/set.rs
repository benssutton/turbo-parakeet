//! The set(max_size) index: a window is skipped for `col = v` iff v is not among its values and
//! its stored set has not overflowed — a set of more than `max_size` entries (its distinct
//! values, plus NULL when the window holds one; 0 = unlimited) is stored empty and skips nothing.
//! The scan keeps per-id row counts for the window, O(1) per row entering and leaving.

use std::ops::Range;

use polars::prelude::*;
use rayon::prelude::*;

use super::{
    avoid, f64s, output, over_ranges, phi, population, u64s, windows, Common, Population,
    WindowSums, NULL,
};

/// One window's contents: rows per id, its distinct values and their population rows, its
/// nulls.
struct Window<'a> {
    pop: &'a Population,
    inside: Vec<u32>,
    present: u64,
    mass: u64,
    nulls: u64,
}

impl<'a> Window<'a> {
    fn new(pop: &'a Population) -> Self {
        Self {
            pop,
            inside: vec![0; pop.counts.len()],
            present: 0,
            mass: 0,
            nulls: 0,
        }
    }

    fn enter(&mut self, id: u32) {
        if id == NULL {
            self.nulls += 1;
            return;
        }
        let slot = &mut self.inside[id as usize];
        if *slot == 0 {
            self.present += 1;
            self.mass += self.pop.counts[id as usize];
        }
        *slot += 1;
    }

    fn leave(&mut self, id: u32) {
        if id == NULL {
            self.nulls -= 1;
            return;
        }
        let slot = &mut self.inside[id as usize];
        *slot -= 1;
        if *slot == 0 {
            self.present -= 1;
            self.mass -= self.pop.counts[id as usize];
        }
    }

    /// The stored set's entries: the distinct values, plus NULL.
    fn entries(&self) -> u64 {
        self.present + u64::from(self.nulls > 0)
    }
}

/// The windows starting at `starts` (each `w` rows), in one pass over rows
/// `starts.start .. starts.end + w − 1`.
pub(crate) fn scan(pop: &Population, w: usize, max_size: u64, starts: Range<usize>) -> WindowSums {
    if starts.is_empty() {
        return WindowSums::default();
    }
    let (nn, d) = (pop.non_null(), pop.d());
    let mut win = Window::new(pop);
    let mut sums = WindowSums::default();
    for r in starts.start..starts.end + w - 1 {
        win.enter(pop.ids[r]);
        if r + 1 < starts.start + w {
            continue;
        }
        let s = r + 1 - w; // the window is rows s..=r
        if s > starts.start {
            win.leave(pop.ids[s - 1]);
        }
        let entries = win.entries();
        sums.entries += u128::from(entries);
        sums.max_entries = sums.max_entries.max(entries);
        if max_size > 0 && entries > max_size {
            sums.overflowed += 1;
            sums.push(0, 0);
        } else {
            sums.push(nn - win.mass, d - win.present);
        }
    }
    sums
}

/// The set baseline under a uniformly random row order (spec §2.5).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Baseline {
    pub skip_rows: f64,
    pub skip_values: f64,
    pub window_distinct: f64,
    pub overflow: f64,
}

/// (count, multiplicity) classes of `counts`, ascending: entries with equal counts have equal
/// absence probabilities, so the baseline's sums run over classes.
fn classes(counts: impl IntoIterator<Item = u64>) -> Vec<(u64, u64)> {
    let mut sorted: Vec<u64> = counts.into_iter().collect();
    sorted.sort_unstable();
    let mut out: Vec<(u64, u64)> = Vec::new();
    for c in sorted {
        match out.last_mut() {
            Some((last, m)) if *last == c => *m += 1,
            _ => out.push((c, 1)),
        }
    }
    out
}

/// Expected entries present in a w-row window of a random order of n rows, one entry of count
/// `without` (if any) set aside.
fn mean_present(n: u64, w: u64, entries: &[(u64, u64)], without: Option<u64>) -> f64 {
    entries
        .iter()
        .map(|&(c, m)| {
            let m = m - u64::from(Some(c) == without);
            if m == 0 {
                0.0
            } else {
                m as f64 * (1.0 - avoid(n, w, c))
            }
        })
        .sum()
}

/// Exact std of the entries present: Σ q(1 − q) + Σ_{u≠u′} [avoid(n_u + n_u′) − q_u q_u′].
fn std_present(n: u64, w: u64, entries: &[(u64, u64)]) -> f64 {
    let q: Vec<f64> = entries.iter().map(|&(c, _)| avoid(n, w, c)).collect();
    let mut var = 0.0;
    for (i, &(c, m)) in entries.iter().enumerate() {
        var += m as f64 * q[i] * (1.0 - q[i]);
        for (j, &(c2, m2)) in entries.iter().enumerate() {
            let pairs = m * (m2 - u64::from(i == j));
            if pairs > 0 {
                var += pairs as f64 * (avoid(n, w, c + c2) - q[i] * q[j]);
            }
        }
    }
    var.max(0.0).sqrt()
}

/// Exact with `max_size` = 0; otherwise the entry count is approximated as normal (exact mean,
/// variance and conditional means; continuity-corrected).
pub(crate) fn baseline(pop: &Population, w: u64, max_size: u64) -> Baseline {
    let n = pop.n();
    let values = classes(pop.counts.iter().copied());
    let entries = classes(
        pop.counts
            .iter()
            .copied()
            .chain((pop.n_null > 0).then_some(pop.n_null)),
    );
    let window_distinct = mean_present(n, w, &entries, None);
    let sigma = if max_size == 0 {
        0.0
    } else {
        std_present(n, w, &entries)
    };
    // P(entry count ≤ max_size) for a mean entry count; 1 when unlimited.
    let fits = |mean: f64| {
        if max_size == 0 {
            return 1.0;
        }
        let x = max_size as f64 + 0.5 - mean;
        if sigma > 0.0 {
            phi(x / sigma)
        } else if x >= 0.0 {
            1.0
        } else {
            0.0
        }
    };
    let (mut rows, mut skip_values) = (0.0, 0.0);
    for &(c, m) in &values {
        let absent = avoid(n, w, c);
        let p = if max_size == 0 || absent == 0.0 {
            absent
        } else {
            absent * fits(mean_present(n - c, w, &entries, Some(c)))
        };
        rows += (c * m) as f64 * p;
        skip_values += m as f64 * p;
    }
    Baseline {
        skip_rows: rows / pop.non_null() as f64,
        skip_values: skip_values / pop.d() as f64,
        window_distinct,
        overflow: 1.0 - fits(window_distinct),
    }
}

/// One column's result row.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct SetStats {
    pub common: Common,
    pub window_distinct_mean: f64,
    pub window_distinct_max: u64,
    pub overflow_fraction: f64,
    pub expected_window_distinct: f64,
    pub expected_overflow_fraction: f64,
}

pub(crate) fn column(s: &Series, window_rows: u64, max_size: u64) -> PolarsResult<SetStats> {
    let pop = population(s)?;
    let (w, k) = windows(pop.n(), window_rows);
    let sums = over_ranges(k, w, pop.d(), |r| scan(&pop, w as usize, max_size, r));
    let b = baseline(&pop, w, max_size);
    let per_window = |x: f64| x / sums.windows as f64; // NaN without a window
    Ok(SetStats {
        common: Common::new(&pop, w, &sums, (b.skip_rows, b.skip_values)),
        window_distinct_mean: per_window(sums.entries as f64),
        window_distinct_max: sums.max_entries,
        overflow_fraction: per_window(sums.overflowed as f64),
        expected_window_distinct: b.window_distinct,
        expected_overflow_fraction: b.overflow,
    })
}

/// `column` + SetIndex's metrics per column; columns in parallel (and each column's window
/// ranges).
pub(crate) fn set_index_impl(
    inputs: &[Series],
    window_rows: u64,
    max_size: u64,
) -> PolarsResult<Series> {
    let rows: Vec<SetStats> = inputs
        .par_iter()
        .map(|s| column(s, window_rows, max_size))
        .collect::<PolarsResult<_>>()?;
    let commons: Vec<Common> = rows.iter().map(|r| r.common).collect();
    let mut fields = Common::fields(&commons);
    fields.extend([
        f64s(
            "window_distinct_mean",
            rows.iter().map(|r| r.window_distinct_mean),
        ),
        u64s(
            "window_distinct_max",
            rows.iter().map(|r| r.window_distinct_max),
        ),
        f64s(
            "overflow_fraction",
            rows.iter().map(|r| r.overflow_fraction),
        ),
        f64s(
            "expected_window_distinct",
            rows.iter().map(|r| r.expected_window_distinct),
        ),
        f64s(
            "expected_overflow_fraction",
            rows.iter().map(|r| r.expected_overflow_fraction),
        ),
    ]);
    output("set_index", inputs, fields)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pop(values: &[Option<i64>]) -> Population {
        population(&Series::new("x".into(), values)).unwrap()
    }

    fn whole(p: &Population, w: usize, max_size: u64) -> WindowSums {
        scan(p, w, max_size, 0..p.ids.len() - w + 1)
    }

    /// A A A B A A A C (spec §7).
    const TOY: [Option<i64>; 8] = [
        Some(0),
        Some(0),
        Some(0),
        Some(1),
        Some(0),
        Some(0),
        Some(0),
        Some(2),
    ];

    #[test]
    fn toy_known_answer() {
        let p = pop(&TOY);
        // windows 0–3 hold {A, B} (1 row of 8 outside, 1 value of 3); window 4 holds {A, C}
        let s = whole(&p, 4, 0);
        assert_eq!(
            (
                s.windows,
                s.rows.sum,
                s.values.sum,
                s.entries,
                s.max_entries,
                s.overflowed
            ),
            (5, 5, 5, 10, 2, 0)
        );
        let o = whole(&p, 4, 1);
        assert_eq!((o.rows.sum, o.values.sum, o.overflowed), (0, 0, 5));
    }

    #[test]
    fn null_counts_toward_max_size() {
        let p = pop(&[Some(1), None, Some(2), None]);
        assert_eq!(whole(&p, 2, 1).overflowed, 3);
        assert_eq!(whole(&p, 2, 2).overflowed, 0);
        let s = whole(&p, 2, 2);
        assert_eq!((s.rows.sum, s.values.sum), (3, 3));
    }

    #[test]
    fn merged_ranges_equal_one_scan_at_every_split() {
        let values: Vec<Option<i64>> = (0..40)
            .map(|i| (i % 7 != 3).then_some((i * 13 % 9) as i64))
            .collect();
        let p = pop(&values);
        for (w, max_size) in [(1, 0), (3, 0), (8, 0), (8, 4)] {
            let k = values.len() - w + 1;
            let all = scan(&p, w, max_size, 0..k);
            for i in 1..k {
                assert_eq!(
                    scan(&p, w, max_size, 0..i).merge(scan(&p, w, max_size, i..k)),
                    all,
                    "w {w}, max_size {max_size}, split {i}"
                );
            }
        }
    }

    /// Entry counts of every w-subset of `values` (each equally likely under a random order),
    /// keeping only subsets that `keep`.
    fn enumerate(
        values: &[Option<i64>],
        w: usize,
        keep: impl Fn(&[Option<i64>]) -> bool,
    ) -> Vec<f64> {
        let n = values.len();
        (0u32..1 << n)
            .filter(|mask| mask.count_ones() as usize == w)
            .filter_map(|mask| {
                let mut held: Vec<Option<i64>> = (0..n)
                    .filter(|i| mask >> i & 1 == 1)
                    .map(|i| values[i])
                    .collect();
                if !keep(&held) {
                    return None;
                }
                held.sort();
                held.dedup();
                Some(held.len() as f64)
            })
            .collect()
    }

    const SMALL: [Option<i64>; 9] = [
        Some(0),
        Some(0),
        Some(0),
        Some(1),
        Some(1),
        Some(2),
        None,
        None,
        Some(3),
    ];

    #[test]
    fn baseline_moments_match_enumeration() {
        let (n, w) = (SMALL.len() as u64, 3u64);
        let p = pop(&SMALL);
        let sizes = enumerate(&SMALL, 3, |_| true);
        let mean = sizes.iter().sum::<f64>() / sizes.len() as f64;
        let var = sizes.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / sizes.len() as f64;
        let entries = classes(p.counts.iter().copied().chain([p.n_null]));
        assert!((mean_present(n, w, &entries, None) - mean).abs() < 1e-12);
        assert!((std_present(n, w, &entries) - var.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn conditional_mean_matches_enumeration() {
        // value 1 (2 rows) absent; NULL also has 2 rows, so `without` removes one count-2 entry
        let (n, w) = (SMALL.len() as u64, 3u64);
        let p = pop(&SMALL);
        let sizes = enumerate(&SMALL, 3, |held| !held.contains(&Some(1)));
        let mean = sizes.iter().sum::<f64>() / sizes.len() as f64;
        let entries = classes(p.counts.iter().copied().chain([p.n_null]));
        assert!((mean_present(n - 2, w, &entries, Some(2)) - mean).abs() < 1e-12);
    }

    #[test]
    fn baseline_unlimited_hand_worked() {
        // 1 2 3 4, w = 2: each value is absent from half the pairs
        let b = baseline(&pop(&[Some(1), Some(2), Some(3), Some(4)]), 2, 0);
        assert!((b.skip_rows - 0.5).abs() < 1e-12 && (b.skip_values - 0.5).abs() < 1e-12);
        assert!((b.window_distinct - 2.0).abs() < 1e-12);
        assert_eq!(b.overflow, 0.0);
    }

    #[test]
    fn baseline_with_max_size_is_a_step_when_the_count_is_fixed() {
        // every window holds exactly 2 entries: σ = 0
        let p = pop(&[Some(1), Some(2), Some(3), Some(4)]);
        let b = baseline(&p, 2, 1);
        assert_eq!((b.skip_rows, b.overflow), (0.0, 1.0));
        let b = baseline(&p, 2, 2);
        assert!((b.skip_rows - 0.5).abs() < 1e-12);
        assert_eq!(b.overflow, 0.0);
    }

    #[test]
    fn impl_returns_one_struct_row_per_column() {
        let out = set_index_impl(&[Series::new("a".into(), &[1i64, 1, 2])], 2, 0).unwrap();
        let fields = out.struct_().unwrap().fields_as_series();
        let names: Vec<_> = fields.iter().map(|f| f.name().to_string()).collect();
        assert_eq!(names.len(), 17);
        assert_eq!(
            names[12..],
            [
                "window_distinct_mean",
                "window_distinct_max",
                "overflow_fraction",
                "expected_window_distinct",
                "expected_overflow_fraction"
            ]
        );
    }

    #[test]
    fn empty_start_range_reads_nothing() {
        let p = pop(&[Some(1), Some(2)]);
        assert_eq!(scan(&p, 4, 0, 2..2), WindowSums::default());
    }
}
