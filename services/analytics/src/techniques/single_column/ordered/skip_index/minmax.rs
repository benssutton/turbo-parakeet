//! The minmax index: a window is skipped for `col = v` iff v < lo or v > hi, its non-null
//! extremes; an all-null window is skipped for every v. Two monotonic deques of row positions
//! give each window's extremes in O(1) amortised per row.

use std::collections::VecDeque;
use std::ops::Range;

use polars::prelude::*;
use rayon::prelude::*;

use super::{
    avoid, output, over_ranges, population, windows, Common, Population, WindowSums, NULL,
};

/// The windows starting at `starts` (each `w` rows), in one pass over rows
/// `starts.start .. starts.end + w − 1`.
pub(crate) fn scan(pop: &Population, w: usize, starts: Range<usize>) -> WindowSums {
    let (ids, nn, d) = (&pop.ids, pop.non_null(), pop.d());
    let mut sums = WindowSums::default();
    let (mut lows, mut highs) = (VecDeque::<usize>::new(), VecDeque::<usize>::new());
    for r in starts.start..starts.end + w - 1 {
        let id = ids[r];
        if id != NULL {
            while lows.back().is_some_and(|&b| ids[b] >= id) {
                lows.pop_back();
            }
            lows.push_back(r);
            while highs.back().is_some_and(|&b| ids[b] <= id) {
                highs.pop_back();
            }
            highs.push_back(r);
        }
        if r + 1 < starts.start + w {
            continue;
        }
        let s = r + 1 - w; // the window is rows s..=r
        while lows.front().is_some_and(|&f| f < s) {
            lows.pop_front();
        }
        while highs.front().is_some_and(|&f| f < s) {
            highs.pop_front();
        }
        match (lows.front(), highs.front()) {
            (Some(&lo), Some(&hi)) => {
                let (lo, hi) = (ids[lo] as usize, ids[hi] as usize);
                sums.push(
                    nn - (pop.cum[hi + 1] - pop.cum[lo]),
                    d - (hi - lo + 1) as u64,
                );
            }
            _ => sums.push(nn, d),
        }
    }
    sums
}

/// (expected_skip_rows, expected_skip_values) under a uniformly random row order: a window
/// skips v iff it holds no row ≤ v or no row ≥ v; both events include the all-null window,
/// which is counted once.
pub(crate) fn baseline(pop: &Population, w: u64) -> (f64, f64) {
    let (n, nn) = (pop.n(), pop.non_null());
    let all_null = avoid(n, w, nn);
    let (mut rows, mut values) = (0.0, 0.0);
    for (v, &c) in pop.counts.iter().enumerate() {
        let p = avoid(n, w, pop.cum[v + 1]) + avoid(n, w, nn - pop.cum[v]) - all_null;
        rows += c as f64 * p;
        values += p;
    }
    (rows / nn as f64, values / pop.d() as f64)
}

/// One column's metrics.
pub(crate) fn column(s: &Series, window_rows: u64) -> PolarsResult<Common> {
    let pop = population(s)?;
    let (w, k) = windows(pop.n(), window_rows);
    let sums = over_ranges(k, w, 0, |r| scan(&pop, w as usize, r));
    Ok(Common::new(&pop, w, &sums, baseline(&pop, w)))
}

/// `column` + MinMaxIndex's metrics per column; columns in parallel (and each column's window
/// ranges).
pub(crate) fn minmax_index_impl(inputs: &[Series], window_rows: u64) -> PolarsResult<Series> {
    let rows: Vec<Common> = inputs
        .par_iter()
        .map(|s| column(s, window_rows))
        .collect::<PolarsResult<_>>()?;
    output("minmax_index", inputs, Common::fields(&rows))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pop(values: &[Option<i64>]) -> Population {
        population(&Series::new("x".into(), values)).unwrap()
    }

    fn whole(p: &Population, w: usize) -> WindowSums {
        scan(p, w, 0..p.ids.len() - w + 1)
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
        let s = whole(&p, 4);
        // windows 0–3 cover [A, B] (1 row of 8 outside, 1 value of 3); window 4 covers [A, C]
        assert_eq!(
            (
                s.windows,
                s.rows.sum,
                s.rows.sum_sq,
                s.values.sum,
                s.values.sum_sq
            ),
            (5, 4, 4, 4, 4)
        );
        let c = Common::new(&p, 4, &s, baseline(&p, 4));
        assert!((c.skip_rows_mean - 0.1).abs() < 1e-15);
        assert!((c.skip_rows_std - 0.05).abs() < 1e-15);
        assert!((c.skip_values_mean - 4.0 / 15.0).abs() < 1e-15);
        assert!((c.skip_values_std - 2.0 / 15.0).abs() < 1e-15);
    }

    #[test]
    fn all_null_window_skips_everything() {
        let p = pop(&[Some(1), None, None, Some(2)]);
        // windows {1}, {}, {2}: rows numerators 1, 2, 1; values 1, 2, 1
        let s = whole(&p, 2);
        assert_eq!((s.rows.sum, s.values.sum), (4, 4));
    }

    #[test]
    fn merged_ranges_equal_one_scan_at_every_split() {
        let values: Vec<Option<i64>> = (0..40)
            .map(|i| (i % 7 != 3).then_some((i * 13 % 9) as i64))
            .collect();
        let p = pop(&values);
        for w in [1, 3, 8] {
            let k = values.len() - w + 1;
            let all = scan(&p, w, 0..k);
            for i in 1..k {
                assert_eq!(
                    scan(&p, w, 0..i).merge(scan(&p, w, i..k)),
                    all,
                    "w {w}, split {i}"
                );
            }
        }
    }

    #[test]
    fn baseline_hand_worked() {
        // 1 2 3 4, w = 2: 6 equally likely pairs, 8 of 24 (pair, query) combinations skipped
        let (rows, values) = baseline(&pop(&[Some(1), Some(2), Some(3), Some(4)]), 2);
        assert!((rows - 1.0 / 3.0).abs() < 1e-12);
        assert!((values - 1.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn short_and_all_null_columns() {
        let c = column(&Series::new("x".into(), &[3i64, 1, 2]), 8).unwrap();
        assert_eq!(
            (
                c.window_rows,
                c.n_windows,
                c.skip_rows_mean,
                c.expected_skip_rows
            ),
            (3, 1, 0.0, 0.0)
        );
        let c = column(&Series::new("x".into(), &[None::<i64>, None, None]), 2).unwrap();
        assert_eq!((c.n_distinct, c.n_windows), (0, 2));
        assert!(c.skip_rows_mean.is_nan() && c.skip_values_std.is_nan());
        assert!(c.expected_skip_values.is_nan());
    }

    #[test]
    fn impl_returns_one_struct_row_per_column() {
        let out = minmax_index_impl(
            &[
                Series::new("a".into(), &[1i64, 1, 2]),
                Series::new("b".into(), &["x", "y", "y"]),
            ],
            2,
        )
        .unwrap();
        assert_eq!(out.len(), 2);
        let fields = out.struct_().unwrap().fields_as_series();
        let names: Vec<_> = fields.iter().map(|f| f.name().to_string()).collect();
        assert_eq!(
            names,
            [
                "column",
                "n_rows",
                "n_non_null",
                "n_distinct",
                "window_rows",
                "n_windows",
                "skip_rows_mean",
                "skip_rows_std",
                "skip_values_mean",
                "skip_values_std",
                "expected_skip_rows",
                "expected_skip_values"
            ]
        );
    }
}
