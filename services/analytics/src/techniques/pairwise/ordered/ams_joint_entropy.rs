//! Joint entropy of column pairs and triplets, estimated by one AMS (CCM) sketch each:
//! O(`num_samples`) memory per combination, independent of the number of distinct values.

use polars::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;

use crate::common::*;
use crate::techniques::ams_entropy::{
    combine, entropy_struct, keyed_columns, name_key, row_count, sketch_seed, AmsEntropy, PAIR,
    TRIPLE,
};

fn name_map(inputs: &[Series]) -> HashMap<String, usize> {
    inputs
        .iter()
        .enumerate()
        .map(|(i, s)| (s.name().to_string(), i))
        .collect()
}

/// H(A,B) per pair (every pair when `kwargs.pairs` is None): `col_a`, `col_b`, `entropy`.
pub(crate) fn ams_pairwise_joint_entropy_impl(
    inputs: &[Series],
    kwargs: PairwiseKwargs,
    num_samples: usize,
    seed: u64,
) -> PolarsResult<Series> {
    let n = inputs.len();
    if n < 2 {
        return entropy_struct(
            "pairwise_entropy",
            vec![("col_a", vec![]), ("col_b", vec![])],
            vec![],
        );
    }
    row_count(inputs, "ams_pairwise_joint_entropy")?;
    let pairs: Vec<(usize, usize)> = match &kwargs.pairs {
        Some(raw) => resolve_pairs(raw, &name_map(inputs))?,
        None => (0..n)
            .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
            .collect(),
    };
    let keys = keyed_columns(inputs, pairs.iter().flat_map(|&(i, j)| [i, j]))?;
    let entropy: Vec<f64> = pairs
        .par_iter()
        .map(|&(i, j)| {
            let id = combine(name_key(inputs[i].name()), name_key(inputs[j].name()));
            let mut sketch = AmsEntropy::new(num_samples, sketch_seed(seed, PAIR, id));
            for (&a, &b) in keys[i].iter().zip(&keys[j]) {
                sketch.update(combine(a, b));
            }
            sketch.estimate()
        })
        .collect();
    let name = |k: usize| inputs[k].name().as_str();
    entropy_struct(
        "pairwise_entropy",
        vec![
            ("col_a", pairs.iter().map(|p| name(p.0)).collect()),
            ("col_b", pairs.iter().map(|p| name(p.1)).collect()),
        ],
        entropy,
    )
}

/// H(A,B,C) per triplet (every triplet when `kwargs.triplets` is None):
/// `col_a`, `col_b`, `col_c`, `entropy`.
pub(crate) fn ams_threeway_joint_entropy_impl(
    inputs: &[Series],
    kwargs: ThreewayKwargs,
    num_samples: usize,
    seed: u64,
) -> PolarsResult<Series> {
    if inputs.len() < 3 {
        return Err(PolarsError::ComputeError(
            "ams_threeway_joint_entropy requires at least three columns".into(),
        ));
    }
    row_count(inputs, "ams_threeway_joint_entropy")?;
    let n = inputs.len();
    let triplets: Vec<(usize, usize, usize)> = match &kwargs.triplets {
        Some(raw) => resolve_triplets(raw, &name_map(inputs))?,
        None => (0..n)
            .flat_map(|i| ((i + 1)..n).flat_map(move |j| ((j + 1)..n).map(move |k| (i, j, k))))
            .collect(),
    };
    let keys = keyed_columns(inputs, triplets.iter().flat_map(|&(i, j, k)| [i, j, k]))?;
    let entropy: Vec<f64> = triplets
        .par_iter()
        .map(|&(i, j, k)| {
            let id = combine(
                name_key(inputs[i].name()),
                combine(name_key(inputs[j].name()), name_key(inputs[k].name())),
            );
            let mut sketch = AmsEntropy::new(num_samples, sketch_seed(seed, TRIPLE, id));
            for ((&a, &b), &c) in keys[i].iter().zip(&keys[j]).zip(&keys[k]) {
                sketch.update(combine(a, combine(b, c)));
            }
            sketch.estimate()
        })
        .collect();
    let name = |k: usize| inputs[k].name().as_str();
    entropy_struct(
        "threeway_entropy",
        vec![
            ("col_a", triplets.iter().map(|t| name(t.0)).collect()),
            ("col_b", triplets.iter().map(|t| name(t.1)).collect()),
            ("col_c", triplets.iter().map(|t| name(t.2)).collect()),
        ],
        entropy,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(n: i64) -> Vec<Series> {
        vec![
            Series::new("hi".into(), (0..n).map(|i| i / 10).collect::<Vec<_>>()),
            Series::new("lo".into(), (0..n).map(|i| i % 10).collect::<Vec<_>>()),
            Series::new("k".into(), vec![7i64; n as usize]),
            Series::new("k2".into(), vec![1i64; n as usize]),
        ]
    }

    fn column<'a>(df: &'a DataFrame, name: &str) -> Vec<&'a str> {
        df.column(name)
            .unwrap()
            .str()
            .unwrap()
            .into_no_null_iter()
            .collect()
    }

    #[test]
    fn composite_key_pair_is_log2_n_and_constant_pair_is_zero() {
        let out =
            ams_pairwise_joint_entropy_impl(&frame(1000), PairwiseKwargs { pairs: None }, 256, 0)
                .unwrap();
        let df = out.into_frame().unnest(["pairwise_entropy"]).unwrap();
        // (hi,lo) (hi,k) (hi,k2) (lo,k) (lo,k2) (k,k2)
        assert_eq!(column(&df, "col_a"), ["hi", "hi", "hi", "lo", "lo", "k"]);
        assert_eq!(column(&df, "col_b"), ["lo", "k", "k2", "k", "k2", "k2"]);
        let h: Vec<f64> = df
            .column("entropy")
            .unwrap()
            .f64()
            .unwrap()
            .into_no_null_iter()
            .collect();
        assert!((h[0] - 1000f64.log2()).abs() < 1e-12);
        assert_eq!(h[5], 0.0);
    }

    #[test]
    fn explicit_pairs_keep_their_order() {
        let pairs = PairwiseKwargs {
            pairs: Some(vec![vec!["lo".into(), "hi".into()]]),
        };
        let out = ams_pairwise_joint_entropy_impl(&frame(100), pairs, 64, 0).unwrap();
        let df = out.into_frame().unnest(["pairwise_entropy"]).unwrap();
        assert_eq!(column(&df, "col_a"), ["lo"]);
        assert_eq!(column(&df, "col_b"), ["hi"]);
    }

    #[test]
    fn one_column_gives_no_pairs() {
        let out =
            ams_pairwise_joint_entropy_impl(&frame(10)[..1], PairwiseKwargs { pairs: None }, 16, 0)
                .unwrap();
        assert_eq!(out.len(), 0);
    }

    #[test]
    fn composite_key_triplet_is_log2_n() {
        let triplets = ThreewayKwargs {
            triplets: Some(vec![vec!["hi".into(), "lo".into(), "k".into()]]),
        };
        let out = ams_threeway_joint_entropy_impl(&frame(1000), triplets, 256, 0).unwrap();
        let df = out.into_frame().unnest(["threeway_entropy"]).unwrap();
        assert_eq!(column(&df, "col_c"), ["k"]);
        let h = df.column("entropy").unwrap().f64().unwrap().get(0).unwrap();
        assert!((h - 1000f64.log2()).abs() < 1e-12);
    }

    #[test]
    fn threeway_needs_three_columns() {
        let r = ams_threeway_joint_entropy_impl(
            &frame(10)[..2],
            ThreewayKwargs { triplets: None },
            16,
            0,
        );
        assert!(r.is_err());
    }

    fn entropy_of(df: &DataFrame, row: usize) -> f64 {
        df.column("entropy")
            .unwrap()
            .f64()
            .unwrap()
            .get(row)
            .unwrap()
    }

    #[test]
    fn pair_estimate_does_not_depend_on_other_requested_pairs() {
        let mut cols = frame(1000);
        cols.push(Series::new(
            "x".into(),
            (0..1000i64).map(|i| (i * 7919) % 13).collect::<Vec<_>>(),
        ));
        let run = |pairs: Vec<Vec<String>>| {
            let out = ams_pairwise_joint_entropy_impl(
                &cols,
                PairwiseKwargs { pairs: Some(pairs) },
                64,
                0,
            )
            .unwrap();
            out.into_frame().unnest(["pairwise_entropy"]).unwrap()
        };
        let lo_x = || vec!["lo".to_string(), "x".to_string()];
        let alone = run(vec![lo_x()]);
        let second = run(vec![vec!["hi".into(), "x".into()], lo_x()]);
        assert_eq!(entropy_of(&alone, 0), entropy_of(&second, 1));
    }

    #[test]
    fn single_zero_row_column_gives_no_pairs() {
        let empty = vec![Series::new("a".into(), Vec::<i64>::new())];
        let out =
            ams_pairwise_joint_entropy_impl(&empty, PairwiseKwargs { pairs: None }, 16, 0).unwrap();
        assert_eq!(out.len(), 0);
    }

    #[test]
    fn nulls_are_a_category_in_pairs() {
        // se = 1.44/sqrt(4096) = 0.023, so 0.1 is ~4.4 se.
        let n = 1000i64;
        let half = Series::new(
            "h".into(),
            (0..n)
                .map(|i| (i % 2 == 0).then_some(1i64))
                .collect::<Vec<_>>(),
        );
        let k = Series::new("k".into(), vec![7i64; n as usize]);
        let out =
            ams_pairwise_joint_entropy_impl(&[half, k], PairwiseKwargs { pairs: None }, 4096, 0)
                .unwrap();
        let df = out.into_frame().unnest(["pairwise_entropy"]).unwrap();
        assert!(
            (entropy_of(&df, 0) - 1.0).abs() < 0.1,
            "{}",
            entropy_of(&df, 0)
        );
    }

    #[test]
    fn null_position_distinguishes_triplet_rows() {
        // Rows (null,1,2) and (1,null,2) alternate: two categories, 1 bit.
        let n = 1000usize;
        let a = Series::new(
            "a".into(),
            (0..n)
                .map(|i| (i % 2 == 1).then_some(1i64))
                .collect::<Vec<_>>(),
        );
        let b = Series::new(
            "b".into(),
            (0..n)
                .map(|i| (i % 2 == 0).then_some(1i64))
                .collect::<Vec<_>>(),
        );
        let c = Series::new("c".into(), vec![2i64; n]);
        let out =
            ams_threeway_joint_entropy_impl(&[a, b, c], ThreewayKwargs { triplets: None }, 4096, 0)
                .unwrap();
        let df = out.into_frame().unnest(["threeway_entropy"]).unwrap();
        assert!(
            (entropy_of(&df, 0) - 1.0).abs() < 0.1,
            "{}",
            entropy_of(&df, 0)
        );
    }
}
