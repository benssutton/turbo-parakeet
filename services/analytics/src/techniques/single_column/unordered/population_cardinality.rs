//! Population cardinality of every column (spec 2026-10-24-cardinality-entropy-techniques
//! §3): Describe's frequency pass (counts, f1, f2, the 3-way split capture history), then
//! the Chao1 and Schnabel estimators. Both are always reported; Schnabel is NaN when
//! undefined (d/n ≥ 0.5 or no recapture).

use polars::prelude::*;
use rayon::prelude::*;

use crate::common::encode_series;
use crate::techniques::cardinality_estimators::{chao1, schnabel};
use crate::techniques::describe::frequencies::frequencies;

struct Row {
    n: u64,
    d: u64,
    f1: u64,
    f2: u64,
    chao: (f64, f64, f64),
    schnabel: (f64, f64, f64),
}

/// `col_name`, `n_non_null`, `n_distinct`, `f1`, `f2`, `chao1`, `chao1_low`, `chao1_high`,
/// `schnabel`, `schnabel_low`, `schnabel_high` per column.
pub(crate) fn population_cardinality_impl(inputs: &[Series], seed: u64) -> PolarsResult<Series> {
    let rows = inputs
        .par_iter()
        .map(|s| {
            let col = encode_series(s)?;
            let f = frequencies(&col, seed, None, None);
            let n = col.is_null.iter().filter(|&&null| !null).count() as u64;
            Ok(Row {
                n,
                d: f.n_unique,
                f1: f.f1,
                f2: f.f2,
                chao: chao1(f.n_unique, f.f1, f.f2),
                schnabel: schnabel(&f.capture_history, f.n_unique, n).unwrap_or((
                    f64::NAN,
                    f64::NAN,
                    f64::NAN,
                )),
            })
        })
        .collect::<PolarsResult<Vec<_>>>()?;
    let u = |name: &str, get: fn(&Row) -> u64| {
        UInt64Chunked::from_vec(name.into(), rows.iter().map(get).collect()).into_series()
    };
    let x = |name: &str, get: fn(&Row) -> f64| {
        Float64Chunked::from_vec(name.into(), rows.iter().map(get).collect()).into_series()
    };
    let series = [
        StringChunked::from_iter(inputs.iter().map(|s| s.name().as_str()))
            .into_series()
            .with_name("col_name".into()),
        u("n_non_null", |r| r.n),
        u("n_distinct", |r| r.d),
        u("f1", |r| r.f1),
        u("f2", |r| r.f2),
        x("chao1", |r| r.chao.0),
        x("chao1_low", |r| r.chao.1),
        x("chao1_high", |r| r.chao.2),
        x("schnabel", |r| r.schnabel.0),
        x("schnabel_low", |r| r.schnabel.1),
        x("schnabel_high", |r| r.schnabel.2),
    ];
    Ok(
        StructChunked::from_series("population_cardinality".into(), inputs.len(), series.iter())?
            .into_series(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(s: Series, seed: u64) -> DataFrame {
        population_cardinality_impl(&[s], seed)
            .unwrap()
            .into_frame()
            .unnest(["population_cardinality"])
            .unwrap()
    }

    #[test]
    fn counts_and_chao1_by_hand() {
        // a×3, b×2, c×1, d×1, null×2: d = 4, f1 = 2, f2 = 1, n = 7
        let s = Series::new(
            "x".into(),
            [
                Some("a"),
                Some("a"),
                Some("a"),
                Some("b"),
                Some("b"),
                Some("c"),
                Some("d"),
                None,
                None,
            ],
        );
        let df = frame(s, 0);
        let u = |c: &str| df.column(c).unwrap().u64().unwrap().get(0).unwrap();
        assert_eq!(
            (u("n_non_null"), u("n_distinct"), u("f1"), u("f2")),
            (7, 4, 2, 1)
        );
        let chao = df.column("chao1").unwrap().f64().unwrap().get(0).unwrap();
        assert!((chao - (4.0 + 2.0 * 1.0 / (2.0 * 2.0))).abs() < 1e-12); // d + f1(f1−1)/(2(f2+1))
                                                                         // d/n = 4/7 ≥ 0.5: Schnabel undefined
        assert!(df
            .column("schnabel")
            .unwrap()
            .f64()
            .unwrap()
            .get(0)
            .unwrap()
            .is_nan());
    }

    #[test]
    fn matches_describe_frequencies() {
        let v: Vec<i64> = (0..200_000i64).map(|i| (i * 7919) % 30_000).collect();
        let s = Series::new("x".into(), v);
        let f = frequencies(&encode_series(&s).unwrap(), 5, None, None);
        let df = frame(s, 5);
        let u = |c: &str| df.column(c).unwrap().u64().unwrap().get(0).unwrap();
        assert_eq!(
            (u("n_distinct"), u("f1"), u("f2")),
            (f.n_unique, f.f1, f.f2)
        );
        let want = schnabel(&f.capture_history, f.n_unique, 200_000).unwrap().0;
        let got = df
            .column("schnabel")
            .unwrap()
            .f64()
            .unwrap()
            .get(0)
            .unwrap();
        assert_eq!(got, want);
    }
}
