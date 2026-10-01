//! Describe's conclusions (spec 2026-10-01 §4, §5), shared by one-shot and streaming:
//! the whole-number range behind `ordinal` and the class rule. analytics/describe/base.py
//! keeps the reference implementation.

use polars::prelude::DataType as PT;

use crate::cardinality_estimators::{pick_estimate, Count, Estimate};
use crate::describe::{FloatStats, Profile, StringStats};

/// (min, max) when every non-null value is a whole number: integers, Decimal with
/// scale 0, floats with no fraction / NaN / infinity, and strings that are all integers
/// without leading zeros. `numeric`: the level's numeric extremes (integer, decimal and
/// float dtypes; None otherwise). Temporal dtypes never qualify.
/// "Integers" means Int/UInt 8–64 (Int128/UInt128 are ineligible upstream). `n_non_null`:
/// non-null values at this level. The integer-string range is dropped when a value
/// overflows the parse (Python drops it past 38 digits; the class cannot differ, since
/// such values exceed 2·n_values).
pub(crate) fn whole_range(
    dtype: &PT,
    n_non_null: u64,
    numeric: Option<(f64, f64)>,
    floats: Option<&FloatStats>,
    strings: Option<&StringStats>,
) -> Option<(f64, f64)> {
    match dtype {
        dt if dt.is_integer() => numeric,
        PT::Decimal(_, Some(0)) => numeric,
        PT::Float32 | PT::Float64 => floats
            .filter(|f| f.n_fractional == 0 && f.n_nan == 0 && f.n_inf == 0)
            .and(numeric),
        PT::String | PT::Categorical(..) | PT::Enum(..) => {
            let st = strings?;
            if st.n_numeric_int != n_non_null || st.n_leading_zero != 0 || st.int_overflow {
                return None;
            }
            Some((st.int_min? as f64, st.int_max? as f64))
        }
        _ => None,
    }
}

/// First match wins: null → constant → boolean → ordinal → categorical → discrete.
/// `n_values`: values at this level, nulls included.
pub(crate) fn classify(
    n_values: u64,
    n_null: u64,
    count: Count,
    whole: Option<(f64, f64)>,
    est: f64,
    threshold: u64,
) -> &'static str {
    if n_null == n_values {
        return "null";
    }
    match count {
        Count::Exact(1) => return "constant",
        Count::Exact(2) => return "boolean",
        _ => {}
    }
    if whole.is_some_and(|(lo, hi)| 0.0 <= lo && hi <= 2.0 * n_values as f64) {
        return "ordinal";
    }
    if est <= threshold as f64 {
        "categorical"
    } else {
        "discrete"
    }
}

/// One level's conclusions: the picked estimate, `estimates_agree`, `unique`, `class`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Conclusions {
    pub est: Estimate,
    pub agree: Option<bool>,
    pub unique: bool,
    pub class: &'static str,
}

/// `n_values`: values at this level, nulls included.
pub(crate) fn conclude(
    dtype: &PT,
    n_values: u64,
    n_null: u64,
    p: &Profile,
    threshold: u64,
) -> Conclusions {
    let n = n_values - n_null;
    let f = &p.freq;
    let count = match f.hll {
        Some((estimate, std_error)) => Count::Hll {
            estimate,
            std_error,
        },
        None => Count::Exact(f.n_unique),
    };
    let (est, agree) = pick_estimate(count, n, f.f1, f.f2, &f.capture_history);
    let unique = n > 0
        && match count {
            Count::Exact(d) => d == n,
            Count::Hll { .. } => f.all_once,
        };
    let whole = whole_range(dtype, n, p.numeric, p.floats.as_ref(), p.strings.as_ref());
    Conclusions {
        est,
        agree,
        unique,
        class: classify(
            n_values,
            n_null,
            count,
            whole,
            est.est_cardinality,
            threshold,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::datatypes::{Categories, FrozenCategories};

    #[test]
    fn classes_in_order() {
        let c = |nv, nn, count, whole, est| classify(nv, nn, count, whole, est, 10);
        assert_eq!(c(4, 4, Count::Exact(0), None, 0.0), "null");
        assert_eq!(c(0, 0, Count::Exact(0), None, 0.0), "null");
        assert_eq!(c(4, 1, Count::Exact(1), Some((7.0, 7.0)), 1.0), "constant");
        assert_eq!(c(4, 0, Count::Exact(2), Some((1.0, 5.0)), 2.0), "boolean");
        assert_eq!(c(5, 0, Count::Exact(5), Some((0.0, 4.0)), 5.0), "ordinal");
        assert_eq!(
            c(4, 0, Count::Exact(4), Some((0.0, 9.0)), 4.0),
            "categorical"
        ); // 9 > 2·4
        assert_eq!(
            c(4, 0, Count::Exact(4), Some((-3.0, 2.0)), 4.0),
            "categorical"
        );
        assert_eq!(c(40, 0, Count::Exact(20), None, 20.0), "discrete");
        let hll = Count::Hll {
            estimate: 2.0,
            std_error: 0.01,
        };
        assert_eq!(c(4, 0, hll, None, 2.0), "categorical"); // HLL is never boolean
    }

    #[test]
    fn whole_ranges() {
        let r = Some((0.0, 3.0));
        assert_eq!(whole_range(&PT::Int64, 4, r, None, None), r);
        assert_eq!(
            whole_range(&PT::Decimal(Some(10), Some(0)), 4, r, None, None),
            r
        );
        assert_eq!(
            whole_range(&PT::Decimal(Some(10), Some(2)), 4, r, None, None),
            None
        );
        assert_eq!(whole_range(&PT::Date, 4, r, None, None), None);
        let whole = FloatStats {
            n_fractional: 0,
            ..Default::default()
        };
        let frac = FloatStats {
            n_fractional: 1,
            ..Default::default()
        };
        assert_eq!(whole_range(&PT::Float64, 4, r, Some(&whole), None), r);
        assert_eq!(whole_range(&PT::Float64, 4, r, Some(&frac), None), None);
        let ints = StringStats {
            n_numeric_int: 4,
            int_min: Some(0),
            int_max: Some(3),
            ..Default::default()
        };
        assert_eq!(whole_range(&PT::String, 4, None, None, Some(&ints)), r);
        let zero = StringStats {
            n_leading_zero: 1,
            ..ints.clone()
        };
        assert_eq!(whole_range(&PT::String, 4, None, None, Some(&zero)), None);
    }

    #[test]
    fn whole_range_branches() {
        let r = Some((0.0, 3.0));
        assert_eq!(whole_range(&PT::UInt32, 4, r, None, None), r);
        let ints = StringStats {
            n_numeric_int: 4,
            int_min: Some(0),
            int_max: Some(3),
            ..Default::default()
        };
        let cats = Categories::global();
        let cat = PT::Categorical(cats.clone(), cats.mapping());
        let en = PT::from_frozen_categories(FrozenCategories::new(["a", "b"]).unwrap());
        assert_eq!(whole_range(&cat, 4, None, None, Some(&ints)), r);
        assert_eq!(whole_range(&en, 4, None, None, Some(&ints)), r);
        let nan = FloatStats {
            n_nan: 1,
            ..Default::default()
        };
        let inf = FloatStats {
            n_inf: 1,
            ..Default::default()
        };
        assert_eq!(whole_range(&PT::Float64, 4, r, Some(&nan), None), None);
        assert_eq!(whole_range(&PT::Float64, 4, r, Some(&inf), None), None);
        assert_eq!(whole_range(&PT::Float64, 4, r, None, None), None);
        assert_eq!(whole_range(&PT::String, 5, None, None, Some(&ints)), None);
        let over = StringStats {
            int_overflow: true,
            ..ints
        };
        assert_eq!(whole_range(&PT::String, 4, None, None, Some(&over)), None);
    }

    #[test]
    fn ordinal_bounds_and_threshold() {
        let c = |whole, est| classify(5, 0, Count::Exact(5), whole, est, 10);
        assert_eq!(c(Some((0.0, 10.0)), 50.0), "ordinal");
        assert_eq!(c(Some((0.0, 10.1)), 50.0), "discrete");
        assert_eq!(c(None, 10.0), "categorical");
        assert_eq!(c(None, 10.5), "discrete");
    }

    #[test]
    fn whole_range_feeds_classify() {
        let ints = StringStats {
            n_numeric_int: 5,
            int_min: Some(0),
            int_max: Some(4),
            ..Default::default()
        };
        let w = whole_range(&PT::String, 5, None, None, Some(&ints));
        assert_eq!(classify(5, 0, Count::Exact(5), w, 5.0, 1), "ordinal");
        let f = FloatStats::default();
        let w = whole_range(&PT::Float64, 5, Some((0.0, 4.0)), Some(&f), None);
        assert_eq!(classify(5, 0, Count::Exact(5), w, 5.0, 1), "ordinal");
    }

    #[test]
    fn conclude_on_a_profile() {
        use polars::prelude::*;
        let s = Series::new("x".into(), &[0i64, 4, 1, 2, 3]);
        let p = crate::describe::profile(&s, 0, false).unwrap();
        let c = conclude(s.dtype(), 5, 0, &p, 10_000);
        assert_eq!(
            (c.class, c.unique, c.est.method.name()),
            ("ordinal", true, "observed")
        );
        assert_eq!((p.min.as_deref(), p.max.as_deref()), (Some("0"), Some("4")));
    }
}
