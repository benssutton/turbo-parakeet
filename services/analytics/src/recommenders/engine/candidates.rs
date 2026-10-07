//! Candidate types for a column and the rules that propose them (spec 2026-09-26 §4, §5.2).

use super::*;
use crate::techniques::cardinality_estimators::{pick_estimate, Count, Estimate};
use crate::techniques::describe::{Profile, StringStats};
use arrow_array::cast::AsArray;
use arrow_array::types::{Decimal128Type, Float64Type};
use arrow_array::{Array, ArrayRef, LargeStringArray};
use arrow_schema::DataType as AT;
use polars::prelude::DataType as PT;
use std::cell::OnceCell;

// ── candidates (Spec B §4, §5.2) ────────────────────────────────────────────

/// Recommend's constructor keywords.
#[derive(Clone, Debug)]
pub(crate) struct Params {
    pub seed: u64,
    pub zstd_level: i32,
    pub categorical_threshold: u64,
    pub boolean_pairs: Vec<(String, String)>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Outcome {
    Chosen,
    Failed,
    Rejected,
    NotTried,
}

impl Outcome {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Outcome::Chosen => "chosen",
            Outcome::Failed => "failed",
            Outcome::Rejected => "rejected",
            Outcome::NotTried => "not_tried",
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Candidate {
    pub target: Target,
    pub rank: Rank,
    pub rule: String,
    /// The metric values the rule tested and what they implied.
    pub evidence: String,
    pub predicted: u64,
    pub projected: f64,
    pub outcome: Outcome,
    pub reason: Option<String>,
}

/// One level of a column: the column, or a list's inner values.
pub(crate) struct Level<'a> {
    /// Polars dtype of these values.
    pub dtype: &'a PT,
    /// The values in Arrow's classic layout (CompatLevel::oldest): the whole level
    /// (one-shot), a sample block, or none (streaming) — the rules read only its type.
    pub values: ArrayRef,
    pub p: &'a Profile,
    pub n_rows: u64,
    pub n_null: u64,
    /// Exact min / max: integers, a decimal's unscaled values, or a temporal's
    /// physical values.
    pub int_range: Option<(i128, i128)>,
    /// Exact min / max of a float level, NaN excluded.
    pub float_range: Option<(f64, f64)>,
    /// Every distinct non-null value as text, when there are at most five (the
    /// boolean-pair rule); empty otherwise.
    pub few_distinct: Vec<String>,
    pub n_midnight: Option<u64>,
    /// Uncompressed size of the original type.
    pub size_bytes: u64,
    /// How `size_bytes` was obtained, for the original candidate's evidence.
    pub size_note: &'static str,
    pub est: Estimate,
    /// "" or "inner: " — prefixes every rule name.
    pub prefix: &'static str,
    /// Text sources: `values` as LargeUtf8, built once and shared by cast, verify and lossy.
    pub text: OnceCell<Result<LargeStringArray, String>>,
}

impl<'a> Level<'a> {
    /// A level over all its values (one-shot): counts, extremes and the few distinct
    /// values are read off `values` at Describe's argmin / argmax / first-occurrence rows.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn of_values(
        dtype: &'a PT,
        values: ArrayRef,
        p: &'a Profile,
        n_midnight: Option<u64>,
        size_bytes: u64,
        est: Estimate,
        prefix: &'static str,
    ) -> Result<Self, String> {
        let row = |i: Option<u64>| i.filter(|&i| (i as usize) < values.len());
        let (lo, hi) = (row(p.range.argmin), row(p.range.argmax));
        let int_range = match (dtype, lo, hi) {
            (PT::Decimal(..), Some(a), Some(b)) => values
                .as_primitive_opt::<Decimal128Type>()
                .map(|d| (d.value(a as usize), d.value(b as usize))),
            (dt, Some(a), Some(b)) if dt.is_integer() => int_at(&values, a).zip(int_at(&values, b)),
            (PT::Date | PT::Datetime(..) | PT::Duration(_) | PT::Time, Some(a), Some(b)) => {
                physical_at(&values, a).zip(physical_at(&values, b))
            }
            _ => None,
        };
        let float_range = match (is_float(dtype), lo, hi) {
            (true, Some(a), Some(b)) => Some((f64_at(&values, a), f64_at(&values, b))),
            _ => None,
        };
        let few_distinct = if is_text(dtype) && p.freq.n_unique <= 5 {
            p.freq
                .first_few
                .iter()
                .map(|&i| text_of(&values.slice(i as usize, 1)).map(|t| t.value(0).to_string()))
                .collect::<Result<_, _>>()?
        } else {
            Vec::new()
        };
        Ok(Level {
            dtype,
            n_rows: values.len() as u64,
            n_null: values.logical_null_count() as u64,
            values,
            p,
            int_range,
            float_range,
            few_distinct,
            n_midnight,
            size_bytes,
            size_note: "measured",
            est,
            prefix,
            text: Default::default(),
        })
    }

    /// A level over one sample block, for `cast_to` / `verify` only (streaming's cross-check).
    pub(crate) fn of_block(dtype: &'a PT, values: ArrayRef, p: &'a Profile) -> Self {
        Level {
            dtype,
            n_rows: values.len() as u64,
            n_null: values.logical_null_count() as u64,
            values,
            p,
            int_range: None,
            float_range: None,
            few_distinct: Vec::new(),
            n_midnight: None,
            size_bytes: 0,
            size_note: "",
            est: pick_estimate(Count::Exact(0), 0, 0, 0, &[0; 7]).0,
            prefix: "",
            text: Default::default(),
        }
    }
}

impl Level<'_> {
    /// The values as text (LargeUtf8), computed on first use.
    pub(crate) fn text(&self) -> Result<&LargeStringArray, String> {
        self.text
            .get_or_init(|| text_of(&self.values))
            .as_ref()
            .map_err(|e| e.clone())
    }

    pub(crate) fn n_rows(&self) -> u64 {
        self.n_rows
    }

    pub(crate) fn n_null(&self) -> u64 {
        self.n_null
    }

    pub(crate) fn n(&self) -> u64 {
        self.n_rows() - self.n_null()
    }

    /// Estimated dictionary cardinality: est_high where an interval exists,
    /// floored at the observed distinct count (an estimate must never claim fewer
    /// distinct values than were actually observed).
    pub(crate) fn cardinality(&self) -> (f64, &'static str) {
        let (c, source) = match self.est.est_high {
            Some(h) => (h, "est_high"),
            None => (self.est.est_cardinality, "est_cardinality"),
        };
        (c.max(self.p.freq.n_unique as f64), source)
    }

    pub(crate) fn shape(&self) -> Shape {
        Shape {
            n: self.n_rows() as f64,
            nulls: self.n_null() as f64,
            sum_len: self.p.sum_len.unwrap_or(0) as f64,
            d: self.p.freq.n_unique as f64,
            sum_len_unique: self.p.freq.sum_len_unique.unwrap_or(0) as f64,
        }
    }
}

pub(crate) fn is_text(dt: &PT) -> bool {
    matches!(dt, PT::String | PT::Categorical(..) | PT::Enum(..))
}

pub(crate) fn is_float(dt: &PT) -> bool {
    matches!(dt, PT::Float32 | PT::Float64)
}

pub(crate) fn text_of(values: &ArrayRef) -> Result<LargeStringArray, String> {
    Ok(arrow_cast(values.as_ref(), &AT::LargeUtf8)?
        .as_string::<i64>()
        .clone())
}

/// Row `i` as an exact integer (integer columns).
pub(crate) fn int_at(a: &ArrayRef, i: u64) -> Option<i128> {
    let v = arrow_cast(a.slice(i as usize, 1).as_ref(), &AT::Decimal128(38, 0)).ok()?;
    let v = v.as_primitive::<Decimal128Type>();
    v.is_valid(0).then(|| v.value(0))
}

/// A temporal value's physical integer (days, or the time unit) at row `i`.
pub(crate) fn physical_at(a: &ArrayRef, i: u64) -> Option<i128> {
    let v = a.slice(i as usize, 1);
    let v = match v.data_type() {
        AT::Date32 | AT::Time32(_) => arrow_cast(
            arrow_cast(v.as_ref(), &AT::Int32).ok()?.as_ref(),
            &AT::Int64,
        ),
        _ => arrow_cast(v.as_ref(), &AT::Int64),
    }
    .ok()?;
    let v = v.as_primitive::<arrow_array::types::Int64Type>();
    v.is_valid(0).then(|| v.value(0) as i128)
}

pub(crate) fn f64_at(a: &ArrayRef, i: u64) -> f64 {
    arrow_cast(a.slice(i as usize, 1).as_ref(), &AT::Float64)
        .map_or(f64::NAN, |v| v.as_primitive::<Float64Type>().value(0))
}

pub(crate) struct Rules<'l, 'a> {
    pub(crate) lvl: &'l Level<'a>,
    pub(crate) shape: Shape,
    pub(crate) out: Vec<Candidate>,
}

impl Rules<'_, '_> {
    fn push(&mut self, target: Target, rank: Rank, rule: &str, evidence: String) {
        let t = target.arrow_type();
        let (c, _) = self.lvl.cardinality();
        let sizes = body_size(&t, &self.shape)
            .and_then(|p| Ok((p, body_size(&t, &self.shape.project(c))?)));
        self.out.push(candidate(
            target,
            rank,
            &format!("{}{rule}", self.lvl.prefix),
            evidence,
            sizes,
        ));
    }

    fn integers(&mut self, lo: i128, hi: i128, from: &str, evidence: &str) {
        let ev = format!("{evidence}min={lo} max={hi}");
        if lo >= 0 && hi <= 1 {
            self.push(
                Target::Boolean,
                Rank::Boolean,
                &format!("{from}→boolean"),
                ev.clone(),
            );
        }
        let uint = if lo >= 0 { narrowest_uint(hi) } else { None };
        let int = narrowest_int(lo, hi);
        if let Some(t) = uint.clone() {
            self.push(
                Target::Fixed(t),
                Rank::UInt,
                &format!("{from}→uint"),
                ev.clone(),
            );
        }
        if let Some(t) = int.clone() {
            self.push(
                Target::Fixed(t),
                Rank::Int,
                &format!("{from}→int"),
                ev.clone(),
            );
        }
        if uint.is_none() && int.is_none() {
            let p = digits(lo.unsigned_abs().max(hi.unsigned_abs()));
            if p <= 38 {
                self.push(
                    Target::Fixed(AT::Decimal128(p, 0)),
                    Rank::Decimal,
                    &format!("{from}→decimal128"),
                    format!("{ev} → p={p}"),
                );
            }
        }
    }

    fn decimal(&mut self, scale: usize) {
        let Some((lo, hi)) = self.lvl.int_range else {
            return;
        };
        let g = self.lvl.p.gcd.unwrap_or(1);
        let k = if g == 0 {
            scale
        } else {
            trailing_zeros10(g).min(scale)
        };
        let f = 10i128.pow(k as u32);
        let (lo, hi, s) = (lo / f, hi / f, scale - k);
        let ev = format!("gcd={g} → {k} trailing zeros, scale {scale}→{s}; ");
        if s == 0 {
            return self.integers(lo, hi, "decimal", &ev);
        }
        let prec = digits(lo.unsigned_abs().max(hi.unsigned_abs())).max(s as u8);
        if prec <= 38 {
            self.push(
                Target::Fixed(decimal_type(prec, s as i8)),
                Rank::Decimal,
                "decimal→decimal",
                format!("{ev}min={lo} max={hi} → p={prec} s={s}"),
            );
        }
    }

    fn float(&mut self) {
        let p = self.lvl.p;
        let Some(f) = p.floats else { return }; // Describe profiles every float column: never None
        let ev = format!(
            "n_nan={} n_inf={} n_fractional={} ",
            f.n_nan, f.n_inf, f.n_fractional
        );
        if let (0, 0, Some((lo, hi))) = (f.n_nan, f.n_inf, self.lvl.float_range) {
            let top = lo.abs().max(hi.abs());
            if f.n_fractional == 0 {
                if top < 1e38 {
                    self.integers(lo as i128, hi as i128, "float", &ev);
                }
            } else if let Some(s) = f.max_frac_digits {
                let int_digits = if top < 1.0 {
                    0
                } else {
                    digits(top.floor() as u128) as u32
                };
                let prec = int_digits + s;
                if prec <= 38 {
                    self.push(
                        Target::Fixed(decimal_type(prec as u8, s as i8)),
                        Rank::Decimal,
                        "float→decimal",
                        format!("{ev}max_frac_digits={s} int_digits={int_digits} → p={prec} s={s}"),
                    );
                }
            }
        }
        if self.lvl.dtype == &PT::Float64 && f.n_f32_inexact == 0 {
            self.push(
                Target::Fixed(AT::Float32),
                Rank::Float,
                "float64→float32",
                "n_f32_inexact=0".into(),
            );
        }
    }

    fn temporal(&mut self) {
        let lvl = self.lvl;
        let (unit, tz) = match lvl.values.data_type() {
            AT::Timestamp(u, tz) => (*u, tz.clone()),
            AT::Duration(u) | AT::Time64(u) | AT::Time32(u) => (*u, None),
            _ => return,
        };
        let g = lvl.p.gcd.unwrap_or(1);
        let coarse = coarsest_unit(g * unit_ns(&unit));
        let ev = format!("gcd={g} ({}) → {}", unit_name(&unit), unit_name(&coarse));
        match lvl.dtype {
            PT::Datetime(_, zone) => {
                if zone.is_none() && lvl.n_midnight == Some(lvl.n()) {
                    self.push(
                        Target::Fixed(AT::Date32),
                        Rank::Date,
                        "datetime→date32",
                        format!("n_midnight={}", lvl.n()),
                    );
                }
                if coarse != unit {
                    self.push(
                        Target::Fixed(AT::Timestamp(coarse, tz)),
                        Rank::Timestamp,
                        "datetime→timestamp",
                        ev,
                    );
                }
            }
            PT::Duration(_) if coarse != unit => self.push(
                Target::Fixed(AT::Duration(coarse)),
                Rank::Duration,
                "duration→duration",
                ev,
            ),
            PT::Time if coarse != unit => self.push(
                Target::Fixed(time_type(coarse)),
                Rank::Time,
                "time→time",
                ev,
            ),
            _ => {}
        }
    }

    fn text(&mut self, params: &Params) -> Result<(), String> {
        let lvl = self.lvl;
        let n = lvl.n();
        let distinct: Vec<String> = if lvl.p.freq.n_unique <= 5 {
            lvl.few_distinct.iter().map(|v| v.to_lowercase()).collect()
        } else {
            Vec::new()
        };
        let pair = params
            .boolean_pairs
            .iter()
            .map(|(t, f)| (t.to_lowercase(), f.to_lowercase()))
            .find(|(t, f)| !distinct.is_empty() && distinct.iter().all(|v| v == t || v == f));
        // Describe profiles every text column, so `strings` is always present; without
        // it only the plain / dictionary candidates apply.
        let Some(st) = lvl.p.strings.as_ref() else {
            self.plain_and_dictionary(params);
            return Ok(());
        };
        let sig = st.iso_max_sig_frac_digits.unwrap_or(0);
        if let Some((t, f)) = pair {
            self.push(
                Target::BoolPair(t.clone(), f.clone()),
                Rank::Boolean,
                "string→boolean",
                format!("distinct={distinct:?} pair=({t:?}, {f:?})"),
            );
        } else if let (true, Some(lo), Some(hi)) = (
            st.n_numeric_int == n && st.n_leading_zero == 0 && !st.int_overflow,
            st.int_min,
            st.int_max,
        ) {
            self.integers(
                lo,
                hi,
                "string",
                &format!("n_numeric_int={n} n_leading_zero=0 "),
            );
        } else if st.n_numeric == n && st.n_leading_zero == 0 {
            self.numeric_text(st, n);
        } else {
            self.iso_text(st, n, sig);
        }
        self.plain_and_dictionary(params);
        Ok(())
    }

    /// Numeric strings: decimal, and float when the places vary and precision is high.
    fn numeric_text(&mut self, st: &StringStats, n: u64) {
        let (i, f) = (
            st.max_int_digits.unwrap_or(0),
            st.max_frac_digits.unwrap_or(0),
        );
        let (min_f, sig_d) = (
            st.min_frac_digits.unwrap_or(0),
            st.max_sig_digits.unwrap_or(0),
        );
        let prec = (i + f).max(1);
        let ev = format!(
            "n_numeric={n} n_leading_zero=0 numeric_max_int_digits={i} numeric_max_frac_digits={f} \
             numeric_min_frac_digits={min_f} numeric_max_sig_digits={sig_d}"
        );
        if prec <= 38 {
            self.push(
                Target::Fixed(decimal_type(prec as u8, f as i8)),
                Rank::Decimal,
                "string→decimal",
                format!("{ev} → p={prec} s={f}"),
            );
        }
        if min_f < f && prec > 18 {
            let why = format!("{ev} → varying places, p={prec} > 18");
            if sig_d <= 6 {
                self.push(
                    Target::Fixed(AT::Float32),
                    Rank::Float,
                    "string→float32",
                    why.clone(),
                );
            }
            if sig_d <= 15 {
                self.push(
                    Target::Fixed(AT::Float64),
                    Rank::Float,
                    "string→float64",
                    why,
                );
            }
        }
    }

    /// ISO 8601 strings: date, time, datetime or datetime with offset.
    fn iso_text(&mut self, st: &StringStats, n: u64, sig: u32) {
        if st.n_iso_date == n {
            self.push(
                Target::Fixed(AT::Date32),
                Rank::Date,
                "string→date32",
                format!("n_iso_date={n}"),
            );
        } else if st.n_iso_time == n {
            self.push(
                Target::Fixed(time_type(iso_unit(sig))),
                Rank::Time,
                "string→time",
                format!("n_iso_time={n} iso_max_sig_frac_digits={sig}"),
            );
        } else if st.n_iso_datetime == n {
            if st.iso_n_midnight == n {
                self.push(
                    Target::Fixed(AT::Date32),
                    Rank::Date,
                    "string→date32",
                    format!("n_iso_datetime={n} iso_n_midnight={n}"),
                );
            } else {
                self.push(
                    Target::Fixed(AT::Timestamp(iso_unit(sig), None)),
                    Rank::Timestamp,
                    "string→timestamp",
                    format!("n_iso_datetime={n} iso_max_sig_frac_digits={sig}"),
                );
            }
        } else if st.n_iso_datetime_tz == n {
            let ev = format!(
                "n_iso_datetime_tz={n} iso_n_offsets={} iso_max_sig_frac_digits={sig}",
                st.offsets.len()
            );
            match st.offsets.iter().next() {
                Some(&m) if st.offsets.len() == 1 => self.push(
                    Target::Fixed(AT::Timestamp(iso_unit(sig), Some(offset_tz(m).into()))),
                    Rank::Timestamp,
                    "string→timestamp",
                    ev,
                ),
                _ => self.push(
                    Target::TimestampWithOffset(iso_unit(sig)),
                    Rank::TimestampWithOffset,
                    "string→timestamp_with_offset (arrow.timestamp_with_offset)",
                    ev,
                ),
            }
        }
    }

    /// Step 2 for text (Spec B §4.3 "always", §5.2): Utf8 with 32-bit offsets, and the dictionary.
    fn plain_and_dictionary(&mut self, params: &Params) {
        let sum_len = self.lvl.p.sum_len.unwrap_or(0);
        if sum_len < 1 << 31 {
            self.push(
                Target::Plain(AT::Utf8),
                Rank::Plain,
                "string→utf8",
                format!("sum_len={sum_len}"),
            );
        }
        self.dictionary(params);
    }

    fn dictionary(&mut self, params: &Params) {
        let (c, source) = self.lvl.cardinality();
        let (key, polars_key) = dictionary_keys(c);
        let threshold = params.categorical_threshold;
        let est = &self.lvl.est;
        let low = est
            .est_low
            .map_or(String::new(), |l| format!(" est_low={l:?}"));
        let order = if self.lvl.p.ranking.is_some() {
            " key_order=frequency"
        } else {
            ""
        };
        let ev = format!(
            "c={c:?} from {source} method={}{low} n_unique={} categorical_threshold={threshold}{order}",
            est.method.name(),
            self.lvl.p.freq.n_unique
        );
        self.push(
            Target::Dictionary(key, polars_key),
            Rank::Dictionary,
            "string→dictionary",
            ev,
        );
        if c > threshold as f64 {
            let last = self.out.last_mut().unwrap();
            last.outcome = Outcome::Rejected;
            last.reason = Some(format!("c={c:?} > categorical_threshold={threshold}"));
        }
    }

    fn binary(&mut self) {
        let sum_len = self.lvl.p.sum_len.unwrap_or(0);
        if sum_len < 1 << 31 {
            self.push(
                Target::Plain(AT::Binary),
                Rank::Plain,
                "binary→binary",
                format!("sum_len={sum_len}"),
            );
        }
    }

    pub(crate) fn original(&mut self) {
        let lvl = self.lvl;
        self.out.push(Candidate {
            target: Target::Original(lvl.values.data_type().clone()),
            rank: Rank::Original,
            rule: format!("{}original", lvl.prefix),
            evidence: format!("{} size_bytes={}", lvl.size_note, lvl.size_bytes),
            predicted: lvl.size_bytes,
            projected: lvl.size_bytes as f64,
            outcome: Outcome::NotTried,
            reason: None,
        });
    }
}

/// Every candidate for one level, in rule order (Spec B §4.2–4.3, §5.2); the
/// original type last.
pub(crate) fn candidates(lvl: &Level, params: &Params) -> Result<Vec<Candidate>, String> {
    let mut r = Rules {
        lvl,
        shape: lvl.shape(),
        out: Vec::new(),
    };
    if lvl.n_rows() > 0 && lvl.n() == 0 {
        r.push(
            Target::Null,
            Rank::Null,
            "all-null→null",
            format!("n_null={} n_rows={}", lvl.n_null(), lvl.n_rows()),
        );
    } else if lvl.n() > 0 {
        match lvl.dtype {
            PT::Int8
            | PT::Int16
            | PT::Int32
            | PT::Int64
            | PT::UInt8
            | PT::UInt16
            | PT::UInt32
            | PT::UInt64 => {
                if let Some((lo, hi)) = lvl.int_range {
                    r.integers(lo, hi, "integer", "");
                }
            }
            PT::Decimal(_, s) => r.decimal(s.unwrap_or(0)),
            PT::Float32 | PT::Float64 => r.float(),
            PT::Datetime(..) | PT::Duration(_) | PT::Time => r.temporal(),
            dt if is_text(dt) => r.text(params)?,
            PT::Binary => r.binary(),
            _ => {}
        }
    }
    r.original();
    Ok(r.out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::arrow_io::export_series;
    use crate::recommenders::engine::choose::tests::{describe_one, params};
    use polars::prelude::{
        CompatLevel, DataType as PT, IntoSeries, NamedFrom, NewChunkedArray, Series,
        TimeUnit as PTimeUnit,
    };

    fn types(s: Series, p: &Params) -> Vec<(String, Outcome)> {
        let d = describe_one(&s, 0).unwrap();
        let lvl = Level::of_values(
            s.dtype(),
            export_series(&s, CompatLevel::oldest()).unwrap(),
            &d.outer,
            d.n_midnight,
            0,
            d.conclusions(10_000).0.est,
            "",
        )
        .unwrap();
        candidates(&lvl, p)
            .unwrap()
            .iter()
            .map(|c| (pa_name(&c.target.arrow_type()), c.outcome))
            .collect()
    }

    fn names(s: Series) -> Vec<String> {
        types(s, &params()).into_iter().map(|(t, _)| t).collect()
    }

    #[test]
    fn integer_rules() {
        assert_eq!(
            names(Series::new("x".into(), &[0i64, 1])),
            ["bool", "uint8", "int8", "int64"]
        );
        assert_eq!(
            names(Series::new("x".into(), &[-200i64, 5])),
            ["int16", "int64"]
        );
    }

    #[test]
    fn decimal_scale_reduced_by_gcd() {
        let dec = polars::prelude::Int128Chunked::from_slice("x".into(), &[120, 340])
            .into_decimal_unchecked(Some(10), 2)
            .into_series();
        assert_eq!(names(dec), ["decimal32(2, 1)", "decimal128(10, 2)"]);
    }

    #[test]
    fn float_rules() {
        assert_eq!(
            names(Series::new("x".into(), &[123.45f64, 99.99])),
            ["decimal32(5, 2)", "double"]
        );
        assert_eq!(
            names(Series::new("x".into(), &[0.5f64, 0.25])),
            ["decimal32(2, 2)", "float", "double"]
        );
        assert_eq!(
            names(Series::new("x".into(), &[0.1f64, f64::NAN])),
            ["double"]
        );
    }

    #[test]
    fn string_rules() {
        let dict = "dictionary<values=string, indices=uint8, ordered=0>";
        assert_eq!(
            names(Series::new("x".into(), &["007", "12"])),
            ["string", dict, "large_string"]
        );
        assert_eq!(
            names(Series::new("x".into(), &["1234567890.1", "0.00000012345"])),
            [
                "decimal128(21, 11)",
                "double",
                "string",
                dict,
                "large_string"
            ]
        );
        let offsets = names(Series::new(
            "x".into(),
            &["2024-01-05T10:00+05:00", "2024-01-05T10:00-03:30"],
        ));
        assert_eq!(
            offsets[0],
            "struct<timestamp: timestamp[s, tz=UTC] not null, offset_minutes: int16 not null>"
        );
        assert_eq!(
            names(Series::new("x".into(), &["True", "false"]))[0],
            "bool"
        );
    }

    #[test]
    fn temporal_rules() {
        let days = Series::new("x".into(), &[0i64, 86_400_000_000])
            .cast(&PT::Datetime(PTimeUnit::Microseconds, None))
            .unwrap();
        assert_eq!(
            names(days),
            ["date32[day]", "timestamp[s]", "timestamp[us]"]
        );
    }

    #[test]
    fn missing_profile_parts_give_no_type_candidates_not_panics() {
        // A profile whose parts do not match the dtype (Describe never builds one): the
        // rules bail instead of panicking — only the original (and text's plain/dictionary).
        let ints = Series::new("x".into(), &[1i64, 2]);
        let d = describe_one(&ints, 0).unwrap();
        let values = export_series(&ints, CompatLevel::oldest()).unwrap();
        for dtype in [PT::Float64, PT::Decimal(Some(10), Some(2)), PT::String] {
            let lvl = Level::of_values(
                &dtype,
                values.clone(),
                &d.outer,
                None,
                0,
                d.conclusions(10_000).0.est,
                "",
            )
            .unwrap();
            let got: Vec<String> = candidates(&lvl, &params())
                .unwrap()
                .iter()
                .map(|c| c.rule.clone())
                .collect();
            let expected: &[&str] = if dtype == PT::String {
                &["string→utf8", "string→dictionary", "original"]
            } else {
                &["original"]
            };
            assert_eq!(got, expected, "{dtype}");
        }
    }

    #[test]
    fn dictionary_evidence_names_the_estimator() {
        let s = Series::new("x".into(), &["a", "b", "a", "b", "c"]);
        let d = describe_one(&s, 0).unwrap();
        let lvl = Level::of_values(
            s.dtype(),
            export_series(&s, CompatLevel::oldest()).unwrap(),
            &d.outer,
            None,
            0,
            d.conclusions(10_000).0.est,
            "",
        )
        .unwrap();
        let evidence = candidates(&lvl, &params())
            .unwrap()
            .into_iter()
            .find(|c| c.rule == "string→dictionary")
            .unwrap()
            .evidence;
        assert!(evidence.contains("method=observed est_low=3"), "{evidence}");
    }

    #[test]
    fn dictionary_gate_rejects() {
        let p = Params {
            categorical_threshold: 1,
            ..params()
        };
        let got = types(Series::new("x".into(), &["a", "b", "a", "b"]), &p);
        assert!(got
            .iter()
            .any(|(t, o)| t.starts_with("dictionary") && *o == Outcome::Rejected));
    }

    #[test]
    fn dictionary_cardinality_floors_at_n_unique() {
        // 3 distinct values, but the estimate handed to Level is (artificially)
        // below that — cardinality() must still report at least n_unique.
        let s = Series::new("x".into(), &["a", "b", "c"]);
        let d = describe_one(&s, 0).unwrap();
        assert_eq!(d.outer.freq.n_unique, 3);
        let lvl = Level::of_values(
            s.dtype(),
            export_series(&s, CompatLevel::oldest()).unwrap(),
            &d.outer,
            d.n_midnight,
            0,
            Estimate {
                est_cardinality: 1.0,
                est_low: Some(1.0),
                est_high: Some(1.0),
                method: crate::techniques::cardinality_estimators::Method::Chao1,
            },
            "",
        )
        .unwrap();
        assert_eq!(lvl.cardinality(), (3.0, "est_high"));
    }
}
