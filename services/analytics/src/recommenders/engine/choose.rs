//! Trying candidates smallest first and choosing one (spec 2026-09-26 §4.1, §4.4, §5.4).

use super::*;
use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, FixedSizeListArray, ListArray, StructArray, UInt64Array};
use arrow_buffer::{NullBuffer, OffsetBuffer, ScalarBuffer};
use arrow_schema::{DataType as AT, Field as AField};
use polars::prelude::DataType as PT;
use std::sync::Arc;

// ── choosing (Spec B §4.1, §4.4, §5.4) ──────────────────────────────────────

pub(crate) struct Chosen {
    pub target: Target,
    pub rank: Rank,
    pub array: ArrayRef,
    pub lossy: bool,
    pub predicted: u64,
    pub projected: f64,
    pub candidates: Vec<Candidate>,
}

/// Candidates in the order tried: rejected last, then smallest projected size, then rank.
pub(crate) fn order(c: &mut [Candidate]) {
    c.sort_by(|a, b| {
        (a.outcome == Outcome::Rejected)
            .cmp(&(b.outcome == Outcome::Rejected))
            .then(a.projected.total_cmp(&b.projected))
            .then(a.rank.cmp(&b.rank))
    });
}

/// Tries the candidates in order; the first success is chosen, failures keep their
/// reason. Only `not_tried` candidates are attempted (rejected ones, and ones that
/// could not be sized, already carry their outcome).
pub(crate) fn first_success<T>(
    mut cands: Vec<Candidate>,
    mut attempt: impl FnMut(&Target) -> Result<T, String>,
) -> (usize, T, Vec<Candidate>) {
    order(&mut cands);
    for i in 0..cands.len() {
        if cands[i].outcome != Outcome::NotTried {
            continue;
        }
        match attempt(&cands[i].target) {
            Ok(a) => {
                cands[i].outcome = Outcome::Chosen;
                return (i, a, cands);
            }
            Err(reason) => {
                cands[i].outcome = Outcome::Failed;
                cands[i].reason = Some(reason);
            }
        }
    }
    unreachable!("the original type is always a candidate and cannot fail")
}

pub(crate) fn chosen_from(i: usize, array: ArrayRef, cands: Vec<Candidate>, lossy: bool) -> Chosen {
    let c = &cands[i];
    Chosen {
        target: c.target.clone(),
        rank: c.rank,
        lossy,
        predicted: c.predicted,
        projected: c.projected,
        array,
        candidates: cands,
    }
}

pub(crate) fn choose(lvl: &Level, params: &Params) -> Result<Chosen, String> {
    let (i, (array, rendered), cands) = first_success(candidates(lvl, params)?, |t| {
        let a = cast_to(t, lvl)?;
        let rendered = verify(t, lvl, &a)?;
        Ok((a, rendered))
    });
    let lossy = lossy(&cands[i].target, lvl, &array, rendered);
    Ok(chosen_from(i, array, cands, lossy))
}

/// The original type only; `why` explains, in its evidence, why nothing else was tried.
pub(crate) fn choose_original(lvl: &Level, why: &str) -> Chosen {
    let mut r = Rules {
        lvl,
        shape: lvl.shape(),
        out: Vec::new(),
    };
    r.original();
    r.out[0].evidence = format!("{}; {why}", r.out[0].evidence);
    let (i, array, cands) = first_success(r.out, |_| Ok(lvl.values.clone()));
    chosen_from(i, array, cands, false)
}

/// Per row of a List/Array column: (start, len) in the child the inner level holds,
/// or None for a null row; that child; the fixed width (None for List). The child is
/// Describe's `flatten` — the values of the non-null rows only, in row order — so the
/// inner profile's argmin/argmax indices point into it: valid row k starts where valid
/// row k−1 ends. A null row can still span values (a List's offsets after
/// `pl.when(mask).then(list).otherwise(None)`; an Array's w slots); those are dropped
/// with a `take`, and `wrap` rebuilds the offsets from row lengths (a null row → empty).
/// (row spans, compacted values, fixed width).
pub(crate) type ListParts = (Vec<Option<(usize, usize)>>, ArrayRef, Option<i32>);

pub(crate) fn list_parts(values: &ArrayRef) -> Result<ListParts, String> {
    // (physical start, len) of every row, then the compacted rows.
    let (spans, child, width): (Vec<(usize, usize, bool)>, ArrayRef, Option<i32>) =
        match values.data_type() {
            AT::LargeList(_) => {
                let l = values.as_list::<i64>();
                let o = l.value_offsets();
                let first = o[0] as usize;
                let spans = (0..l.len())
                    .map(|i| {
                        (
                            o[i] as usize - first,
                            (o[i + 1] - o[i]) as usize,
                            l.is_valid(i),
                        )
                    })
                    .collect();
                (
                    spans,
                    l.values().slice(first, o[l.len()] as usize - first),
                    None,
                )
            }
            AT::FixedSizeList(_, width) => {
                let f = values.as_fixed_size_list();
                let w = *width as usize;
                let child = f.values().slice(f.value_offset(0) as usize, f.len() * w); // offset non-zero for a slice
                (
                    (0..f.len()).map(|i| (i * w, w, f.is_valid(i))).collect(),
                    child,
                    Some(*width),
                )
            }
            t => return Err(format!("not a list type: {}", pa_name(t))),
        };
    let mut k = 0;
    let rows = spans
        .iter()
        .map(|&(_, len, valid)| {
            valid.then(|| {
                k += len;
                (k - len, len)
            })
        })
        .collect();
    if !spans.iter().any(|&(_, len, valid)| !valid && len > 0) {
        return Ok((rows, child, width)); // no null row spans values: the child is already compact
    }
    let idx = UInt64Array::from_iter_values(
        spans
            .iter()
            .filter(|s| s.2)
            .flat_map(|&(start, len, _)| (start..start + len).map(|j| j as u64)),
    );
    let compact =
        arrow_select::take::take(child.as_ref(), &idx, None).map_err(|e| e.to_string())?;
    Ok((rows, compact, width))
}

/// `take` whose null indices give null rows; a struct keeps its nulls at struct
/// level only, its children null-free (as `from_text` builds timestamp_with_offset
/// and as `body_size` predicts it).
pub(crate) fn take_rows(a: &ArrayRef, idx: &UInt64Array) -> Result<ArrayRef, String> {
    let e = |e: arrow_schema::ArrowError| e.to_string();
    match a.data_type() {
        AT::Struct(fields) if !a.is_empty() => {
            let s = a.as_struct();
            let dense = UInt64Array::from_iter_values(idx.iter().map(|i| i.unwrap_or(0)));
            let cols = s
                .columns()
                .iter()
                .map(|c| arrow_select::take::take(c.as_ref(), &dense, None))
                .collect::<Result<Vec<_>, _>>()
                .map_err(e)?;
            let valid: Vec<bool> = idx
                .iter()
                .map(|i| i.is_some_and(|i| s.is_valid(i as usize)))
                .collect();
            let nulls = valid.iter().any(|v| !v).then(|| NullBuffer::from(valid));
            StructArray::try_new(fields.clone(), cols, nulls)
                .map(|x| Arc::new(x) as ArrayRef)
                .map_err(e)
        }
        _ => arrow_select::take::take(a.as_ref(), idx, None).map_err(e),
    }
}

/// A list column rebuilt around its recast inner values (`rows` from `list_parts`).
pub(crate) fn wrap(
    t: &Target,
    values: &ArrayRef,
    rows: &[Option<(usize, usize)>],
    inner: &ArrayRef,
) -> Result<ArrayRef, String> {
    let field = |c: &ArrayRef| Arc::new(AField::new("item", c.data_type().clone(), true));
    let nulls = values.logical_nulls();
    match t {
        Target::Original(_) => Ok(values.clone()),
        Target::Scalar(_) => take_rows(
            inner,
            &UInt64Array::from(
                rows.iter()
                    .map(|r| r.map(|(start, _)| start as u64))
                    .collect::<Vec<_>>(),
            ),
        ),
        Target::List(_) => {
            let mut offsets = vec![0i32];
            for r in rows {
                let len = i32::try_from(r.map_or(0, |(_, len)| len)).map_err(|e| e.to_string())?;
                offsets.push(offsets.last().unwrap() + len);
            }
            ListArray::try_new(
                field(inner),
                OffsetBuffer::new(ScalarBuffer::from(offsets)),
                inner.clone(),
                nulls,
            )
            .map(|a| Arc::new(a) as ArrayRef)
            .map_err(|e| e.to_string())
        }
        Target::FixedList(_, w) => {
            // Re-expand the compacted inner values: a null row gets w null slots.
            let child = if rows.iter().all(Option::is_some) {
                inner.clone()
            } else {
                let w = *w as usize;
                let idx: Vec<Option<u64>> = rows
                    .iter()
                    .flat_map(|r| (0..w).map(move |j| r.map(|(start, _)| (start + j) as u64)))
                    .collect();
                take_rows(inner, &UInt64Array::from(idx))?
            };
            FixedSizeListArray::try_new(field(&child), *w, child, nulls)
                .map(|a| Arc::new(a) as ArrayRef)
                .map_err(|e| e.to_string())
        }
        t => Err(format!("not a list target: {t:?}")),
    }
}

/// A candidate with its (predicted, projected) sizes; one that cannot be sized is
/// listed as failed with the reason, and never tried.
pub(crate) fn candidate(
    target: Target,
    rank: Rank,
    rule: &str,
    evidence: String,
    sizes: Result<(f64, f64), String>,
) -> Candidate {
    let (predicted, projected, outcome, reason) = match sizes {
        Ok((p, q)) => (p as u64, q, Outcome::NotTried, None),
        Err(e) => (0, f64::INFINITY, Outcome::Failed, Some(e)),
    };
    Candidate {
        target,
        rank,
        rule: rule.into(),
        evidence,
        predicted,
        projected,
        outcome,
        reason,
    }
}

/// A list level's outer candidates (scalar / list / array, then the original) around
/// the inner choice `it` (with its rank and sizes), unordered.
#[allow(clippy::too_many_arguments)]
pub(crate) fn list_candidates(
    lvl: &Level,
    inner: &Level,
    it: &Target,
    rank: Rank,
    predicted: u64,
    projected: f64,
    width: Option<i32>,
) -> Vec<Candidate> {
    let (n, nulls) = (lvl.n_rows() as f64, lvl.n_null() as f64);
    let kept = matches!(it, Target::Original(_));
    let (c, _) = inner.cardinality();
    let mut outer = Vec::new();
    let nested = kept && matches!(inner.dtype, PT::List(_) | PT::Array(..) | PT::Struct(_));
    let single = lvl.p.range.min_len == Some(1) && lvl.p.range.max_len == Some(1);
    if single && !(lvl.n_null() > 0 && inner.n_null() > 0) && !nested {
        let shape = Shape {
            n,
            nulls: nulls + inner.n_null() as f64,
            ..inner.shape()
        };
        let t = it.arrow_type();
        outer.push(candidate(
            Target::Scalar(Box::new(it.clone())),
            rank,
            "list→scalar",
            format!(
                "min_len=1 max_len=1 n_null={} inner_n_null={}",
                lvl.n_null(),
                inner.n_null()
            ),
            body_size(&t, &shape).and_then(|p| Ok((p, body_size(&t, &shape.project(c))?))),
        ));
    }
    match width {
        None if inner.n_rows() < 1 << 31 => outer.push(candidate(
            Target::List(Box::new(it.clone())),
            Rank::List,
            "large_list→list",
            format!("inner_n_values={}", inner.n_rows()),
            Ok((
                validity(n, nulls) + pad(4.0 * (n + 1.0)) + predicted as f64,
                validity(n, nulls) + pad(4.0 * (n + 1.0)) + projected,
            )),
        )),
        // An Array whose inner type is kept is the original type: no candidate.
        Some(w) if !kept => {
            // The child holds w slots per row; a null row's slots are null.
            let wf = w as f64;
            let shape = Shape {
                n: n * wf,
                nulls: inner.n_null() as f64 + nulls * wf,
                ..inner.shape()
            };
            let t = it.arrow_type();
            outer.push(candidate(
                Target::FixedList(Box::new(it.clone()), w),
                Rank::List,
                "array→array",
                format!("width={w}"),
                body_size(&t, &shape).and_then(|p| {
                    Ok((
                        validity(n, nulls) + p,
                        validity(n, nulls) + body_size(&t, &shape.project(c))?,
                    ))
                }),
            ));
        }
        _ => {}
    }
    let mut rules = Rules {
        lvl,
        shape: lvl.shape(),
        out: outer,
    };
    rules.original();
    rules.out
}

/// Lists: choose the inner type first, then wrap it — as a scalar when every list
/// holds one item, else as a List with 32-bit offsets (Array keeps its width).
///
/// The reported candidates are the outer level's (scalar / list / array / original,
/// in the order tried) followed by the inner level's (rules prefixed "inner: "), so
/// a list column shows two `chosen` entries: the outer choice and the inner choice.
pub(crate) fn choose_list(
    lvl: &Level,
    inner: &Level,
    rows: &[Option<(usize, usize)>],
    width: Option<i32>,
    params: &Params,
) -> Result<Chosen, String> {
    let ic = choose(inner, params)?;
    let cands = list_candidates(
        lvl,
        inner,
        &ic.target,
        ic.rank,
        ic.predicted,
        ic.projected,
        width,
    );
    let (i, array, cands) = first_success(cands, |t| wrap(t, &lvl.values, rows, &ic.array));
    let lossy = !matches!(cands[i].target, Target::Original(_)) && ic.lossy;
    let mut chosen = chosen_from(i, array, cands, lossy);
    chosen.candidates.extend(ic.candidates);
    Ok(chosen)
}

/// A choice made from statistics alone (streaming): the target and its sizes, no array.
pub(crate) struct Pick {
    pub target: Target,
    pub rank: Rank,
    pub predicted: u64,
    pub projected: f64,
    pub lossy: bool,
    /// In the order tried; a list's outer candidates, then its inner ones.
    pub candidates: Vec<Candidate>,
    /// A list's inner choice; its candidates were moved into this pick's list.
    pub inner: Option<Box<Pick>>,
}

pub(crate) fn pick_at(
    i: usize,
    cands: Vec<Candidate>,
    lossy: bool,
    inner: Option<Box<Pick>>,
) -> Pick {
    let c = &cands[i];
    let (target, rank, predicted, projected) = (c.target.clone(), c.rank, c.predicted, c.projected);
    Pick {
        target,
        rank,
        predicted,
        projected,
        lossy,
        candidates: cands,
        inner,
    }
}

/// `choose` with `prove` in place of cast + verify.
pub(crate) fn pick_by_stats(lvl: &Level, params: &Params) -> Result<Pick, String> {
    let (i, (), cands) = first_success(candidates(lvl, params)?, |t| prove(t, lvl));
    let lossy = lossy_by_stats(&cands[i].target, lvl);
    Ok(pick_at(i, cands, lossy, None))
}

/// `choose_list` from statistics: wrapping a proven inner type cannot fail.
pub(crate) fn pick_list_by_stats(
    lvl: &Level,
    inner: &Level,
    width: Option<i32>,
    params: &Params,
) -> Result<Pick, String> {
    let mut ic = pick_by_stats(inner, params)?;
    let cands = list_candidates(
        lvl,
        inner,
        &ic.target,
        ic.rank,
        ic.predicted,
        ic.projected,
        width,
    );
    let (i, (), mut cands) = first_success(cands, |_| Ok(()));
    let lossy = !matches!(cands[i].target, Target::Original(_)) && ic.lossy;
    cands.extend(std::mem::take(&mut ic.candidates));
    Ok(pick_at(i, cands, lossy, Some(Box::new(ic))))
}

#[cfg(test)]
pub(in crate::recommenders::engine) mod tests {
    use super::*;
    use crate::common::ipc_sizes::ipc_body_bytes;
    use crate::common::ipc_sizes::sizes_of;
    use crate::techniques::describe::Described;
    use polars::prelude::PolarsResult;

    use crate::common::arrow_io::export_series;

    /// Describe with the streaming proof statistics on: these tests also build statistics
    /// Levels from the profile (`prove` / `lossy_by_stats` read them).
    pub(in crate::recommenders::engine) fn describe_one(
        s: &Series,
        seed: u64,
    ) -> PolarsResult<Described> {
        crate::techniques::describe::describe_one(s, seed, true)
    }
    use polars::prelude::{CompatLevel, DataType as PT, NamedFrom, Series, TimeUnit as PTimeUnit};

    pub(in crate::recommenders::engine) fn params() -> Params {
        Params {
            seed: 0,
            zstd_level: 1,
            categorical_threshold: 10_000,
            boolean_pairs: vec![("true".into(), "false".into())],
        }
    }

    /// The one-shot choice (cast + verify) and the statistics-only pick for one series.
    fn both(s: Series, p: &Params) -> (Chosen, Pick) {
        let d = describe_one(&s, 0).unwrap();
        let values = export_series(&s, CompatLevel::oldest()).unwrap();
        let size = ipc_body_bytes(values.as_ref(), None).unwrap();
        let lvl = Level::of_values(
            s.dtype(),
            values,
            &d.outer,
            d.n_midnight,
            size,
            d.conclusions(10_000).0.est,
            "",
        )
        .unwrap();
        (choose(&lvl, p).unwrap(), pick_by_stats(&lvl, p).unwrap())
    }

    fn outcomes(c: &[Candidate]) -> Vec<(String, Outcome)> {
        c.iter().map(|c| (c.rule.clone(), c.outcome)).collect()
    }

    #[test]
    fn statistics_choose_like_cast_and_verify() {
        let strs = |v: &[Option<&str>]| Series::new("x".into(), v);
        let mut yes_no = params();
        yes_no.boolean_pairs = vec![("yes".into(), "no".into())];
        let cases: Vec<(Series, Params)> = vec![
            (strs(&[Some("1.50"), Some("2.25"), None]), params()),
            (strs(&[Some("1.5"), Some("2.25")]), params()),
            (strs(&[Some("-0"), Some("5")]), params()),
            (
                strs(&[Some("2024-01-01 10:00:00"), Some("2024-01-02 11:00:00")]),
                params(),
            ),
            (
                strs(&[
                    Some("2024-01-01T10:00:00Z"),
                    Some("2024-01-01T11:00:00+00:00"),
                ]),
                params(),
            ),
            (strs(&[Some("10:00:00.5"), Some("11:00:00")]), params()),
            (strs(&[Some("Yes"), Some("no"), Some("yes")]), yes_no),
            (strs(&[Some("true"), Some("false")]), params()),
            (
                strs(&[Some("1234567890.1"), Some("0.00000012345")]),
                params(),
            ),
            (strs(&[Some("2024-01-01"), Some("2024-02-01")]), params()),
            (
                strs(&[Some("2024-01-01T00:00:00"), Some("2024-02-01T00:00:00")]),
                params(),
            ),
            (strs(&[Some("0"), Some("1"), Some("1")]), params()),
            (Series::new("x".into(), &[0.0f64, -0.0, 1.5]), params()),
            (Series::new("x".into(), &[1.5f64, 2.25]), params()),
            (Series::new("x".into(), &[300i64, -2, 7]), params()),
            (strs(&[Some("-0"), Some("1")]), params()),
            (strs(&[Some("-0"), Some("0")]), params()),
            (
                strs(&[
                    Some("2300-01-01T00:00:00.123456789+01:00"),
                    Some("2024-01-01T00:00:00+02:00"),
                ]),
                params(),
            ),
            (
                Series::new("x".into(), &[86_400_000i64 * 100_000_000, 0])
                    .cast(&PT::Datetime(PTimeUnit::Milliseconds, None))
                    .unwrap(),
                params(),
            ),
        ];
        for (k, (s, p)) in cases.into_iter().enumerate() {
            // Not `{s:?}`: Polars panics formatting a datetime outside chrono's range.
            let case = format!("case {k} ({})", s.dtype());
            let (chosen, pick) = both(s, &p);
            assert_eq!(pick.target, chosen.target, "{case}");
            assert_eq!(pick.lossy, chosen.lossy, "{case}");
            assert_eq!(
                outcomes(&pick.candidates),
                outcomes(&chosen.candidates),
                "{case}"
            );
        }
    }

    #[test]
    fn statistics_reject_a_float_that_underflows() {
        let tiny = format!("0.{}1", "0".repeat(400));
        let (chosen, pick) = both(Series::new("x".into(), &[tiny.as_str(), "1"]), &params());
        assert_eq!(pick.target, chosen.target);
        let f64c = pick
            .candidates
            .iter()
            .find(|c| c.rule == "string→float64")
            .unwrap();
        assert_eq!(f64c.outcome, Outcome::Failed);
        assert!(f64c
            .reason
            .as_deref()
            .unwrap()
            .starts_with("n_f64_roundtrip_fail=1"));
    }

    #[test]
    fn statistics_reject_nanoseconds_out_of_range() {
        let s = Series::new(
            "x".into(),
            &["2300-01-01T00:00:00.123456789", "2024-01-01T00:00:00"],
        );
        let (chosen, pick) = both(s, &params());
        assert_eq!(pick.target, chosen.target);
        let ns = pick
            .candidates
            .iter()
            .find(|c| c.rule == "string→timestamp")
            .unwrap();
        assert_eq!(ns.outcome, Outcome::Failed);
        assert!(ns.reason.as_deref().unwrap().contains("iso_instant"));
    }

    #[test]
    fn list_statistics_choose_like_cast_and_verify() {
        let item = |v: &[i64]| Series::new("".into(), v);
        for s in [
            Series::new("x".into(), &[item(&[1]), item(&[2]), item(&[300])]),
            Series::new("x".into(), &[item(&[1, 2]), item(&[3]), item(&[])]),
        ] {
            let d = describe_one(&s, 0).unwrap();
            let classic = export_series(&s, CompatLevel::oldest()).unwrap();
            let sz = sizes_of(&s, &classic, 1).unwrap();
            let rec = recommend(&s, &classic, &d, d.conclusions(10_000), &sz, &params()).unwrap();
            let inner = d.inner.as_ref().unwrap();
            let (_, child, width) = list_parts(&classic).unwrap();
            let outer = Level::of_values(
                s.dtype(),
                classic.clone(),
                &d.outer,
                None,
                sz[0],
                d.conclusions(10_000).0.est,
                "",
            )
            .unwrap();
            let child_size = ipc_body_bytes(child.as_ref(), None).unwrap();
            let il = Level::of_values(
                inner.values.dtype(),
                child,
                &inner.profile,
                None,
                child_size,
                d.conclusions(10_000).1.unwrap().est,
                "inner: ",
            )
            .unwrap();
            let pick = pick_list_by_stats(&outer, &il, width, &params()).unwrap();
            assert_eq!(pa_name(&pick.target.arrow_type()), rec.arrow_type, "{s:?}");
            assert_eq!(
                outcomes(&pick.candidates),
                outcomes(&rec.candidates),
                "{s:?}"
            );
            assert_eq!(pick.lossy, rec.lossy);
        }
    }
}
