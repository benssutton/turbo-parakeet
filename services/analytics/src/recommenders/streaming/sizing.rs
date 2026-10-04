//! Original and recommended sizes from statistics: analytic IPC bodies, Polars view layouts, block recasts.

use arrow_array::{new_empty_array, Array, ArrayRef};
use arrow_schema::{DataType as AT, Field};
use polars::prelude::{polars_err, DataType as PT, PolarsResult, Series};

use crate::common::arrow_io::import_array;
use crate::common::ipc_sizes::classic_layout;
use crate::recommenders::engine::{
    body_size, cast_to, list_parts, pa_name, pad, polars_layout, to_polars_layout, validity,
    verify, wrap, Level, Pick, Shape, Target,
};
use crate::recommenders::streaming::partial::{has_int_range, LevelStats, ViewSim};
use crate::recommenders::streaming::reservoir::Block;
use crate::techniques::cardinality_estimators::Estimate;
use crate::techniques::describe::Profile;

/// A Level built from statistics: the rules read its counts, extremes and few distinct
/// values; `values` is empty (only its type is read). Its size is set by the caller.
pub(crate) fn level<'a>(
    dtype: &'a PT,
    classic: &AT,
    p: &'a Profile,
    st: &LevelStats,
    est: Estimate,
    prefix: &'static str,
) -> Level<'a> {
    Level {
        dtype,
        values: new_empty_array(classic),
        p,
        n_rows: st.n,
        n_null: st.n_null,
        // Boolean and Enum extremes also carry integer keys (the value, the category
        // code); the rules read an integer range only for integer-backed numeric and
        // temporal dtypes.
        int_range: has_int_range(dtype).then(|| st.int_range()).flatten(),
        float_range: st.float_range(),
        few_distinct: st.few_distinct(),
        n_midnight: st.n_midnight,
        size_bytes: 0,
        size_note: "",
        est,
        prefix,
        text: Default::default(),
    }
}

/// A Categorical / Enum level's dictionary values as Polars exports them: every
/// category of the dtype's mapping, in id order (`CategoricalMapping::to_arrow`) — for
/// an Enum its categories, for a Categorical every value its Categories object has
/// seen (other columns' included when it is shared, e.g. the global one). Read at
/// `result`: the mapping only grows.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Cats {
    n: f64,
    sum_len: f64,
    /// Polars' view blocks over the categories (the native layout).
    views: u64,
}

pub(crate) fn cats_of(dtype: &PT) -> Option<Cats> {
    let m = match dtype {
        PT::Categorical(_, m) | PT::Enum(_, m) => m,
        _ => return None,
    };
    let mut c = Cats::default();
    let mut views = ViewSim::default();
    for i in 0..m.num_cats_upper_bound() {
        let len = m.cat_to_str(i as _).map_or(0, str::len) as u64;
        c.n += 1.0;
        c.sum_len += len as f64;
        views.push(len);
    }
    c.views = views.bytes();
    Some(c)
}

/// A dictionary's keys: validity + keys of `key`'s width.
pub(crate) fn dictionary_keys(key: &AT, s: &Shape) -> std::result::Result<f64, String> {
    let w = key
        .primitive_width()
        .ok_or_else(|| format!("no predicted size for key {}", pa_name(key)))?;
    Ok(validity(s.n, s.nulls) + pad(s.n * w as f64))
}

/// The classic layout's IPC body (spec §5.3): `body_size`; a Categorical / Enum
/// dictionary (keys + every category, `Cats`); LargeList and FixedSizeList over an
/// analytic inner level (whose shape excludes a null row's slots). Structs have no
/// analytic form.
///
/// Known gap: an Array row that is null but holds values in its slots (hand-built
/// arrays only; Polars and pyarrow leave them empty or null) is counted as null
/// slots, so variable-width values under it are missed.
pub(crate) fn classic_body(
    classic: &AT,
    o: &ViewShape,
    inner: Option<&ViewShape>,
) -> std::result::Result<f64, String> {
    let s = &o.shape;
    let inner = || inner.copied().ok_or_else(|| "no inner level".to_string());
    match classic {
        AT::Dictionary(k, values) => {
            let c = o.cats.ok_or("a dictionary without categories")?;
            let offsets = match **values {
                AT::Utf8 => 4.0,
                AT::LargeUtf8 => 8.0,
                _ => return Err(format!("no predicted size for {}", pa_name(classic))),
            };
            Ok(dictionary_keys(k, s)? + pad(offsets * (c.n + 1.0)) + pad(c.sum_len))
        }
        AT::LargeList(f) => Ok(validity(s.n, s.nulls)
            + pad(8.0 * (s.n + 1.0))
            + classic_body(f.data_type(), &inner()?, None)?),
        AT::FixedSizeList(f, w) => {
            // The child holds w slots per row; a null row's slots count as null.
            let (i, w) = (inner()?, *w as f64);
            let shape = Shape {
                n: s.n * w,
                nulls: i.shape.nulls + s.nulls * w,
                ..i.shape
            };
            Ok(validity(s.n, s.nulls)
                + classic_body(f.data_type(), &ViewShape { shape, ..i }, None)?)
        }
        AT::Struct(_) => Err("no analytic size".into()),
        t => body_size(t, s),
    }
}

/// The original type's uncompressed size: analytic where the classic layout has a
/// closed form (predicted = measured), else the per-batch measured sum.
pub(crate) fn original_size(
    classic: &AT,
    o: &ViewShape,
    inner: Option<&ViewShape>,
    st: &LevelStats,
) -> (u64, &'static str) {
    match classic_body(classic, o, inner) {
        Ok(b) => (b as u64, "analytic"),
        Err(_) => (st.size_bytes, "per-batch sum of"),
    }
}

/// The original type's IPC body in Polars' layout, as one-shot measures the Series
/// Polars imports; None where there is no analytic form (structs, nested lists).
///
/// The model: a freshly built Polars frame (view input), or one zero-copy import of a
/// compact string array (non-view input). A string_view input keeps its own buffers —
/// for a Polars-built frame, blocks of 8 KiB doubling that hold only values over 12
/// bytes (`ViewSim`). A Utf8/Binary input is converted zero-copy (polars-compute
/// `binary_to_binview`): when any value is over 12 bytes the whole values buffer
/// becomes one data buffer, else there is none. A sliced or filtered frame carries
/// arbitrary buffers, which no statistic predicts. A Categorical / Enum exports its
/// keys and every category as views Polars builds (`Cats`).
pub(crate) fn original_polars(
    classic: &AT,
    o: &ViewShape,
    inner: Option<&ViewShape>,
) -> Option<f64> {
    let s = &o.shape;
    let v = validity(s.n, s.nulls);
    Some(match classic {
        AT::Utf8 | AT::LargeUtf8 | AT::Binary | AT::LargeBinary => {
            let data = match (o.views_input, o.all > 0) {
                (true, _) => o.all as f64,
                (false, true) => pad(s.sum_len),
                (false, false) => 0.0,
            };
            v + pad(16.0 * s.n) + data
        }
        AT::Dictionary(k, _) => {
            let c = o.cats?;
            dictionary_keys(k, s).ok()? + pad(16.0 * c.n) + c.views as f64
        }
        AT::LargeList(f) => {
            v + pad(8.0 * (s.n + 1.0)) + original_polars(f.data_type(), inner?, None)?
        }
        AT::FixedSizeList(f, w) => {
            let (i, w) = (inner?, *w as f64);
            let shape = Shape {
                n: s.n * w,
                nulls: i.shape.nulls + s.nulls * w,
                ..i.shape
            };
            v + original_polars(f.data_type(), &ViewShape { shape, ..*i }, None)?
        }
        AT::Struct(_) | AT::List(_) => return None,
        t => body_size(&polars_layout(t, &AT::UInt32), s).ok()?,
    })
}

/// A level's analytic inputs: its shape, Polars' view blocks over all values and
/// over the distinct ones, whether the input held it as views, and a Categorical /
/// Enum level's categories.
#[derive(Clone, Copy)]
pub(crate) struct ViewShape {
    shape: Shape,
    all: u64,
    distinct: u64,
    views_input: bool,
    cats: Option<Cats>,
    /// The per-batch measured Polars-layout size (where there is no closed form) of
    /// the classic layout rebuilt as `to_polars_layout` does.
    rebuilt_polars_bytes: u64,
}

pub(crate) fn view_shape(lvl: &Level, st: &LevelStats, views_input: bool) -> ViewShape {
    ViewShape {
        shape: lvl.shape(),
        all: st.views.as_ref().map_or(0, ViewSim::bytes),
        distinct: st
            .sample
            .as_ref()
            .filter(|d| d.is_exact())
            .map_or(0, |d| d.views.bytes()),
        views_input,
        cats: cats_of(lvl.dtype),
        rebuilt_polars_bytes: st.rebuilt_polars_bytes,
    }
}

/// Uncompressed IPC body of `t` in Polars' layout (Spec B §5.5), from statistics:
/// what `to_polars_layout` + `ipc_body_bytes` measure in one-shot.
pub(crate) fn polars_body(
    t: &Target,
    o: &ViewShape,
    inner: Option<&ViewShape>,
    key: &AT,
) -> std::result::Result<f64, String> {
    let s = &o.shape;
    let v = validity(s.n, s.nulls);
    let inner = || inner.copied().ok_or_else(|| "no inner level".to_string());
    Ok(match t {
        Target::Plain(_) => v + pad(16.0 * s.n) + o.all as f64,
        Target::Dictionary(..) => {
            let w = key.primitive_width().unwrap_or(4) as f64;
            v + pad(s.n * w) + pad(16.0 * s.d) + o.distinct as f64
        }
        Target::Scalar(it) => {
            let i = inner()?;
            let shape = Shape {
                n: s.n,
                nulls: s.nulls + i.shape.nulls,
                ..i.shape
            };
            polars_body(it, &ViewShape { shape, ..i }, None, key)?
        }
        Target::List(it) => v + pad(8.0 * (s.n + 1.0)) + polars_body(it, &inner()?, None, key)?,
        Target::FixedList(it, w) => {
            let (i, w) = (inner()?, *w as f64);
            let shape = Shape {
                n: s.n * w,
                nulls: i.shape.nulls + s.nulls * w,
                ..i.shape
            };
            v + polars_body(it, &ViewShape { shape, ..i }, None, key)?
        }
        // A list's kept inner level, laid out as `to_polars_layout` does: a dictionary
        // takes the Polars key `key` (one-shot widens an Enum's UInt8 keys to it).
        Target::Original(t) => {
            let t = match t {
                AT::Dictionary(_, v) => AT::Dictionary(Box::new(key.clone()), v.clone()),
                t => t.clone(),
            };
            // No closed form (a struct, say): as measured per batch, rebuilt as
            // one-shot's `to_polars_layout` rebuilds the recast array (exact for one
            // batch; a Struct's dictionary fields take Polars' UInt32 keys).
            original_polars(&t, o, inner().ok().as_ref()).unwrap_or(o.rebuilt_polars_bytes as f64)
        }
        t => body_size(&polars_layout(&t.arrow_type(), key), s)?,
    })
}

/// The recommended array has nulls (as one-shot's `logical_null_count() > 0`).
pub(crate) fn nullable(t: &Target, o: &LevelStats, inner: Option<&LevelStats>) -> bool {
    match t {
        Target::Null => o.n > 0,
        Target::Scalar(_) => o.n_null + inner.map_or(0, |i| i.n_null) > 0,
        _ => o.n_null > 0,
    }
}

/// Column `name`'s rows in block `b` as one Series of `dtype` (absent pieces: nulls),
/// with compact buffers: appending keeps each piece's view buffers, so the block is
/// rebuilt as one-shot's Series is built from its input — through its classic layout
/// (a zero-copy import), or, for view input, with views built as Polars builds them.
pub(crate) fn block_series(
    b: &Block,
    name: &str,
    dtype: &PT,
    views_input: (bool, bool),
) -> PolarsResult<Series> {
    let mut out = Series::new_empty(name.into(), dtype);
    for piece in &b.pieces {
        let s = match piece.cols.iter().find(|(f, _)| f.name() == name) {
            Some((f, a)) => {
                let s = import_array(f, a)?;
                // An Array's cast gives null rows' slots a validity buffer: cast only
                // when the type differs (a Categorical's mapping, say).
                if s.dtype() == dtype {
                    s
                } else {
                    s.cast(dtype)?
                }
            }
            None => Series::full_null(name.into(), piece.rows as usize, dtype),
        };
        out.append(&s)?;
    }
    let mut c = classic_layout(&out)?;
    if views_input.0 || views_input.1 {
        // View input keeps its buffers: rebuild them as Polars builds a frame's views.
        c = to_polars_layout(&c, &AT::UInt32).map_err(|e| polars_err!(ComputeError: "{e}"))?;
    }
    let s = import_array(&Field::new(name, c.data_type().clone(), true), &c)?;
    if s.dtype() == dtype {
        Ok(s)
    } else {
        s.cast(dtype)
    }
}

/// The block cast to the pick and verified against itself: the cross-check (spec §5.1
/// step 6). A failure means a statistic was wrong — never a silent fallback.
pub(crate) fn recast(
    pick: &Pick,
    dtype: &PT,
    classic: &ArrayRef,
    p: &Profile,
    inner: Option<(&PT, &Profile)>,
) -> std::result::Result<ArrayRef, String> {
    if let (Some(ip), Some((idt, iprof)), false) = (
        &pick.inner,
        inner,
        matches!(pick.target, Target::Original(_)),
    ) {
        let (rows, child, _) = list_parts(classic)?;
        let il = Level::of_block(idt, child, iprof);
        let ia = cast_to(&ip.target, &il)?;
        verify(&ip.target, &il, &ia)?;
        return wrap(&pick.target, classic, &rows, &ia);
    }
    let lvl = Level::of_block(dtype, classic.clone(), p);
    let a = cast_to(&pick.target, &lvl)?;
    verify(&pick.target, &lvl, &a)?;
    Ok(a)
}
