//! The Polars layout of a recommended array (spec 2026-09-26 §5.5).

use super::*;
use arrow_array::builder::make_view;
use arrow_array::cast::AsArray;
use arrow_array::{
    Array, ArrayRef, BinaryViewArray, FixedSizeListArray, LargeListArray, StringViewArray,
    StructArray,
};
use arrow_buffer::{Buffer, ScalarBuffer};
use arrow_schema::{DataType as AT, Field as AField, Fields};
use std::sync::Arc;

// ── Polars layout of the result (Spec B §5.5) ───────────────────────────────

/// `a` converted to the Arrow layout Polars exports for it (`polars_layout`),
/// recursing through lists and structs; `key` is a dictionary's Polars key.
pub(crate) fn to_polars_layout(a: &ArrayRef, key: &AT) -> Result<ArrayRef, String> {
    let item = |c: &ArrayRef| Arc::new(AField::new("item", c.data_type().clone(), true));
    match a.data_type() {
        AT::Dictionary(..) => {
            let keyed = arrow_cast(
                a.as_ref(),
                &AT::Dictionary(Box::new(key.clone()), Box::new(AT::Utf8)),
            )?;
            let d = keyed.as_any_dictionary();
            Ok(d.with_values(polars_views(d.values().as_ref(), &AT::Utf8View)?))
        }
        AT::List(f) | AT::LargeList(f) => {
            let large = arrow_cast(a.as_ref(), &AT::LargeList(f.clone()))?;
            let l = large.as_list::<i64>();
            let child = to_polars_layout(l.values(), key)?;
            LargeListArray::try_new(item(&child), l.offsets().clone(), child, l.nulls().cloned())
                .map(|x| Arc::new(x) as ArrayRef)
                .map_err(|e| e.to_string())
        }
        AT::FixedSizeList(_, w) => {
            let f = a.as_fixed_size_list();
            let child = to_polars_layout(f.values(), key)?;
            FixedSizeListArray::try_new(item(&child), *w, child, f.nulls().cloned())
                .map(|x| Arc::new(x) as ArrayRef)
                .map_err(|e| e.to_string())
        }
        AT::Struct(fields) => {
            let s = a.as_struct();
            let cols = s
                .columns()
                .iter()
                .map(|c| to_polars_layout(c, key))
                .collect::<Result<Vec<_>, _>>()?;
            let fields: Fields = fields
                .iter()
                .zip(&cols)
                .map(|(f, c)| {
                    Arc::new(AField::new(
                        f.name(),
                        c.data_type().clone(),
                        f.is_nullable(),
                    ))
                })
                .collect();
            StructArray::try_new(fields, cols, s.nulls().cloned())
                .map(|x| Arc::new(x) as ArrayRef)
                .map_err(|e| e.to_string())
        }
        AT::Utf8 | AT::LargeUtf8 => polars_views(a.as_ref(), &AT::Utf8View),
        AT::Binary | AT::LargeBinary => polars_views(a.as_ref(), &AT::BinaryView),
        t => arrow_cast(a.as_ref(), &polars_layout(t, key)),
    }
}

/// Polars' view-array data blocks (polars-arrow `binview`): the first holds 8 KiB,
/// each next one doubles (capped at 16 MiB) and grows to fit a larger value.
pub(crate) const VIEW_BLOCK: usize = 8 * 1024;
pub(crate) const VIEW_MAX_BLOCK: usize = 16 * 1024 * 1024;

/// Strings / binaries as the Utf8View / BinaryView array Polars builds from them
/// (`MutableBinaryViewArray::push_value_into_buffer`): values of ≤ 12 bytes inline
/// in the view, longer ones appended to the current block, never straddling blocks —
/// so the measured sizes are Polars' own (arrow-cast would reuse the source buffer).
pub(crate) fn polars_views(a: &dyn Array, to: &AT) -> Result<ArrayRef, String> {
    let bytes = arrow_cast(a, &AT::LargeBinary)?;
    let bytes = bytes.as_binary::<i64>();
    let mut views = Vec::with_capacity(bytes.len());
    let (mut blocks, mut current, mut capacity) = (Vec::<Buffer>::new(), Vec::<u8>::new(), 0usize);
    for v in bytes.iter() {
        let Some(v) = v else {
            views.push(0u128);
            continue;
        };
        if v.len() <= 12 {
            views.push(make_view(v, 0, 0));
            continue;
        }
        if capacity < current.len() + v.len() {
            if !current.is_empty() {
                blocks.push(Buffer::from_vec(std::mem::take(&mut current)));
            }
            capacity = (capacity * 2)
                .clamp(VIEW_BLOCK, VIEW_MAX_BLOCK)
                .max(v.len());
        }
        views.push(make_view(v, blocks.len() as u32, current.len() as u32));
        current.extend_from_slice(v);
    }
    if !current.is_empty() {
        blocks.push(Buffer::from_vec(current));
    }
    let (views, nulls) = (ScalarBuffer::from(views), bytes.nulls().cloned());
    let e = |e: arrow_schema::ArrowError| e.to_string();
    match to {
        AT::Utf8View => StringViewArray::try_new(views, blocks, nulls)
            .map(|x| Arc::new(x) as ArrayRef)
            .map_err(e),
        _ => BinaryViewArray::try_new(views, blocks, nulls)
            .map(|x| Arc::new(x) as ArrayRef)
            .map_err(e),
    }
}
