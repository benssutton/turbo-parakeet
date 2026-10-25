//! Exemplar cells: first non-null and first null rows, and the value.

use polars::prelude::{AnyValue, PolarsResult, Series};

use crate::common::ipc_sizes::classic_layout;
use crate::common::text::display_value;
use crate::techniques::describe::Row;

/// A column's exemplar rows and the value at the first non-null one.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Exemplars {
    pub row: Option<u64>,
    pub value: Option<String>,
    pub null_row: Option<u64>,
}

/// Row `i` of `s` as display text (classic Arrow layout, as `min` / `max`).
pub(crate) fn exemplar_value(s: &Series, i: u64) -> PolarsResult<Option<String>> {
    let one = classic_layout(&s.slice(i as i64, 1))?;
    Ok(display_value(one.as_ref()))
}

/// The five exemplar cells in schema order; the `*_id` cells are null here (`with_ids`).
pub(crate) fn exemplar_cells(e: Option<&Exemplars>) -> Row {
    let u = |v: Option<u64>| v.map_or(AnyValue::Null, AnyValue::UInt64);
    match e {
        Some(e) => vec![
            u(e.row),
            AnyValue::Null,
            e.value
                .clone()
                .map_or(AnyValue::Null, |v| AnyValue::StringOwned(v.into())),
            u(e.null_row),
            AnyValue::Null,
        ],
        None => vec![AnyValue::Null; 5],
    }
}
