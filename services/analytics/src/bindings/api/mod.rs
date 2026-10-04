//! The language-neutral core: every entry point takes an Arrow RecordBatch plus
//! plain parameters and returns an Arrow RecordBatch (Bloom: bytes). No pyo3 and no
//! Polars type appears in any signature — bindings (bindings/python.rs; later Java / C) wrap
//! exactly this module. Kernels compute on Polars Series behind arrow_io.
//!
//! Input batches must be valid Arrow: the entry points do not re-check them (an O(n)
//! pass over every value). A RecordBatch built with arrow-rs's safe constructors is
//! valid by construction; one imported through the C Data Interface is not, so callers
//! import with `arrow_io::read_stream` / `CheckedReader` (bindings/python.rs and bindings/capi.rs do), or
//! check with `arrow_io::validate_batch`, before calling in — arrow-rs and Polars may
//! panic or read out of bounds on a malformed batch.

mod recommenders;
mod techniques;

pub use crate::common::error::{Error, Result};
pub use recommenders::*;
#[allow(unused_imports)] // used by bindings/python.rs only
pub use techniques::*;

use crate::common::error::compute;

use std::collections::HashSet;

use arrow_array::RecordBatch;
use polars::prelude::{PolarsResult, Series};

use crate::common::arrow_io::{export_struct, import_batch};
use crate::common::PairwiseKwargs;

/// `batch`'s columns as Series. `batch` must be valid Arrow (see the module docs).
fn columns(batch: &RecordBatch) -> Result<Vec<Series>> {
    let schema = batch.schema();
    let mut seen: HashSet<&str> = HashSet::new();
    for f in schema.fields().iter() {
        if !seen.insert(f.name().as_str()) {
            return Err(Error::InvalidInput(format!(
                "duplicate column {:?}",
                f.name()
            )));
        }
    }
    import_batch(batch).map_err(|e| Error::InvalidInput(e.to_string()))
}

fn table(out: PolarsResult<Series>) -> Result<RecordBatch> {
    out.and_then(|s| export_struct(&s)).map_err(compute)
}

fn check_names<'a>(batch: &RecordBatch, names: impl IntoIterator<Item = &'a String>) -> Result<()> {
    let schema = batch.schema();
    let known: HashSet<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    match names.into_iter().find(|n| !known.contains(n.as_str())) {
        Some(n) => Err(Error::InvalidInput(format!("unknown column {n:?}"))),
        None => Ok(()),
    }
}

fn pairwise(batch: &RecordBatch, pairs: Option<&[(String, String)]>) -> Result<PairwiseKwargs> {
    let Some(pairs) = pairs else {
        return Ok(PairwiseKwargs { pairs: None });
    };
    check_names(batch, pairs.iter().flat_map(|(a, b)| [a, b]))?;
    Ok(PairwiseKwargs {
        pairs: Some(
            pairs
                .iter()
                .map(|(a, b)| vec![a.clone(), b.clone()])
                .collect(),
        ),
    })
}
