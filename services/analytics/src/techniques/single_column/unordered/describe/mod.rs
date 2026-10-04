//! Describe: the per-column profile for choosing narrower, more compressible Arrow types
//! (spec docs/superpowers/specs/2026-09-26-describe-technique-design.md). `conclusions` turns
//! the profile into min / max, cardinality estimates and a class.

pub(crate) mod conclusions;
mod extremes;
mod floats;
mod frequencies;
mod profile;
mod scanners;

pub(crate) use extremes::*;
pub(crate) use floats::*;
pub(crate) use frequencies::*;
pub(crate) use profile::*;
pub(crate) use scanners::*;

use polars::prelude::*;
use rayon::prelude::*;

/// `fields()` then the private `input_fields()`, one row per input.
pub(crate) fn describe_columns_impl(
    inputs: &[Series],
    seed: u64,
    threshold: u64,
) -> PolarsResult<Series> {
    let rows: Vec<Row> = inputs
        .par_iter()
        .map(|s| {
            describe_one(s, seed, false).map(|d| {
                let mut row = d.row(threshold);
                row.extend(d.input_row());
                row
            })
        })
        .collect::<PolarsResult<_>>()?;
    let mut schema = fields();
    schema.extend(input_fields());
    assemble("describe", &schema, &rows)
}
