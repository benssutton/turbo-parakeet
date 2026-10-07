//! The recommendation engine: the narrowest value-preserving Arrow type per column — candidate
//! types, cast and verify, choice, Polars layout (spec
//! docs/superpowers/specs/2026-09-26-recommend-technique-design.md). Items are re-exported flat:
//! callers write `crate::recommenders::engine::recommend`.

mod analytic_sizes;
mod candidates;
mod cast;
mod choose;
mod column;
mod polars_layout;
mod top_k;
mod types;
mod units;
mod verify;

pub(crate) use analytic_sizes::*;
pub(crate) use candidates::*;
pub(crate) use cast::*;
pub(crate) use choose::*;
pub(crate) use column::*;
pub(crate) use polars_layout::*;
pub(crate) use top_k::*;
pub(crate) use types::*;
pub(crate) use units::*;
pub(crate) use verify::*;

// What the engine's files share with the layers around them: the text helpers (common) and
// the recommenders' output schema.
use crate::common::text::*;
