//! Error and Result, shared by the bindings and the recommenders.

use std::fmt;

use polars::prelude::PolarsError;

#[derive(Debug, PartialEq)]
pub enum Error {
    /// The caller's input: unknown or duplicate column names, a malformed Bloom
    /// array or invalid Bloom/LSH parameters, an Arrow type no kernel accepts, or
    /// a column of the wrong type for the kernel.
    InvalidInput(String),
    /// Any other failure inside a kernel.
    Compute(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidInput(m) | Error::Compute(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn compute(e: PolarsError) -> Error {
    let msg = e.to_string();
    match e {
        PolarsError::ColumnNotFound(_)
        | PolarsError::SchemaMismatch(_)
        | PolarsError::InvalidOperation(_)
        | PolarsError::ShapeMismatch(_) => Error::InvalidInput(msg),
        _ => Error::Compute(msg),
    }
}
