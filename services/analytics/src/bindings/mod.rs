//! The outside of the crate: the language-neutral API and its Python and C bindings.

pub(crate) mod api;
mod capi;
#[cfg(feature = "python")]
mod python;
