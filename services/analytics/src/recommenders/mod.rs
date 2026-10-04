//! The recommenders: the narrowest value-preserving Arrow type per column, from one frame
//! (`oneshot`) or from batches over time (`streaming`), on a shared `engine`.

pub(crate) mod engine;
pub(crate) mod oneshot;
pub(crate) mod streaming;
