//! Single-column techniques that ignore row order: Describe, GCD, distinct-count sketches.

pub(crate) mod ams_entropy;
pub(crate) mod cardinality_estimators;
pub(crate) mod describe;
pub(crate) mod distinct_count;
pub(crate) mod gcd;
pub(crate) mod heavy_keeper;
pub(crate) mod hll;
pub(crate) mod marginal_entropy;
pub(crate) mod space_saving;
pub(crate) mod top_k;
