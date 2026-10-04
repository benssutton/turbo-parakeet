//! Single-column techniques that ignore row order: Describe, GCD, distinct-count sketches.

pub(crate) mod cardinality_estimators;
pub(crate) mod describe;
pub(crate) mod gcd;
pub(crate) mod hll;
