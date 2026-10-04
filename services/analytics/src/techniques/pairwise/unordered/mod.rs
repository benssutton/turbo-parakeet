//! Pairwise techniques over value sets, where row i of one column need not match row i of the
//! other: membership (Bloom filters) and similarity (MinHash).

pub(crate) mod bloomfilter;
pub(crate) mod minhash;
