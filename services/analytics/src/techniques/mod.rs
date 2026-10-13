//! The analytical techniques, by the taxonomy in ToDo.md: single-column or pairwise, ordered
//! or not. Each is a stand-alone module with an Arrow-in / Arrow-out kernel. The leaves are
//! re-exported flat: `crate::techniques::describe`, `crate::techniques::hll`, …

pub(crate) mod pairwise;
pub(crate) mod single_column;

pub(crate) use pairwise::ordered::{
    ams_joint_entropy, ari, chi_squared, contingency, joint_entropy,
};
pub(crate) use pairwise::unordered::{bloomfilter, minhash};
pub(crate) use single_column::ordered::runs_test;
pub(crate) use single_column::unordered::{
    ams_entropy, cardinality_estimators, describe, gcd, heavy_keeper, hll, marginal_entropy,
    space_saving, top_k,
};
