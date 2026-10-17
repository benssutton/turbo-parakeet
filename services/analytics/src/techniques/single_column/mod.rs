//! Techniques that look at one column at a time: `unordered` ignores row order, `ordered`
//! (the runs test, the skip indexes) depends on it.

pub(crate) mod ordered;
pub(crate) mod unordered;
