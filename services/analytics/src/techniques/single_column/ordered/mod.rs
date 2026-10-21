//! Single-column techniques where row order matters: the runs test and the skip-index techniques
//! (ClickHouse minmax / set), plus the delta technique.

pub(crate) mod delta;
pub(crate) mod runs_test;
pub(crate) mod skip_index;
