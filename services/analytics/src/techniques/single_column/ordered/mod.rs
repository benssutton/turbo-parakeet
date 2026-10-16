//! Single-column techniques where row order matters: the runs test and the skip-index techniques
//! (ClickHouse minmax / set).

pub(crate) mod runs_test;
pub(crate) mod skip_index;
