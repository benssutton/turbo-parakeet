//! Leaf utilities shared by everything else: Arrow I/O, column encoding, IPC sizes. Imports
//! nothing from the other layers (enforced by scripts/check_layering.py).

pub(crate) mod arrow_io;
pub(crate) mod encode;
pub(crate) mod error;
pub(crate) mod ipc_sizes;
pub(crate) mod text;
