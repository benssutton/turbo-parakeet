//! Leaf utilities shared by everything else: Arrow I/O, column encoding and selection, IPC
//! sizes, text forms, errors. Imports nothing from the other layers (enforced by
//! scripts/check_layering.py).

pub(crate) mod arrow_io;
pub(crate) mod encode;
pub(crate) mod error;
pub(crate) mod ipc_sizes;
pub(crate) mod selection;
pub(crate) mod text;

pub(crate) use encode::*;
pub(crate) use selection::*;
