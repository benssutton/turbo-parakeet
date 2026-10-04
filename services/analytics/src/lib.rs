// Without the `python` feature only the C ABI (bindings/capi.rs) consumes the API, and it binds
// the recommenders only: the other kernels are unused until they are bound too.
#![cfg_attr(not(feature = "python"), allow(dead_code))]

mod bindings;
mod common;
mod recommenders;
mod techniques;

use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;
