// Without the `python` feature only the C ABI (capi.rs) consumes api.rs, and it binds a
// single entry point so far: the other kernels are unused until they are bound too.
#![cfg_attr(not(feature = "python"), allow(dead_code))]

mod api;
mod ari;
mod arrow_io;
mod bloomfilter;
mod capi;
mod cardinality_estimators;
mod chi_squared;
mod contingency;
mod describe;
mod entropy;
mod gcd;
mod minhash;
mod partial;
#[cfg(feature = "python")]
mod python;
mod recommend;
mod reservoir;
mod shared;
mod sizes;

use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;
