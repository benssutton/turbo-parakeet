mod api;
mod ari;
mod arrow_io;
mod bloomfilter;
mod cardinality_estimators;
mod chi_squared;
mod contingency;
mod describe;
mod entropy;
mod gcd;
mod minhash;
mod python;
mod recommend;
mod shared;
mod sizes;

use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;
