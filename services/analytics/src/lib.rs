mod bloomfilter;
mod minhash;
mod shared;
mod entropy;
mod chi_squared;
mod contingency;
mod ari;
mod gcd;
mod describe;
mod sizes;
mod cardinality_estimators;
mod recommend;

use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;