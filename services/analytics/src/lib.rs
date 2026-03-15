mod bloomfilter;
//mod minhash;
mod shared;
mod entropy;
mod chi_squared;
mod type_conversion;

use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;