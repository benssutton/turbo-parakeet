//mod bloomfilter;
//mod minhash;
mod entropy;

use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;