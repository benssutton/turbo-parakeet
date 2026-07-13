# Categorical encoding fix — design

**Date:** 2026-07-13
**Status:** Approved (design), pending implementation

## Problem

`encode_series` in [shared.rs](../../../services/analytics/src/shared.rs) encodes
`Categorical`/`Enum` columns as their **physical u32 code cast straight to u64**:

```rust
DataType::Categorical(_, _) | DataType::Enum(_, _) => {
    let phys = series.to_physical_repr();
    phys.u32()?.iter().map(|v| v.map_or((0, true), |x| (x as u64, false))).unzip()
}
```

Physical codes are assigned per-Series in first-seen order. This single line
causes two distinct problems, which are really one root cause:

1. **Cross-frame correctness (Issue #1).** The same string value gets different
   codes in different Series (`"cat_5"` may be code `3` in dataframe A and `17`
   in dataframe B). A bloom filter or MinHash signature built on one frame's
   codes gives wrong answers when queried with another frame's values, even when
   the underlying strings are identical. This breaks the core cross-dataframe
   containment / similarity use case the project exists for.

2. **Hash clustering (Issue #2).** Codes are small sequential integers
   (`0,1,2,…`). Small integers fed through the bloom filter's xxh3-based double
   hashing on a small filter produce correlated bit patterns, inflating the
   empirical false-positive rate (~4× the configured target was observed in
   testing). This is a distribution artefact, not a correctness bug — the
   no-false-negatives guarantee still holds — but it makes categorical FP rates
   untestable against the configured target.

## Root cause

Both problems stem from encoding by raw physical code. Resolving each category
to its **string value** and hashing it the way the `String` branch already does
fixes both at once:

- Same string → same key regardless of code assignment → fixes Issue #1.
- The key becomes a well-distributed 64-bit foldhash output instead of a tiny
  sequential integer → fixes Issue #2 as a side effect.

`foldhash::fast::FixedState` is already the hash function for the `String`,
`List`, `Array`, `Int128`, and `Decimal` arms of `encode_series`, and for band
hashing and permutation coefficients in [minhash.rs](../../../services/analytics/src/minhash.rs).
It avalanches small integers well. After this change, a categorical `"cat_5"`
encodes to the *same* u64 as the string `"cat_5"` — categorical encoding becomes
byte-identical to the `String` branch, which is the correct semantic.

## Decisions

- **Scope: global, in `encode_series`.** The fix lives in the one encoding arm,
  so all four consumers (bloom, minhash, entropy, chi²) inherit it. This was
  chosen over a bloom/minhash-only fix to avoid maintaining two categorical
  encodings and to close the latent correctness gap if entropy/chi² are ever
  used across frames.
- **Resolve in Rust, not Python.** Rust already holds the rev-map; resolving
  inline avoids serializing category mappings across the FFI boundary and keeps
  the Python wrapper ignorant of categorical internals.
- **Per-unique lookup, not per-row string hash.** Categoricals have few uniques,
  so hash each distinct category once into a `Vec<u64>` indexed by code, then map
  each row's code through the table: O(n_unique) hashes + O(n) lookups. This is
  faster than the current per-row code cast's downstream cost and than a naive
  per-row string hash.

## Change

### 1. Core — `encode_series` Categorical/Enum arm ([shared.rs:190-196](../../../services/analytics/src/shared.rs#L190-L196))

```rust
DataType::Categorical(_, _) | DataType::Enum(_, _) => {
    let build_hasher = FoldHashFixed::default();
    let ca = series.categorical()?;
    let rev_map = ca.get_rev_map();
    // Hash each distinct category once; physical codes index into this table.
    let code_hashes: Vec<u64> = (0..rev_map.len() as u32)
        .map(|code| hash_one(&build_hasher, rev_map.get(code)))
        .collect();
    ca.physical()
        .iter()
        .map(|v| v.map_or((0, true), |code| (code_hashes[code as usize], false)))
        .unzip()
}
```

Exact Polars API names (`get_rev_map`, `physical`, `RevMapping::get`/`len`) and
whether `Series::categorical()` accepts an `Enum` dtype are confirmed at build
time. If `Enum` needs separate handling, it is branched out using the same
string-resolution logic against the enum's category list.

### 2. Ripple updates

- **[test_bloom_filter.py](../../../tests/test_bloom_filter.py):**
  - Re-include categorical in `CUSTOM_FP_RATE_COLS` and `CROSS_IMPL_FP_COLS`
    (both were defined solely to route around this bug).
  - Delete the "Note on Categorical FP rate testing" docstring caveat.
  - Add an explicit cross-frame regression test: build a bloom filter on one
    categorical Series, query membership with a **separately constructed**
    categorical Series holding the same string values, assert zero false
    negatives. This exercises the cross-frame path that same-Series tests miss.

- **[CLAUDE.md](../../../CLAUDE.md):** update the cross-cutting note that reads
  "String / list / decimal types route through foldhash → u64" to include
  categorical/enum.

## Validation

1. `maturin develop --release` from `services/analytics/`.
2. Full non-slow suite green, with categorical now **inside** the FP-rate and
   cross-implementation assertions rather than skipped.
3. The new cross-frame categorical regression test passes.

## Risk

Equivalence classes are unchanged: distinct category → distinct key (modulo
negligible foldhash collision, ~6×10⁻¹¹ per pair at 50K rows, same as the
existing String path). Entropy and chi² counts are therefore provably
unaffected — only the bit pattern of each key changes, and neither consumer
compares keys across columns. The only behavioural change is that cross-column
set operations (bloom, minhash) become correct.
