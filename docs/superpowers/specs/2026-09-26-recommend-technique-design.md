# Recommend Technique — Design (Spec B)

> See also: [the streaming recommender](2026-09-29-streaming-recommender-design.md), which recommends from batches added over time.

> **Superseded in part** by [the Arrow FFI interface](2026-09-27-arrow-ffi-interface-design.md): Int128 / UInt128 columns are now ineligible (so the Int128 rules, the "over 38 digits → null" GCD case and null sizes / recommendations for nested Int128 no longer apply), and the Rust entry points are `api.rs` functions behind the `analytics.analytics` binding, not Polars plugins.

**Date:** 2026-09-26
**Status:** Implemented (branch `recommend-technique`)
**Builds on:** Spec A, `docs/superpowers/specs/2026-09-26-describe-technique-design.md` (the Describe technique).

## 1. Purpose

`recommend` is a per-column technique that recommends, for every column, the narrowest **value-preserving** Arrow type (and, for strings, whether to dictionary-encode it), then **proves** the recommendation: it casts the column, verifies the round trip row by row and measures the recast column's Arrow IPC size, uncompressed and with ZSTD. It also reports the equivalent **Polars** type and its measured sizes, since Polars is a specific Arrow layout with subtly different types and sizes.

Recasting production data is invasive, so **transparency** governs the design: every statistic a rule reads is a Describe metric visible in the output (implemented and agreement-tested in all three Describe implementations), and every candidate type considered is listed with the rule that produced it, the metric values it tested, its predicted size and its outcome.

### Success criteria

- `RecommendRust(**params).add(frames).result()` returns Describe's full table plus the recommendation columns (§6), on the uniform technique contract.
- For every successful candidate, the predicted uncompressed size equals the measured size.
- Recast sizes agree with independent oracles: pyarrow casting to `rec_arrow_type`, and Polars casting to `rec_polars_type`, each measured with `_sizes.py`.
- The six new Describe metrics (§3) agree across `DescribeRust`, `DescribeDataFusion` and `DescribePolars` ★.

### Decisions made during brainstorming

| Decision | Choice |
|---|---|
| Output | Report only: recommendation, verification outcome and sizes. The recast data is discarded after measuring |
| Architecture | All in Rust: one plugin entry `describe_and_recommend` computes Describe's metrics, estimates, recommends, casts, verifies and measures |
| Rust layout | Flatten `src/describe/` into `src/describe.rs`; move `sizes.rs` to `src/`; new `src/cardinality_estimators.rs` and `src/recommend.rs` |
| Estimators | Ported to Rust for the recommender; `estimators.py` stays as the Python reference; the two are agreement-tested |
| Choosing a type | Candidates come from the type hierarchy; each has an analytically predicted size; they are tried **smallest projected size first, ties broken by hierarchy rank**; the first to cast and verify is chosen; on failure, the next candidate is tried. The original type is a candidate like any other, ranked last among equal sizes, and cannot fail |
| Sampling | Integer/decimal widths from observed values. Dictionary key width and the dictionary-vs-plain choice from population cardinality: `est_high` where an interval exists, else `est_cardinality`. Nullability from observed nulls |
| Transparency | Any statistic a rule reads is a Describe metric in all three implementations; every candidate is reported with evidence |
| Arrow-native | `recommend.rs` and `sizes.rs` operate on **arrow-rs** arrays; a thin adapter converts Polars Series zero-copy via the C Data Interface. The Python↔Rust boundary moving to Arrow tables later changes only the adapter |
| Polars | A layout profile of Arrow: `polars_layout(arrow_type)` maps a recommended Arrow type to the Arrow type Polars would hold; sizes are measured on that layout |
| Leading zeros | An integer-looking string with a leading zero keeps the column a String (see Spec A §5.1 rationale). Trailing zeros after a decimal point may be lost (`lossy_formatting`) |
| Varying offsets | Arrow's canonical extension `arrow.timestamp_with_offset` |
| Benchmark | Describe's benchmark covers the new metrics; the recommender reports parallel speedup only (no Python implementation to compare against) |

## 2. Layout

```
services/analytics/
├── src/
│   ├── describe.rs                # merged from describe/{frequency,numeric,patterns,range,mod}.rs;
│   │                              #   profile() → typed Profile; entry describe_columns
│   ├── sizes.rs                   # moved from describe/; ported to arrow-rs ArrayData; entry column_sizes
│   ├── cardinality_estimators.rs  # Chao1, Schnabel, Duj1, intervals, selection rule
│   ├── recommend.rs               # Arrow-native: rules → candidates → cast → verify → measure; polars_layout();
│   │                              #   entry describe_and_recommend
│   ├── gcd.rs                     # kernel reused by describe.rs for the `gcd` metric
│   └── lib.rs                     # mod describe; mod sizes; mod cardinality_estimators; mod recommend;
└── analytics/
    ├── _plugin.py                 # + describe_and_recommend wrapper
    ├── describe/                  # + new metrics (base, polars, datafusion, rust); size_polars_bytes redefined
    └── recommend/
        ├── __init__.py            # Recommend, RecommendRust; REFERENCE = IMPLEMENTATIONS[0] = "RecommendRust"
        ├── base.py                # Recommend(Describe): METRICS = Describe.METRICS + REC_METRICS; boolean_pairs
        └── rust.py                # RecommendRust
```

`src/describe/` (including `mod.rs`) is removed. `describe.rs` is ≈1,000 lines including its unit tests — comparable to `entropy.rs`. The four former submodules become sections of one file; `pub(crate)` plumbing between them disappears. `profile()` returns a typed `Profile { freq, range, floats, strings, … }`, converted to the output row only at the end, so the recommender reads fields by name.

**Arrow crates:** add `arrow-array`, `arrow-buffer`, `arrow-data`, `arrow-schema`, `arrow-cast` (a version with Decimal32/Decimal64, pinned during implementation; no `pyarrow` feature, so no pyo3 conflict). The Describe metric kernels stay on Polars in this spec; only `sizes.rs` and `recommend.rs` are Arrow-native. Series cross into arrow-rs zero-copy: `Series::to_arrow(CompatLevel)` → polars-arrow C Data Interface export → arrow-rs `from_ffi`. The IPC ZSTD framing stays our own (arrow-rs's IPC writer does not expose the ZSTD level).

## 3. New Describe metrics

Added to `Describe.METRICS` and computed by **all three** Describe implementations, on outer values and (with the `inner_` prefix) on inner values. Agreement is exact.

| Metric | Type | Applies to | Definition |
|---|---|---|---|
| `gcd` | Decimal(38, 0) | Int/UInt 8–64, Int128, Decimal, Date, Datetime, Duration, Time | As the Gcd technique: GCD of the magnitudes of the physical values (Decimal → unscaled, Date → days, Datetime/Duration → time unit, Time → ns); nulls skipped; all-null / all-zero / zero-row → 0; over 38 digits → null. Else null. Rust reuses `gcd.rs` |
| `sum_len` | UInt64 | String, Categorical, Enum (UTF-8 bytes), Binary | Total bytes over non-null values. Else null |
| `sum_len_unique` | UInt64 | as `sum_len` | Total bytes over distinct non-null values (Rust: at each key's `first_idx` from the frequency map). Else null |
| `iso_max_sig_frac_digits` | UInt32 | String, Categorical, Enum | Max fractional-second digits over ISO times and datetimes (both kinds) **after removing trailing zeros** (`"10:00:00.120"` → 2, `"10:00:00.000"` → 0); null if there are none |
| `numeric_min_frac_digits` | UInt32 | String, Categorical, Enum | Min fraction digits over the `n_numeric` values after removing trailing zeros (integers count 0; `"1.50"` → 1). With `numeric_max_frac_digits` it shows whether decimal places vary. Null if `n_numeric = 0` |
| `numeric_max_sig_digits` | UInt32 | String, Categorical, Enum | Max significant digits of a single `n_numeric` value: all digits after removing leading zeros (across the dot) and trailing fraction zeros (`"0.00120"` → 2, `"1200"` → 4, `"12.50"` → 3, `"0"` → 0). Decides whether a value can round-trip through Float32 (≤ 6) / Float64 (≤ 15). Null if `n_numeric = 0` |

Placement: `gcd`, `sum_len`, `sum_len_unique` in group A after `max_len`; `iso_max_sig_frac_digits` in group C after `iso_max_frac_digits`; `numeric_min_frac_digits` and `numeric_max_sig_digits` in group C after `numeric_max_frac_digits`.

Polars: `gcd` via the physical values (as `GcdMath`); `sum_len` = `str.len_bytes().sum()` / `bin.size()`; `sum_len_unique` on `unique()`; `iso_max_sig_frac_digits` from the captured fraction with `str.strip_chars_end("0")`. DataFusion: SQL where exact (`SUM(octet_length(...))`, over `DISTINCT` for unique), else pyarrow in the class, listed in its docstring (Spec A §5.3).

**Redefinition:** `size_polars_bytes` becomes the **uncompressed** IPC body of the column in Polars' native layout (`CompatLevel.newest()`), measured like `size_polars_zstd_bytes`. It was `Series.estimated_size()`. Original and recommended Polars sizes are then measured identically, and `DescribeRust._compute` no longer overrides the plugin's value in Python.

## 4. Step 1 — type rules

`n = n_rows − n_null`. min/max are the values at `argmin`/`argmax`. Rules apply to the column's values; for List/Array the same rules apply to the inner values through the `inner_*` metrics, and the outer List is kept (§4.4). A rule is **value-preserving on the observed data** — the original value is reconstructible from the narrower type.

### 4.1 Candidate set and order

Each rule emits candidates. Every candidate has a predicted frame size (§5.1) and a projected population size (§5.3). Candidates are tried in ascending projected population size; ties are broken by **hierarchy rank**:

Null < Boolean < UInt < Int < Decimal < Float < Date32 < Time32/Time64 < Timestamp < Duration < timestamp_with_offset < Dictionary < Utf8/Binary < List < original (last among equal sizes).

Candidates rejected by a gate (§5.2) are listed **last**, after every candidate that can be tried. A list column reports its outer candidates (scalar / list / array / original) followed by its inner level's, whose rules are prefixed `inner: ` — so it shows two `chosen` entries, the outer choice and the inner choice.

Within a signedness, only the narrowest fitting width is emitted (a wider one cannot succeed where the narrowest fails). The **original** Arrow type (Spec A's classic layout, `CompatLevel.oldest()`) is always a candidate, ordered by its size like the others, and cannot fail — so a candidate larger than the original is never tried (e.g. a Float64 column needing Decimal128 keeps Float64).

`rec_nullable` = the recommended array has nulls (the Arrow field's `nullable` flag; the IPC layout already omits the validity buffer without nulls). Usually `n_null > 0`, but a list → scalar recast turns `[null]` into a null row, so it is read off the recast array rather than the source.

### 4.2 Rules by source type

| Source | Candidates |
|---|---|
| any, `n_rows > 0` and `n_null == n_rows` | Null |
| Integer (Int/UInt 8–64, Int128) | Boolean if `0 ≤ min` and `max ≤ 1`; narrowest UInt if `min ≥ 0`; narrowest Int; Decimal128(p, 0) if outside the 64-bit ranges (p = digits of max(\|min\|, \|max\|) ≤ 38) |
| Decimal(p, s) | Let `k` = trailing decimal zeros of `gcd` (capped at `s`; `gcd = 0` → `k = s`). New scale `s' = s − k`, precision `p' = p_obs − k` where `p_obs` = digits of max(\|min\|, \|max\|) unscaled. If `s' = 0`: the Integer rules on the scaled values. Else Decimal32 if `p' ≤ 9`, Decimal64 if `≤ 18`, Decimal128 if `≤ 38` |
| Float32 / Float64 | If `n_nan = n_inf = 0` and `n_fractional = 0`: the Integer rules on min/max (including Decimal128(p, 0) beyond the 64-bit ranges). If `n_nan = n_inf = 0` and `n_fractional > 0`: Decimal(p, s) with `s = max_frac_digits`, `p = int_digits + s`, where `int_digits` = digits of ⌊max(\|min\|, \|max\|)⌋ (0 for values below 1), width by `p` as above, only if `p ≤ 38`. Float32 if the source is Float64 and `n_f32_inexact = 0` |
| Datetime (naive) | Date32 if `n_midnight = n`. Timestamp in the coarsest unit u ∈ {s, ms, µs, ns} dividing `gcd` (in the column's unit; `gcd = 0` → s), only when u differs from the column's unit (the same unit is the original) |
| Datetime (tz-aware) | Timestamp(u, same tz), u as above. Never Date32 (the local date needs the zone to be reconstructed) |
| Duration | Duration(u), u as above, only when u changes |
| Time | Time32(s) if 10⁹ \| `gcd`; Time32(ms) if 10⁶ \| `gcd`; Time64(µs) if 10³ \| `gcd`; else Time64(ns); only when the unit changes |
| String / Categorical / Enum | §4.3 |
| Binary | Binary (32-bit offsets) if `sum_len < 2³¹` |
| List / LargeList / Array | §4.4 |
| Boolean, Date, Struct | original only |

`"-0.0"` / `-0.0` becoming `0` counts as equal and sets `rec_lossy_formatting`.

### 4.3 String / Categorical / Enum

First matching row wins (then step 2 applies to whatever is still a string):

| Condition | Candidates |
|---|---|
| `n_unique ≤ 5` (so top-5 lists every distinct value) and every distinct value matches one `boolean_pairs` entry, case-insensitive (`"Y"`, `"y"`, `"n"` match `("y", "n")`) | Boolean (first element of the pair → true) |
| `n_numeric_int = n`, `n_leading_zero = 0`, `numeric_int_min/max` not null | Integer rules on `numeric_int_min/max` |
| `n_numeric = n`, `n_leading_zero = 0` | Decimal(p, s): `s = numeric_max_frac_digits`, `p = numeric_max_int_digits + s`, width by `p`, only if `p ≤ 38`. **Plus**, when decimal places vary (`numeric_min_frac_digits < numeric_max_frac_digits`) and `p > 18` (no Decimal64 fits): Float32 if `numeric_max_sig_digits ≤ 6`, Float64 if `≤ 15`. Being smaller, the Float is tried before Decimal128, which remains the fallback if the Float fails verification |
| `n_iso_date = n` | Date32 |
| `n_iso_time = n` | Time32/Time64 with unit from `iso_max_sig_frac_digits`: 0 → s, ≤ 3 → ms, ≤ 6 → µs, else ns |
| `n_iso_datetime = n` | Date32 if `iso_n_midnight = n`; else Timestamp(unit from `iso_max_sig_frac_digits`, no tz) |
| `n_iso_datetime_tz = n`, `iso_n_offsets = 1` | Timestamp(unit, that fixed offset: `"+05:00"`; zero offset → `"UTC"`) |
| `n_iso_datetime_tz = n`, `iso_n_offsets > 1` | `arrow.timestamp_with_offset` (§4.5) |
| always | Utf8 (32-bit offsets) if `sum_len < 2³¹` (for Categorical/Enum sources this is the "drop the dictionary" candidate) |

**Leading zeros** (comment required in `recommend.rs`, as in Spec A §5.1): an integer-looking string with a leading zero (`"007"`) must stay a String — identifiers such as UUID fragments, account numbers or zip codes can be all digits with significant leading zeros. A value with a decimal point (`"007.50"`) is unlikely to be an identifier, so only numeric equivalence matters for it; differing leading or trailing zeros set `rec_lossy_formatting`.

Strings become Float only through the varying-decimal-places rule above: a column such as `1234567890.1` and `0.00000012345` needs Decimal128(21, 11), yet each value has ≤ 15 significant digits and is exact in Float64 at half the width. Float verification is a canonical comparison of the text with arrow-cast's rendering of the Float (significant digits and decimal point, independent of formatting: `"1.50"` ≡ `"1.5"`, `"0.00000012345"` ≡ `"1.2345e-7"`); `rec_lossy_formatting` is set when the text changes. Exponent forms are not numeric (Spec A §4.3). A Timestamp(ns) outside 1677–2262 fails its cast; the next candidate is the string itself (a coarser unit would lose digits).

### 4.4 Lists

| Condition | Candidates |
|---|---|
| `min_len = max_len = 1` (over non-null lists), not (`n_null > 0` and `inner_n_null > 0`), and the kept inner type is not itself List / Array / Struct | Scalar: the inner rules applied to the inner values (a null list becomes a null scalar) |
| otherwise | List(x) with x from the inner rules, using List (32-bit offsets) when `inner_n_values < 2³¹`; Array keeps its fixed size, and Array(x, w) is a candidate only when the inner type changes (else it is the original) |

If both null lists and null elements occur, a null list and `[null]` would collide, so the column stays a list. Only one nesting level is analysed (Describe's inner values); deeper levels keep their types.

The inner values are exactly Describe's `flatten` (null lists dropped, then the values of the non-empty lists), so the inner profile's indices point into them. A null list may still span values behind its offsets (common after `pl.when(mask).then(list).otherwise(None)`): those values are not inner values — the child is compacted to the valid rows' ranges (a `take`) and the recast list rebuilds its offsets from row lengths, a null row becoming empty. Array columns compact the same way (the w slots of each valid row) and re-expand a null row to w null slots.

### 4.5 Varying offsets — `arrow.timestamp_with_offset`

Arrow's canonical extension type: storage `Struct{timestamp: Timestamp(u, "UTC") not null, offset_minutes: Int16 not null}`, field metadata `ARROW:extension:name = arrow.timestamp_with_offset`. Nulls live at struct level. The unit comes from `iso_max_sig_frac_digits`; offsets and UTC instants come from `describe.rs`'s ISO scanner (Polars' `%z` parsing discards the offset). `rec_arrow_type` reports the storage type; `rule` names the extension.

## 5. Step 2, sizes, and the cast loop

### 5.1 Predicted frame size (uncompressed Arrow IPC body)

`pad(x)` = x rounded up to 8; `N = n_rows`; `V = pad(⌈N/8⌉)` if `n_null > 0` else 0; `d = n_unique`.

| Target | Predicted bytes |
|---|---|
| Null | 0 |
| Boolean | V + pad(⌈N/8⌉) |
| fixed width w (ints, Decimal32/64/128, floats, Date32, Time32/64, Timestamp, Duration) | V + pad(N·w) |
| Utf8 / Binary | V + pad(4(N+1)) + pad(`sum_len`) |
| LargeUtf8 / LargeBinary | V + pad(8(N+1)) + pad(`sum_len`) |
| Dictionary<k, Utf8> | V + pad(N·k) + pad(4(d+1)) + pad(`sum_len_unique`) |
| timestamp_with_offset | V + pad(8N) + pad(2N) |
| List(x) / LargeList(x) | V + pad(4(N+1)) or pad(8(N+1)) + predicted(x) over the inner values (`inner_n_values`, `inner_n_null`) |
| FixedSizeList(x, w) | V + predicted(x) over n·w inner slots, whose nulls include the w slots of every null row (the child keeps them; Polars exports them as null) |

These mirror `sizes.rs` exactly, so **predicted = measured** for every successful candidate (asserted in tests; a mismatch is a bug). A type with no analytic size (none of the rules emits one) makes its candidate `failed` with the reason — never a wrong prediction.

### 5.2 Step 2 — dictionary encoding

Applies to string results (outer, or inner values of a List) and to Categorical/Enum sources (already dictionaries: step 2 narrows the key or drops the dictionary).

- Cardinality `c` = `est_high` if not null, else `est_cardinality` (inner: `inner_est_*`), floored at `n_unique` (an estimate never claims fewer distinct values than were observed). The evidence names the estimator (`method`) and `est_low` when present.
- Gate: `c ≤ categorical_threshold`; failing the gate the candidate is listed with outcome `rejected`.
- Key width `k`: UInt8 if `c ≤ 256`, UInt16 if `c ≤ 65,536`, else UInt32.
- The dictionary candidate joins the candidate list and is ordered by projected population size like every other candidate (§4.1).

### 5.3 Projected population size

`P` = `population_rows` for the frame, else `n_rows`; `r = P / n_rows`. Row-proportional terms scale by `r` (validity, fixed-width values, offsets, `sum_len`); for a dictionary, its offsets use `c` in place of `d` and its values scale `sum_len_unique · c / d`. Without `population_rows`, `r = 1` and, when the estimate is exact, `c = d`, so projection = prediction.

### 5.4 The loop (per column, in `recommend.rs`)

1. **Candidates** from §4 and §5.2, ordered per §4.1.
2. **Cast** with `arrow-cast` using `safe: false` (in arrow-rs `safe: true` turns failures into nulls; `false` makes them errors), with hand-written paths where exactness demands it:
   - String sources: every target is built from `describe.rs`'s exact parsers (`parse_decimal`, `parse_iso`), never from `arrow-cast`'s string parsing; a value that does not fit (unit remainder, i64 or precision overflow) fails the cast.
   - Float → Decimal: the exact decimal digits of the shortest round-trip representation (`ryu`), never multiply-and-round.
   - Boolean from `boolean_pairs`; `timestamp_with_offset`; Timestamp with a fixed-offset zone: built from the ISO scanner.
   - Any unparsable or out-of-range value fails the cast (reason: first failing row and value).
3. **Verify** row by row against an **exact reference** of the original, with nulls matched:
   - numeric / temporal sources: the recast values cast back to the source type equal the originals (floats: `f64::from_str(decimal_text)` equals the original bits; `-0.0 = 0.0`);
   - string sources: the recast values are rendered to text by `arrow-cast` and compared with the original text by value — canonical decimal digits for numbers, `parse_iso` components (days, time-of-day ns, UTC instant, offset) for temporals — so the check runs through code independent of the parser that built them;
   - dictionary: decoded values equal the originals;
   - list → scalar: verified structurally — the inner values are verified as above, and the scalar column is a `take` of each row's only item (a null list → a null row).
   `rec_lossy_formatting` = some recast value, cast to Utf8, differs textually from the original's text (always false for non-string sources except `-0.0`).
4. **Measure** with `sizes.rs`: plain and ZSTD at `zstd_level` on the recast array.
5. On failure, record `outcome = failed` with the reason and try the next candidate; untried candidates are `not_tried`.
6. **Polars layout**: cast the chosen array to `polars_layout(rec_arrow_type)` (§5.5), verify (widening only, so it cannot fail in practice), measure plain and ZSTD.

A result whose ZSTD size exceeds the original's is still returned (the hierarchy governs), with both sizes visible.

When the original is chosen, `rec_polars_type` is the `dtype` descriptor and its Polars sizes are Describe's `size_polars_bytes` / `size_polars_zstd_bytes`. Columns whose sizes are null (List/Array/Struct nesting Int128, which pyarrow cannot import) get null recommendation columns.

### 5.5 `polars_layout` — Arrow type → Polars type → Polars' Arrow layout

Checked against Polars 1.41 / pyarrow 24.

| Recommended Arrow type | `rec_polars_type` | Measured layout (`CompatLevel.newest()`) |
|---|---|---|
| Int/UInt 8–64, Float32/64, Boolean, Date32, Null | same | same |
| Decimal32/64/128(p, s) | Decimal(p, s) | decimal128(p, s) |
| Timestamp(s, tz) | Datetime(ms, tz) | timestamp(ms, tz) |
| Timestamp(ms/µs/ns, tz) | Datetime(same, tz) | same |
| Time32(s/ms), Time64(µs/ns) | Time | time64(ns) |
| Duration(s) / (ms/µs/ns) | Duration(ms) / same | duration(ms) / same |
| Utf8 / LargeUtf8 | String | string_view |
| Binary / LargeBinary | Binary | binary_view |
| Dictionary<k, Utf8> | `Enum(categories)` when `population_rows` equals the frame's rows (exact cardinality; Polars picks the key width); else `Categorical(Categories(name=<column>, physical=k′))` | dictionary<string_view, k′> (Enum `ordered=1`) |
| timestamp_with_offset | Struct{timestamp: Datetime(u′, UTC), offset_minutes: Int16}, u′ = u with s → ms | struct of the mapped children |
| List(x) | List(mapped x) | large_list(mapped x) |
| FixedSizeList(x, w) | Array(mapped x, w) | fixed_size_list |

Polars reserves one key code, so its key width `k′` is UInt8 for `c ≤ 255`, UInt16 for `c ≤ 65,535`, else UInt32 (checked: an Enum of 256 categories is UInt16; a UInt8 Categorical refuses a 256th category) — one less than Arrow's boundaries for `k`.

## 6. Contract and output

`analytics.recommend`: `SCOPE = "per_column"`, `ARITY = 1`, `EXACT = True`; `DESCRIPTORS = {"dtype": pl.String}`. Keywords are Describe's (`population_rows`, `categorical_threshold`, `zstd_level`, `seed`) plus `boolean_pairs: tuple[tuple[str, str], ...] = (("true", "false"),)` (validated: pairs of distinct non-empty strings, compared case-insensitively). Eligibility as Describe.

`RecommendRust._compute` makes one `describe_and_recommend(df, seed, zstd_level, population_rows, categorical_threshold, boolean_pairs)` call per frame; its struct output holds every Describe metric, the size metrics and `REC_METRICS`. Conclusions are Describe's (Python base, Python estimators). Output order: keys, `status`, `dtype`, Describe METRICS, REC_METRICS, Describe CONCLUSIONS.

`REC_METRICS` (null unless `status = computed`):

| Column | Type | Meaning |
|---|---|---|
| `rec_nullable` | Boolean | the recommended array has nulls (§4.1) |
| `rec_arrow_type` | String | pyarrow's `str(type)` spelling, e.g. `decimal64(6, 2)`, `dictionary<values=string, indices=uint8, ordered=0>` |
| `rec_arrow_size_bytes`, `rec_arrow_size_zstd_bytes` | UInt64 | measured |
| `rec_polars_type` | String | Python `str(dtype)` |
| `rec_polars_size_bytes`, `rec_polars_size_zstd_bytes` | UInt64 | measured on the Polars layout |
| `rec_lossy_formatting` | Boolean | §5.4 |
| `rec_candidates` | List(Struct) | one entry per candidate, in the order tried; rejected candidates last; list columns: outer candidates, then inner ones prefixed `inner: ` (§4.1) |

`rec_candidates` struct fields: `arrow_type` (String), `rule` (String, e.g. `float→decimal`), `evidence` (String: the metric values the rule tested and what they implied, e.g. `n_nan=0 n_inf=0 n_fractional=120 max_frac_digits=2 int_digits=4 → p=6 s=2`; for dictionaries also `c=<value> from est_high|est_cardinality`, the estimator `method=exact|duj1|schnabel|chao1` and `est_low` when present; the key width is in `arrow_type`), `predicted_bytes` (UInt64), `projected_population_bytes` (Float64), `outcome` (Enum{chosen, failed, rejected, not_tried}; it crosses the plugin boundary as String and is cast to the Enum in Python), `reason` (String; null unless failed/rejected).

The Rust cardinality estimates behind `c` appear in `evidence`; they must equal the Python `est_high` / `est_cardinality` (§7).

## 7. Testing

### 7.1 `tests/test_recommend.py` (accuracy only)

1. **Contract** — schema, canonical order, ineligible rows null, `result().to_arrow()`.
2. **Reference agreement** — none (single implementation). Instead, **independent oracles** on `describe_mixed`, `stringified(describe_mixed)` and `large_dataset.arrow`:
   - `predicted_bytes` = `rec_arrow_size_bytes` for the chosen candidate;
   - pyarrow: `pa.array(original).cast(<rec_arrow_type>, safe=False)` measured by `_sizes.py` matches `rec_arrow_size_*` (ZSTD RTOL 0.01);
   - Polars: `series.cast(<rec_polars_type>)` measured by `_sizes.py` on `CompatLevel.newest()` matches `rec_polars_size_*` (ZSTD RTOL 0.01);
   - rows whose source is a string are skipped by these two oracles when the library cannot perform the cast (pyarrow and Polars cannot parse every ISO form); every row with a non-string source must be checked;
   - Rust estimates in `evidence` equal the Python estimators (RTOL 1e-9).
3. **Known answers** — one case per rule row in §4.2–4.5, including: `{0,1}` ints → Boolean; UInt8 vs Int8 tie → UInt8; Int128 beyond 64 bits → Decimal128(p, 0); Decimal scale reduced by `gcd`, and to integers when `s' = 0`; floats → Decimal32 (`123.45`), → Decimal64 (`1234567.891`), → Int (whole), → Float32 (`0.0009765625`: Float32 beats Decimal64(10, 10) on size); Decimal32 beats Float32 on the tie (`0.5`, `0.25`); NaN keeps a Float; `"007"` stays String (and never becomes a Decimal); `"1.50"` → Decimal with `rec_lossy_formatting`; varying decimal places beyond Decimal64 → Float64 (≤ 15 significant digits), → Float32 (≤ 6), and → Decimal128 when a value exceeds 15 significant digits (Float candidate `failed` or absent); fixed decimal places beyond Decimal64 → Decimal128; a Float64 column needing Decimal128 keeps Float64; `boolean_pairs` default and with `("y", "n")`; ISO date / time / naive / fixed offset / varying offsets (`timestamp_with_offset`); `"…00.000"` → Timestamp(s); ns out of range → String with a `failed` candidate; naive Datetime all-midnight → Date32, tz-aware → Timestamp; unit narrowing by `gcd` for Datetime / Duration / Time; single-item lists → scalar, and kept as List when null lists and null elements both occur; LargeList → List; dictionary key boundaries (Arrow 256 / 257 and 65,536 / 65,537 distinct; Polars 255 / 256); gate rejection above `categorical_threshold`; `population_rows` projection choosing plain over dictionary; Enum vs Categorical in `rec_polars_type`; all-null → Null.
4. **Conclusions** — inherited from Describe; one `with_metrics` check that REC columns are null on non-computed rows.

### 7.2 `tests/test_describe.py`

The six new metrics join the reference-agreement block (exact) and get known answers (e.g. `numeric_min_frac_digits` / `numeric_max_sig_digits` of `["1.50", "0.00120", "7"]` = 0 / 2; `gcd` of `[10, 20, 30]` = 10; `sum_len_unique` of `["ab", "ab", "c"]` = 3; `iso_max_sig_frac_digits` of `"10:00:00.120"` = 2). `size_polars_bytes` is re-tested under its new definition.

### 7.3 Rust unit tests

`cargo test --lib`: `cardinality_estimators::` (hand-worked Chao1 / Schnabel / Duj1 incl. intervals and validity), `recommend::` (rule tables, candidate ordering, float → decimal exactness, `polars_layout`), `sizes::` (existing framing tests, ported to arrow-rs), `describe::` (existing tests plus the new metrics).

### 7.4 Benchmarks

`benchmark_describe.py` unchanged (now timing the new metrics in all three implementations). `benchmark_recommend.py` times `RecommendRust` on the Describe datasets and reports the parallel speedup (Rust@1 ÷ Rust@N) only; `tests/performance/harness.py` needed no change: with no non-Rust implementation it reports the parallel split only.

## 8. Build and documentation

- `Cargo.toml`: add the arrow-rs crates (§2). `ryu`, `zstd` and `chrono-tz` are already present.
- CLAUDE.md: add **Recommend — `analytics.recommend`** under "1. Per-column"; update the project structure (flat `src/`), the Rust plugin list (`describe_and_recommend`) and the Describe metric list.

## 9. Out of scope

- Run-length / REE encoding (a future step 2 option).
- Returning or applying the recast table.
- Detecting that a string holds a list or struct (hierarchy items 14–15); expanding Struct fields.
- Decimal scale reduction for Float/String sources (their scale is already minimal) and negative scales.
- Migrating the Describe metric kernels and the Python↔Rust boundary to arrow-rs (the adapter isolates that future change).
