# Describe Technique — Design (Spec A)

**Date:** 2026-09-26
**Status:** Approved in brainstorming; awaiting written-spec review
**Follow-up:** Spec B (recommender, cast verification, recast sizes) — separate spec, consumes this technique's output.

## 1. Purpose

`describe` is a per-column technique that profiles every column in order to decide, later (Spec B), whether it can be stored as a narrower or more compressible **Arrow** type without losing information. Spec A delivers the profile: metrics, cardinality estimates and a classification, with a single-pass Rust implementation benchmarked against a Polars/pyarrow reference and a DataFusion implementation.

### Success criteria

- `Describe*(**params).add(frames).result()` returns one row per frame column on the uniform technique contract (`docs/superpowers/specs/2026-09-24-uniform-technique-interface-design.md`).
- `DescribeRust` and `DescribeDataFusion` agree with the reference `DescribePolars` within the tolerances in §7.
- `tests/test_describe.py` has the standard four blocks; `tests/performance/benchmark_describe.py` reports the algorithmic, parallel and total speedups for `DescribeRust`.

### Decisions made during brainstorming

| Decision | Choice |
|---|---|
| Target type system | **Arrow** types; sizes measured as Arrow IPC with ZSTD buffer compression |
| ZSTD level | `zstd_level=1` (pyarrow's default for IPC) |
| Cardinality estimation | Picked by rule, never averaged: `population_rows` equal to the frame's row count → exact; larger → Haas–Stokes Duj1; otherwise Schnabel (multi-sample Lincoln–Petersen over a seeded 3-way split) when valid; otherwise Chao1. Closed-form 95% intervals where one exists; `estimates_agree` flags when Chao1 and Schnabel intervals disagree |
| Estimate agreement | Estimates depend on the random split, so implementations agree on them within 10%, not exactly |
| Sizes | Arrow (classic layout) and Polars (in-memory; native-layout IPC), each uncompressed and ZSTD. ClickHouse sizes deferred |
| Classification | Null → Constant → Boolean → Ordinal → Categorical → Discrete (first match wins). No "Continuous" class and no normality test |
| Lossless (used by Spec B) | **Value-preserving**: the original value can be reconstructed from the narrower type; formatting may change |
| Leading zeros | An integer-looking string with a leading zero keeps the column a String (see §5.3) |
| Datetime strings | ISO 8601 / RFC 3339 only |
| Struct | The whole struct value is hashed as one value; fields are not expanded (call `unnest()` first to describe fields) |
| Implementation approach | One single-pass Rust kernel; ★ Polars + pyarrow reference; DataFusion (Python) as a third benchmarked implementation |
| Recommender | Out of scope here — Spec B, a separate function taking this output |

## 2. Package layout

```
services/analytics/
├── src/describe.rs   # plugin entry describe_columns(frame): one struct row per column;
│                     #   frequency (key → (count, first_idx): n_unique, entropy, f1, f2, top-5),
│                     #   range (argmin/argmax first occurrence, min/max length),
│                     #   patterns (numeric-string and ISO 8601 byte scanners, no regex),
│                     #   numeric (float stats: ryu shortest repr, f32 round trip, NaN/inf/fractional)
├── src/sizes.rs      # plugin entry column_sizes(frame, zstd_level): IPC-framed bytes over arrow-rs ArrayData
├── src/shared.rs     # encode_series extended to Struct (whole value hashed)
├── src/lib.rs        # mod describe;
└── analytics/
    ├── _plugin.py    # + describe_columns, column_sizes wrappers
    └── describe/
        ├── __init__.py    # Describe, DescribeRust; REFERENCE = "DescribePolars";
        │                  # IMPLEMENTATIONS = ("DescribeRust", "DescribeDataFusion", "DescribePolars")
        ├── base.py        # Describe: METRICS, eligibility, rendering, estimators, classification, agreement
        ├── _sizes.py      # pyarrow IPC size helper shared by the two Python implementations
        ├── rust.py        # DescribeRust
        ├── polars.py      # DescribePolars ★
        └── datafusion.py  # DescribeDataFusion (lazy import; skip if datafusion missing)
```

Two plugin entry points: `describe_columns` does everything that scans values; `column_sizes` is separate because Spec B calls it again on recast columns. Both are private, reached only through `analytics._plugin` by `DescribeRust`.

## 3. Contract

- `SCOPE = "per_column"`, `ARITY = 1`, `EXACT = True`.
- `DESCRIPTORS = {"dtype": pl.String}` (Python `str(dtype)`, every row, as in `Gcd`).
- Constructor keyword-only arguments (validated at construction; `ValueError` if out of range):
  - `population_rows: int | dict[str, int] | None = None` — an int applies to every frame; a dict maps frame name → population row count. A value smaller than that frame's row count raises `ValueError` in `result()`.
  - `categorical_threshold: int = 10_000` (must be ≥ 0).
  - `zstd_level: int = 1` (must be a valid ZSTD level, 1–22). A metric parameter, passed to every implementation.
  - `seed: int = 0` — seeds the 3-way split used by the Schnabel estimator (§4.1, §6.2).
- **Eligible:** every dtype except `Object`, `Null` and `UInt128` (which cannot cross the plugin boundary). Ineligible columns are listed with `status="ineligible"` and null metrics/conclusions.
- Metrics that do not apply to a column's dtype are null (e.g. float stats on a String column, `inner_*` on scalars).
- Null = not computed / not applicable; NaN = computed but undefined.

## 4. Metrics (produced by every implementation)

Min, max and top-5 values are reported as **row indices of first occurrence**, never as values. The base renders them (§6.1), so float/date/list formatting differences between Rust, Polars and DataFusion can never cause mismatches, and one output column never has to hold several value types.

### 4.1 Group A — whole values (every eligible dtype; lists and structs as whole values)

| Metric | Type | Definition |
|---|---|---|
| `n_rows` | UInt64 | rows in the column |
| `n_null` | UInt64 | null rows |
| `n_unique` | UInt64 | distinct non-null values; floats canonicalised (`-0.0` = `0.0`, all NaN payloads = one value) |
| `entropy` | Float64 | Shannon entropy in bits over value frequencies with **null as its own category** (equals `h_a` of pairwise entropy). Zero-row column → NaN |
| `f1`, `f2` | UInt64 | non-null values occurring exactly once / exactly twice |
| `argmin`, `argmax` | UInt64 | first row index of the minimum / maximum non-null value; null when there is none |
| `min_len`, `max_len` | UInt64 | String/Categorical/Enum: UTF-8 bytes of the value; Binary: bytes; List/Array: element count; else null |
| `gcd` | Decimal(38, 0) | Int/UInt 8–64, Int128, Decimal, Date, Datetime, Duration, Time: GCD of the magnitudes of the physical values, as the Gcd technique (Decimal → unscaled, Date → days, Datetime/Duration → time unit, Time → ns); nulls skipped; all-null / all-zero / zero-row → 0; over 38 digits → null. Else null |
| `sum_len` | UInt64 | String/Categorical/Enum (UTF-8 bytes), Binary: total bytes over non-null values. Else null |
| `sum_len_unique` | UInt64 | as `sum_len`, over distinct non-null values (Rust: at each key's `first_idx` from the frequency map). Else null |
| `top5_idx` | List(UInt64) | first-occurrence row index of each of the ≤5 most frequent non-null values |
| `top5_count` | List(UInt64) | their counts |
| `capture_history` | List(UInt64), length 7 | distinct non-null values by the set of split subsets they occur in. Each row is assigned to subset 0, 1 or 2 by a seeded pseudo-random draw; element `k−1` counts values whose subset mask (bit *i* = seen in subset *i*) equals `k`, for `k = 1..7` |

**Split:** each implementation uses its own seeded generator — Rust: `splitmix64(seed + row_index) % 3`; Python implementations: `numpy.random.default_rng(seed).integers(0, 3, n_rows)`. Subsets are equal-sized in expectation. Because the generators differ, `capture_history` is not compared exactly between implementations (§7.1).

**Ordering for argmin/argmax** (matches Polars `min()`/sort):
- integers, Decimal, Date, Datetime, Duration, Time: physical value;
- floats: numeric order, NaN excluded (±inf included);
- Boolean: false < true;
- String, Binary: byte order; Categorical: string value; Enum: category order (to be confirmed against Polars during implementation);
- List, Array, Struct: Polars row-encoded (lexicographic) order.

**Top-5 order:** count descending, then first-occurrence index ascending.

### 4.2 Group B — float columns (Float32/Float64; else null)

| Metric | Type | Definition |
|---|---|---|
| `n_nan`, `n_inf` | UInt64 | NaN count; ±inf count |
| `n_fractional` | UInt64 | finite values that are not whole numbers |
| `max_frac_digits` | UInt32 | decimal places of the shortest round-trip decimal representation, exponent-aware: `max(0, fraction_digits − exponent)` (`0.1` → 1, `1e-7` → 7, `1.5e20` → 0); over finite values; null if none |
| `n_f32_inexact` | UInt64 | finite values where f64→f32→f64 changes the value; null for Float32 columns |

### 4.3 Group C — string columns (String/Categorical/Enum; else null)

| Metric | Type | Definition |
|---|---|---|
| `n_numeric` | UInt64 | values matching exactly `-?[0-9]+(\.[0-9]+)?` (ASCII digits; optional leading minus; at most one dot with a digit on each side). `"5."`, `".5"`, `"+5"`, `"1e5"`, `" 5"` do not match |
| `n_numeric_int` | UInt64 | subset of `n_numeric` with no dot |
| `n_leading_zero` | UInt64 | integer-looking values (`n_numeric_int`) whose digits start with `0` and have length > 1 (`"007"`, `"-012"`; not `"0"`, not `"-0"`) |
| `numeric_int_min`, `numeric_int_max` | Decimal(38, 0) | range of the integer-looking values; null if there are none or any has more than 38 significant digits |
| `numeric_max_int_digits` | UInt32 | max significant integer-part digits over all `n_numeric` values (leading zeros ignored; `"0.5"` → 0) |
| `numeric_max_frac_digits` | UInt32 | max fraction digits after trailing zeros are removed (`"1.50"` → 1) |
| `numeric_min_frac_digits` | UInt32 | min fraction digits over the `n_numeric` values after removing trailing zeros (integers count 0; `"1.50"` → 1); with `numeric_max_frac_digits` shows whether decimal places vary. Null if `n_numeric = 0` |
| `numeric_max_sig_digits` | UInt32 | max significant digits of a single `n_numeric` value: all digits after removing leading zeros (across the dot) and trailing fraction zeros (`"0.00120"` → 2, `"1200"` → 4, `"12.50"` → 3, `"0"` → 0); decides whether a value round-trips through Float32 (≤ 6) / Float64 (≤ 15). Null if `n_numeric = 0` |
| `n_iso_date` | UInt64 | values that are exactly an ISO date |
| `n_iso_time` | UInt64 | values that are exactly an ISO time |
| `n_iso_datetime` | UInt64 | ISO datetimes without offset |
| `n_iso_datetime_tz` | UInt64 | ISO datetimes with `Z` or `±HH:MM` |
| `iso_max_frac_digits` | UInt32 | max fractional-second digits (0–9) over ISO times/datetimes; null if none |
| `iso_max_sig_frac_digits` | UInt32 | max fractional-second digits over ISO times/datetimes (both kinds) after removing trailing zeros (`"10:00:00.120"` → 2, `"10:00:00.000"` → 0); null if none |
| `iso_n_offsets` | UInt64 | distinct offsets among `n_iso_datetime_tz` values (`Z`, `+00:00` and `-00:00` are one offset) |
| `iso_n_midnight` | UInt64 | ISO datetimes (both kinds) whose time is exactly `00:00:00` with any fraction all zeros |

The four ISO counts are disjoint.

### 4.4 Group D — Datetime columns (else null)

| Metric | Type | Definition |
|---|---|---|
| `n_midnight` | UInt64 | non-null values at exactly 00:00:00 local time — in the column's timezone if tz-aware, else of the naive value |

Time-unit granularity (e.g. all values whole milliseconds) is not computed here; Spec B reuses `Gcd`.

### 4.5 Group E — sizes (every eligible dtype)

| Metric | Type | Definition |
|---|---|---|
| `size_bytes` | UInt64 | Arrow IPC record-batch body length of the column (classic layout), uncompressed |
| `size_zstd_bytes` | UInt64 | the same body length with ZSTD buffer compression at `zstd_level` |
| `size_polars_bytes` | UInt64 | IPC body length of the column in Polars' native layout (`CompatLevel.newest()`), uncompressed |
| `size_polars_zstd_bytes` | UInt64 | IPC body length with ZSTD of the column in Polars' native layout (`CompatLevel.newest()`: StringView/BinaryView), i.e. what `write_ipc(compression="zstd")` writes |

Arrow layout: the column as `Series.rechunk().to_arrow(compat_level=pl.CompatLevel.oldest())` — LargeUtf8, LargeBinary, LargeList (not Polars' internal view types). Framing follows Arrow IPC: every buffer is padded to 8 bytes; a validity buffer is written only if the column has nulls; with compression, each buffer is prefixed by its 8-byte uncompressed length and compressed as one ZSTD frame. How pyarrow writes a buffer that does not shrink (length prefix `-1`, raw bytes) is confirmed against pyarrow during implementation and mirrored in Rust. The reference definition is pyarrow's `body_length` of the IPC message (§5.2). The Polars ZSTD size uses the same framing on the `CompatLevel.newest()` layout (view types include their variadic data buffers).

### 4.6 Group F — inner values (List/Array only; else null)

`inner_n_values` (UInt64, total elements including null elements) plus `inner_`-prefixed copies of every metric in groups A, B and C, computed on the values **one nesting level down** (the flattened child array, respecting offsets). `inner_argmin`, `inner_argmax` and `inner_top5_idx` index into the flattened values; `inner_capture_history` splits by flattened-element index with the same seed. For `List(List(x))` the inner values are `List(x)` values. `inner_n_rows` is not repeated (`inner_n_values` replaces it).

## 5. Implementations

### 5.1 DescribeRust — `describe_columns` + `column_sizes`

Rayon-parallel across columns; within a column, 64K-row chunks (as in `gcd.rs`). Per column, one pass:

1. **Frequencies** (`frequency.rs`): `encode_series` → u64 key per value (nulls out-of-band; floats canonicalised; strings/categorical/enum/nested/struct via foldhash). Each chunk builds a foldhash map `key → (count, first_idx, subset_mask)`, where `subset_mask |= 1 << (splitmix64(seed + row) % 3)`; chunk maps are merged (count summed, first_idx = min, masks OR-ed). One O(card) sweep yields `n_unique`, `entropy`, `f1`, `f2`, `capture_history` (a 7-bin count of masks), and the top-5 by partial selection with the §4.1 tie-break. Hash collisions on strings are possible in principle (≈6×10⁻¹¹ per pair at 50K rows, as documented in CLAUDE.md) and accepted.
2. **Range** (`range.rs`): typed comparisons of physical values for scalars; byte comparison for strings/binary; Polars row encoding for List/Array/Struct. Ties keep the lowest index.
3. **Scanners** (`patterns.rs`): hand-written byte state machines, one forward pass per value, no backtracking and no regex, so run time is linear in input length for any input (safe against adversarial strings). Numeric grammar as §4.3. ISO grammar:
   - date `YYYY-MM-DD` (year 0000–9999; month 01–12; day valid for the month, Gregorian leap years);
   - time `HH:MM[:SS[.f{1,9}]]` (HH 00–23, MM 00–59, SS 00–59; no leap second);
   - datetime = date, then `T` or a single space, then time;
   - offset `Z` or `±HH:MM` (HH 00–23, MM 00–59);
   - uppercase `T` and `Z` only.
4. **Floats** (`numeric.rs`): `ryu` shortest representation for `max_frac_digits`; f32 round trip; NaN/inf/fractional counts.
5. **Datetime** `n_midnight`: naive → `physical.rem_euclid(units_per_day) == 0`; tz-aware → local time via Polars' `timezones` feature (chrono-tz).
6. **Inner values**: flatten the List/Array child through its offsets and run steps 1–4 on it.
7. **Sizes** (`sizes.rs`, separate entry point): per column, rechunk, `to_arrow(CompatLevel::oldest())`, walk the buffers recursively (validity, offsets, values, children), apply §4.5 framing, `zstd::bulk::compress(buffer, level)`. Polars sizes: the same buffer walk and framing on `to_arrow(CompatLevel::newest())`.

`shared.rs::encode_series` gains Struct support: a struct value's key is a foldhash of its fields' keys (null fields distinguished from null struct). This does not change existing keys for other dtypes.

**Code comment requirement (leading zeros):** `patterns.rs` (where `n_leading_zero` is counted) and, in Spec B, the recommender must carry this rationale:

> An integer-looking string with a leading zero (`"007"`) must stay a String: identifiers such as UUID fragments, account numbers or zip codes can be all digits with significant leading zeros, and casting to an integer would lose them. A value with a single decimal point (`"007.50"`) is unlikely to be an identifier, so only numeric equivalence matters for it — differing leading or trailing zeros are acceptable.

### 5.2 DescribePolars ★ — reference

Polars expressions, one `select` per column (no Python-level row loops):
- Frequencies: `with_row_index()`, a subset column from `numpy.random.default_rng(seed).integers(0, 3, n_rows)`, normalise floats (`-0.0` → `0.0`, NaN → one canonical NaN), `group_by(value).agg(pl.len(), pl.col("index").first(), <OR of 1 << subset>)` → group A metrics including `capture_history`.
- Range: `arg_min` / `arg_max`; nested types via a stable `arg_sort` (first index among equals).
- Numeric strings: anchored `str.contains` with the §4.3 pattern (Polars' regex engine is the Rust `regex` crate: linear time, safe).
- ISO strings: anchored `str.contains` for each form's shape, then `str.to_date` / `str.to_time` / `str.to_datetime(format=..., strict=False)` for calendar validity.
- Floats: `max_frac_digits` from the shortest string form (exponent-aware parse); f32 round trip via `cast(pl.Float32).cast(pl.Float64)`.
- `n_midnight`: `dt.time() == time(0)`.
- Inner values: `list.explode()` (or `Array` → `explode`) respecting offsets, then the same expressions.
- Sizes (`_sizes.py`): `pa.record_batch([arrow_column])` written through `pa.ipc.new_stream` with `IpcWriteOptions(compression=None)` and `IpcWriteOptions(compression="zstd", ...)` at `zstd_level` (via `pa.Codec("zstd", compression_level=zstd_level)`); size = the record-batch message's `body_length`. Polars sizes: the same IPC measurement, plain and ZSTD, on `to_arrow(compat_level=pl.CompatLevel.newest())`.

### 5.3 DescribeDataFusion

- Per frame: `frame.to_arrow(compat_level=oldest)` plus a row-number column and the same numpy subset column as §5.2, registered with a `datafusion.SessionContext`.
- SQL per column: `COUNT`, `COUNT(col)`; frequencies via a `GROUP BY` subquery with `COUNT(*)`, `MIN(row)` and `BIT_OR(1 << subset)` → entropy, f1, f2, top-5, capture history; argmin/argmax via `ORDER BY value, row LIMIT 1`; `regexp_like` (Rust regex, linear) for shapes; `try_cast(... AS DATE / TIMESTAMP / TIME)` for calendar validity; `unnest` for inner values.
- Sizes (Arrow and Polars): the shared `_sizes.py` helper.
- Any metric that cannot be expressed exactly in DataFusion SQL is computed with pyarrow in this class and listed in the class docstring — never approximated.

### 5.4 Edge cases

| Case | Behaviour |
|---|---|
| Zero-row column | eligible; counts 0; `entropy` NaN; indices and `min_len`/`max_len` null; sizes computed |
| All-null column | `n_unique` 0; `entropy` 0.0; indices null |
| List with only empty/null lists | inner metrics computed on zero inner values (as zero-row) |
| Ineligible dtype | `status="ineligible"`, all metrics/conclusions null, `dtype` filled |
| Plugin error | propagates unchanged |

## 6. Base conclusions (`Describe._conclude`)

Computed identically for every implementation from the metrics and the collected frames (`self._collected`). All are null unless `status == "computed"`. The output column order is keys, `status`, `dtype`, METRICS, then these.

### 6.1 Rendering

| Column | Type | Value |
|---|---|---|
| `min`, `max` | String | `str()` of the Python value at `argmin` / `argmax` |
| `top5` | List(Struct{value: String, count: UInt64}) | `str()` of each value at `top5_idx`, with `top5_count` |
| `inner_min`, `inner_max`, `inner_top5` | as above | from the flattened values |

### 6.2 Cardinality estimators

Let `n = n_rows − n_null`, `d = n_unique`, `P` = this frame's population rows, `q = n_rows / P`, `z = 1.96`.

**Chao1** (bias-corrected; interval: Chao 1987 log-normal, variance as in the EstimateS user guide):
- `chao1 = d + f1·(f1−1) / (2·(f2+1))`
- variance: if `f2 > 0`: `f1(f1−1)/(2(f2+1)) + f1(2f1−1)²/(4(f2+1)²) + f1²·f2·(f1−1)²/(4(f2+1)⁴)`; if `f2 = 0`: `f1(f1−1)/2 + f1(2f1−1)²/4 − f1⁴/(4·chao1)`
- with `T = chao1 − d`, `K = exp(z·sqrt(ln(1 + var/T²)))`: interval `[d + T/K, d + T·K]`; `T = 0` → `[d, d]`

**Schnabel** (multi-sample Lincoln–Petersen) from `capture_history`. Let `S1, S2, S3` be the distinct-value sets of the three subsets:
- catches `C_t = |S_t|`; marked before occasion `t`: `M_1 = 0`, `M_2 = |S1|`, `M_3 = |S1 ∪ S2|`; recaptures `R_2 = |S1 ∩ S2|`, `R_3 = |S3 ∩ (S1 ∪ S2)|`; `R = R_2 + R_3`; `A = C_2·M_2 + C_3·M_3`
- `schnabel = A / (R + 1)`
- interval: Byar's closed-form Poisson limits on `R` — `R_lo = R·(1 − 1/(9R) − z/(3√R))³`, `R_hi = (R+1)·(1 − 1/(9(R+1)) + z/(3√(R+1)))³` — giving `[A / R_hi, A / R_lo]`
- **valid** only when `d / n < 0.5` and `R ≥ 1`; otherwise `schnabel` and its interval are null

**Duj1** (Haas–Stokes): `d / (1 − (1−q)·f1/n)`; no closed-form interval.

| Column | Type | Rule |
|---|---|---|
| `unique` | Boolean | `d == n` and `n > 0` |
| `chao1`, `chao1_low`, `chao1_high` | Float64 | as above; always reported |
| `schnabel`, `schnabel_low`, `schnabel_high` | Float64 | as above; null when not valid |
| `est_cardinality` | Float64 | first rule that applies: `P == n_rows` → `d`; `P > n_rows` → Duj1 (`n == 0` → 0); Schnabel valid → `schnabel`; else `chao1` |
| `est_method` | Enum{exact, duj1, schnabel, chao1} | the rule used |
| `est_low`, `est_high` | Float64 | interval of the chosen method: exact → `[d, d]`; Duj1 → null; Schnabel / Chao1 → their intervals |
| `estimates_agree` | Boolean | Chao1 and Schnabel intervals overlap; null when Schnabel is not valid. False signals strong frequency skew — treat both as lower bounds |

Estimators are never averaged: they rest on different assumptions (Chao1 is a lower bound; Lincoln–Petersen/Schnabel is biased low when some values are much more common than others).

Inner equivalents (`inner_unique`, `inner_chao1`, `inner_schnabel`, … `inner_estimates_agree`) use `inner_n_values − inner_n_null` as `n`, `inner_capture_history`, and the same `q`.

### 6.3 Classification

`class` and `inner_class`: `Enum{null, constant, boolean, ordinal, categorical, discrete}`, first match wins. `N` = `P` if given, else `n_rows`. For `inner_class`, `N` = `inner_n_values / q` (the estimated number of inner values in the population; `inner_n_values` when `P` is absent).

1. **null** — `n_null == n_rows` (zero-row columns included).
2. **constant** — `n_unique == 1` (nulls may be present).
3. **boolean** — `n_unique == 2`.
4. **ordinal** — all non-null values are whole numbers with `0 ≤ min` and `max ≤ 2N`. Whole-number columns: integer dtypes; Decimal with scale 0; floats with `n_fractional == n_nan == n_inf == 0`; strings with `n_numeric_int == n` and `n_leading_zero == 0` (range from `numeric_int_min/max`). Temporal dtypes never qualify.
5. **categorical** — `est_cardinality ≤ categorical_threshold`.
6. **discrete** — everything else.

## 7. Testing

### 7.1 `tests/test_describe.py` (accuracy only, never timed)

A new `datagen.describe_mixed(n, seed)` frame covers: Int8–Int64, UInt8–UInt64, Int128, floats with NaN/±inf/−0.0, Decimal, Date, naive and tz-aware Datetime, Duration, Time, Boolean, Binary, String (free text, numeric, leading-zero, ISO date/time/datetime/datetime-tz), Categorical, Enum, List(Int64), List(String), Array, Struct, an all-null column, plus a zero-row frame and one column of each ineligible dtype.

1. **Contract** — every implementation: schema, canonical order, ineligible rows null.
2. **Reference agreement** — `DescribeRust`, `DescribeDataFusion` vs `DescribePolars`. `Describe.agreement` is overridden for per-metric tolerances: all metrics exact except
   - `entropy` / `inner_entropy`: RTOL 1e-9;
   - `size_zstd_bytes`, `size_polars_zstd_bytes`: RTOL 0.01;
   - `capture_history` / `inner_capture_history`: not compared element-wise (the split generators differ); instead the Schnabel estimate derived from each implementation's history must agree with the reference's within RTOL 0.10 (null in both, or both valid).
3. **Known answers** (reference included):
   - entropy and f1/f2 on small hand-worked columns; top-5 tie-break; argmin first occurrence; `-0.0`/NaN canonicalisation;
   - `max_frac_digits` for `0.1`, `1e-7`, `1.5e20`; `n_f32_inexact` for `0.1` (inexact) vs `0.5` (exact);
   - numeric scanner: `"5."`, `".5"`, `"1.2.3"`, `"+5"`, `"1e5"`, `" 5"`, Arabic-Indic digits (no match); `"007"`, `"-012"` (leading zero); `"0"`, `"-0"` (not leading zero); `"007.50"` (numeric, int digits 1, frac digits 1);
   - ISO scanner: `2024-02-29` valid, `2023-02-29` invalid, `24:00` invalid, lowercase `t` invalid, 9 vs 10 fractional digits, `Z` ≡ `+00:00` ≡ `-00:00` for `iso_n_offsets`;
   - Enum ordering; tz-aware `n_midnight` across a DST change; equal struct values count once; list inner metrics;
   - sizes vs hand-computed values (1,000 Int32 no nulls → 4,000 bytes; with nulls → + 128-byte validity buffer padded to 8);
   - adversarial long inputs (e.g. `"0"*10**6 + "." + "0"*10**6 + "."`) return correct counts (linearity is guaranteed by construction; not timed);
   - `size_polars_bytes` equals the uncompressed IPC body length of the Polars native (`CompatLevel.newest()`) layout;
   - `capture_history` sums to `n_unique`; a column where every value occurs in every subset (e.g. 10 values × 10,000 rows) puts all mass in element 7;
   - **stringified columns**: a new `datagen.stringified(frame)` casts ints, floats, Decimal, Date, Datetime (naive and tz-aware), Time, Boolean and `List(Int64)` columns to String (`List(String)`), applied to `describe_mixed` and to 25 non-string columns of `large_dataset.arrow`. Expected: integer columns `n_numeric_int == n`; Date `n_iso_date == n`; naive Datetime (Polars writes `"2024-01-05 10:00:00.000"`) `n_iso_datetime == n`; tz-aware Datetime (Polars writes `"…+00:00"`) `n_iso_datetime_tz == n` and `iso_n_offsets == 1`; stringified lists → the same checks on inner values; floats written in exponent form (Polars writes `1e-7`) are deliberately not numeric; Boolean (`"true"`/`"false"`) is neither numeric nor ISO. Expectations are asserted on the strings Polars actually produces.
4. **Conclusions** via `with_metrics`: each estimator branch (exact, Duj1, Schnabel, Chao1) including Schnabel invalid when `d/n ≥ 0.5` or `R = 0`; hand-worked Chao1 and Schnabel intervals (both `f2 > 0` and `f2 = 0`, and `T = 0`); `estimates_agree` true and false; `population_rows` as int and as dict; `ValueError` for `population_rows < n_rows`; one case per class including ordinal-before-categorical (`0..4` integer codes → ordinal); a NaN-entropy row.

Missing optional library (datafusion, pyarrow) → visible skip naming the library.

### 7.2 Rust unit tests

`cargo test --lib describe::` — scanner grammar tables (accept/reject cases), exponent-aware decimal-place counting, IPC framing arithmetic, subset-mask merge and capture-history binning.

## 8. Benchmark — `tests/performance/benchmark_describe.py`

About 20 lines calling `harness.run("analytics.describe", [...])`:

| Dataset | Purpose |
|---|---|
| `large_dataset.arrow` (50K × 101) | realistic mix |
| narrow 10M × 4 (int, float, numeric string, ISO string) | within-column chunk parallelism and merge |
| wide 1M × 100, mixed dtypes | across-column parallelism |
| nested 1M × 3 (List(Int64), List(String), Struct) | inner values and row encoding |

The harness reports algorithmic (fastest non-Rust ÷ Rust@1), parallel and total speedups; `budget_s` skips implementations that are too slow on a shape.

## 9. Build and documentation

- `Cargo.toml`: add `zstd` and `ryu`; enable Polars features `timezones`, `dtype-struct`, `dtype-array` and row encoding as required.
- `pyproject.toml`: unchanged (datafusion and pyarrow are optional imports).
- CLAUDE.md: add **Describe — `analytics.describe`** under "1. Per-column"; add `describe/` to the project structure and the Rust plugin list.

## 10. Out of scope (Spec B)

Delivered by Spec B: docs/superpowers/specs/2026-09-26-recommend-technique-design.md.

Deferred (not in Spec A or B): ClickHouse column sizes (MergeTree serialisation model and compressed-block approximation).
