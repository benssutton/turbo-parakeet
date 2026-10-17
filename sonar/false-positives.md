# SonarQube findings reviewed and rejected

Findings below were investigated and are **not** to be "fixed". Each is marked *False positive*
in the local SonarQube server (status lives in its database: `docker compose down -v`
loses it, so re-mark after a reset using this list). Review before acting on a Sonar finding
of the same rule; add new entries here when a finding is rejected.

## `python:S5807` — "Change or remove this string; X is not defined" (12, BLOCKER)

**Why it is wrong:** the packages' `__init__.py` define a module-level `__getattr__`
(PEP 562, `analytics.base.lazy_attributes`), so every name in `__all__` resolves on access;
`from analytics.gcd import *` works. The optional implementations (those needing polars-ds,
scipy, datasketch, fastbloom, scikit-learn, DataFusion) are imported lazily on purpose, so
`import analytics.<technique>` does not require every optional library. Sonar's rule only
sees names bound statically.

**Do not:** remove the names from `__all__`, or import the optional modules eagerly.

| File (`services/analytics/analytics/`) | Names |
|---|---|
| `gcd/__init__.py` | `GcdNumpy`, `GcdMath` |
| `describe/__init__.py` | `DescribePolars`, `DescribeDataFusion` |
| `chi_squared/__init__.py` | `ChiSquaredScipy`, `ChiSquaredPolarsDS` |
| `membership/__init__.py` | `BloomFastbloom` |
| `similarity/__init__.py` | `MinHashDatasketch` |
| `pairwise_entropy/__init__.py` | `PairwiseEntropyPolars` |
| `threeway_entropy/__init__.py` | `ThreewayEntropyPolars` |
| `adjusted_rand/__init__.py` | `AdjustedRandSklearn` |
| `top_k/__init__.py` | `TopKPolars` |

## `python:S3776` — cognitive complexity 16 against the limit of 15 (1, CRITICAL)

Accepted like the Rust cases below: one point over, no natural seam. Marked *False positive*;
do not refactor to chase the score.

- `top_k/base.py` `TopK.agreement` (16): a single pass building found / wanted / rel per
  column, then three checks

## `rust:S3776` — cognitive complexity 16–17 against the limit of 15 (5, CRITICAL)

Not false positives in the strict sense: the scores are right, but the functions were judged
not worth splitting (no natural seam; splitting costs readability for one or two points).
Marked *False positive* in Sonar so the dashboard is clean; do not refactor to chase the score.

- `recommenders/streaming/mod.rs` `Streaming::add` (16)
- `recommenders/streaming/partial.rs` `BatchStats::of` (16): one pass per statistic, no seam
- `recommenders/engine/candidates.rs` `Rules::float` (16)
- `techniques/single_column/unordered/gcd.rs` `gcd_slice` (17): generic closures on a hot path inflate the score
- `common/encode.rs` `encode_series` (17): a flat dtype dispatch, one arm per type

(`DistinctSample::absorb` reached 19 and was split instead: `admit_exact` holds the
exact-phase bookkeeping.)

## `python:S1172` — unused function parameter (3, MAJOR)

`top_k/base.py` `_item_problems(self, floor, label, item)` (the three flagged parameters).

**Why it is wrong:** it is a hook with a fixed signature. The base returns `[]`;
`TopKSpaceSavingRust._item_problems` (`top_k/space_saving.py`) overrides it and uses `label`,
`item`, `true` and `floor`; `agreement()` passes all four.

**Do not:** remove a parameter, which would break the override.

## `python:S1940` — "use the opposite operator" for `not a <= b` (2, MINOR)

`pairwise_entropy/ams.py` lines 83 and 90: `not abs(got - ref) <= bound`.

**Why it is wrong:** the form is deliberately NaN-safe. A NaN estimate makes `abs(got - ref) <=
bound` False, so it is reported as a mismatch; the suggested `abs(got - ref) > bound` is False
for NaN and would silently pass it.

**Do not:** rewrite the comparisons as `>`.

## `python:S5778` — more than one call that can raise inside `pytest.raises` (1, MAJOR)

`tests/test_streaming_recommend.py`, `test_malformed_input_is_a_value_error`: the one-shot
check is `tech.add({"t": data}).result()` because Polars-built malformed data is refused in
`result()` while the rest is refused in `add()`. Both calls are the pipeline under test;
the operands are already built outside the block. Marked *False positive*; the other
`S5778` findings were fixed by building operands outside the block.

## `java:S1602` — "useless curly braces" around a lambda body (1, MINOR)

`NativeRecommender.java`, `Free.run`: `Native.run(() -> { free.invokeExact(handle); })`. The
braces are required: as an expression lambda, `FREE.invokeExact(handle)` links as returning
`Object` and throws `WrongMethodTypeException` (the handle returns `void`). A comment in the
code says so. Marked *False positive*; do not remove the braces.

## `java:S112` — generic `Throwable` in a `throws` clause (2, MAJOR)

`Native.java`, `Call.run` and `VoidCall.run` (`throws Throwable`). `MethodHandle.invokeExact`
is declared `throws Throwable`, and these two interfaces exist to carry that from the
downcalls to one place, `Native.call`, which rethrows runtime exceptions and errors and wraps
anything else in `AnalyticsException`. A narrower clause does not compile. Marked *False
positive*; do not narrow it. (The `RuntimeException`s thrown by the binding were separately
replaced by `AnalyticsException`.)

## Accepted, deferred (tests frozen) — 14 findings

Genuine findings in test files that were frozen for the change that introduced this list.
Marked *Accepted* in Sonar (not false positives); fix when tests may change.

- `python:S5778` (9): building operands / looking up the class outside `pytest.raises` —
  `tests/test_set_index.py:160,162,305`; `tests/test_minmax_index.py:157,159,164,290`;
  `tests/test_runs_test.py:155`; `tests/test_top_k.py:180`
- `python:S9073` (5, minor): split into separate asserts — `tests/test_set_index.py:246`;
  `tests/test_runs_test.py:201,227`; `tests/test_datagen.py:187,195`
