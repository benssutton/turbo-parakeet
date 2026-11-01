# SonarQube findings reviewed and rejected

Findings below were investigated and are **not** to be "fixed". Each is marked *False positive*
in the local SonarQube server (status lives in its database: `docker compose down -v`
loses it, so re-mark after a reset using this list). Review before acting on a Sonar finding
of the same rule; add new entries here when a finding is rejected.

## `python:S5807` — "Change or remove this string; X is not defined" (16, BLOCKER)

**Why it is wrong:** the packages' `__init__.py` define a module-level `__getattr__`
(PEP 562, `turbo_parakeet.techniques.base.lazy_attributes`), so every name in `__all__` resolves on access;
`from turbo_parakeet.techniques.gcd import *` works. The optional implementations (those needing polars-ds,
scipy, datasketch, fastbloom, scikit-learn, DataFusion) are imported lazily on purpose, so
`import turbo_parakeet.techniques.<technique>` does not require every optional library. Sonar's rule only
sees names bound statically.

`describe` and `chi_squared` were not flagged in the 2026-10-24 analysis, although they use the same
pattern; their rows are kept so a future flag is recognised.

**Do not:** remove the names from `__all__`, or import the optional modules eagerly.

| File (`services/analytics/python/turbo_parakeet/`) | Names |
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
| `distinct_count/__init__.py` | `DistinctCountPolars`, `DistinctCountPolarsApprox` |
| `marginal_entropy/__init__.py` | `MarginalEntropyPolars` |
| `population_cardinality/__init__.py` | `PopulationCardinalityPolars` |

## `python:S3776` — cognitive complexity 16–17 against the limit of 15 (6, CRITICAL)

Accepted like the Rust cases below: one or two points over, no natural seam. Marked *False positive*;
do not refactor to chase the score.

- `top_k/base.py` `TopK.agreement` (16): a single pass building found / wanted / rel per
  column, then three checks
- `turbo_parakeet/recommenders/chbench.py` `_predicted_inproc` (16)
- `turbo_parakeet/techniques/base.py` `_requested` (17)
- `chbench/harness/engines/inproc.py` `InProcEngine.call` (16): one request / reply exchange with its timeout and
  dead-worker handling
- `chbench/cli.py` `_cmd_datasets` (17)
- `chbench/harness/validate.py` `resolve_type` (17)

(The 2026-10-31 analysis also flagged 19 functions at 18–143 in the bench and datagen; all were split.)

## `rust:S3776` — cognitive complexity 16–17 against the limit of 15 (6, CRITICAL)

Not false positives in the strict sense: the scores are right, but the functions were judged
not worth splitting (no natural seam; splitting costs readability for one or two points).
Marked *False positive* in Sonar so the dashboard is clean; do not refactor to chase the score.

- `recommenders/streaming/mod.rs` `Streaming::add` (16)
- `recommenders/engine/candidates.rs` `Rules::float` (16)
- `techniques/single_column/unordered/gcd.rs` `gcd_slice` (17): generic closures on a hot path inflate the score
- `common/encode.rs` `encode_series` (17): a flat dtype dispatch, one arm per type
- `bindings/api/recommenders.rs` `skip_params` (16): one check per skip-index keyword, each its own error
- `common/clickhouse_sizes.rs` test `delta_transform_cases` (17): a table of hand-worked cases in one unit test

(`DistinctSample::absorb` reached 19, `BatchStats::of` 20 and `Streaming::row` 29, and the ClickHouse size
emulator's `streams` 20 and `gcd_transform` 23; all were split instead. `DistinctSample::absorb`: `admit_exact` holds the
exact-phase bookkeeping.)

## `python:S107` — too many parameters (4, MAJOR)

`recommenders/oneshot.py` `OneShotRecommender.__init__` (16), `recommenders/streaming.py`
`StreamingRecommender.__init__` (19), and their `_plugin.py` wrappers `oneshot_recommender` /
`streaming_recommender`. Every parameter is a keyword-only recommender option that mirrors the
Rust API, the C ABI (`capi.rs`) and the Java `OneShotParams` / `StreamingParams`; grouping them
into a config object would change the public API of four bindings for no behavioural gain.
Marked *Accepted* (won't fix), not *False positive*: the count is right.

## `python:S1172` — unused function parameter (3, MAJOR)

`top_k/base.py` `_item_problems(self, floor, label, item)` (the three flagged parameters).

**Why it is wrong:** it is a hook with a fixed signature. The base returns `[]`;
`TopKSpaceSavingRust._item_problems` (`top_k/space_saving.py`) overrides it and uses `label`,
`item`, `true` and `floor`; `agreement()` passes all four.

**Do not:** remove a parameter, which would break the override.

The same holds for the bench's in-process engine protocol (`chbench/harness/engines/inproc.py`):
`datafusion_engine.py` `cast(type_spec, table_id)`, `attach(table_id, path, held)` and
`plan(table_id, query)` (3) do not need the flagged parameters, but the Polars engine's
implementations of the same methods do. Marked *False positive*.

`chbench/harness/validate.py` `expand_grid(..., clickhouse_version)` (1): unused today, kept on
purpose because the applicability matrices are per ClickHouse version (its docstring). Marked
*Accepted*.

## `python:S1940` — "use the opposite operator" for `not a <= b` (6, MINOR)

`pairwise_entropy/ams.py` lines 83 and 90: `not abs(got - ref) <= bound`. In synthdata:
`column/solve.py` `solve_monotone` (`not lo < hi`, `not abs(achieved - target) <= tol`),
`column/spec.py` (`not (self.distribution.sigma > 0)`) and `frame/generate.py`
(`not rel["nmi"] > 1 - 1e-9`).

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

## `python:S5754` — `except BaseException` that does not re-raise (2, CRITICAL)

`chbench/harness/engines/base.py` `_interrupt_all` and `chbench/harness/orchestrate.py`
`_close_all`. Deliberate: in cleanup, one engine whose `interrupt()` / `close()` fails, including
on a second Ctrl-C, must not skip the others or mask the exception already propagating; the
failure becomes a note on that exception (or a warning when the run succeeded). Marked *False
positive*; do not narrow to `Exception`.

## `python:S6353` — `[0-9]` instead of `\d` (6, MINOR)

`chbench/harness/inproc_types.py`, the Polars / Arrow type-spec regexes. In a Python `str`
pattern `\d` matches every Unicode decimal digit (`"٣"`), which these parsers must reject;
`[0-9]` is ASCII only. Marked *False positive*; do not "simplify".

## `python:S1244` — floating-point equality (4, MAJOR)

`synthdata/column/metrics.py` (`mean == 0.0`, `sd == 0.0`, twice): guards against dividing by
an exact zero; `frame/generate.py` `rel["containment"] != 1.0`: the ratio of contained to all
values is exactly 1.0 when every value is contained, and a foreign key must be fully contained.
Marked *False positive*.

## Other single-case rejections (all *False positive*)

- `python:S1515` (2): `chbench/harness/queries.py` and `synthdata/synth/pairwise.py`, a lambda
  over the loop variable passed as a `key=` to `min` / `max` / `sorted` and consumed in the same
  iteration.
- `python:S3516` (1): `chbench/cli.py` `_cmd_datasets` always returns 0: every subcommand handler
  returns the exit code through the same dispatch.
- `python:S5655` (1): `chbench/harness/orchestrate.py`, `skipped(v, ...)` with
  `v = dataclasses.replace(variant, ...)`, which is a `TableVariant`; Sonar loses the type.
- `python:S5863` (2): `test_queries.py` / `test_pairwise.py` determinism tests compare two
  separate evaluations of the same seeded call.
- `python:S7517` (2): `turbo_parakeet/techniques/base.py`, `used` is a dict used as an ordered
  set of `(frame, column)` keys; the loops unpack the key tuples.
- `python:S3415` (1): `services/datagen/tests/column/test_sequence.py` asserts actual-first throughout.

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

## Accepted, deferred (tests frozen) — resolved

The 14 test findings once deferred here (`python:S5778` and `python:S9073` in
`test_set_index.py`, `test_minmax_index.py`, `test_runs_test.py`, `test_top_k.py` and
`test_datagen.py`) have since been fixed: none came back in the 2026-10-24 analysis after the
server's database was reset.
