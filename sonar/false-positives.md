# SonarQube findings reviewed and rejected

Findings below were investigated and are **not** to be "fixed". Each is marked *False positive*
in the local SonarQube server (status lives in its database: `docker compose down -v`
loses it, so re-mark after a reset using this list). Review before acting on a Sonar finding
of the same rule; add new entries here when a finding is rejected.

## `python:S5807` — "Change or remove this string; X is not defined" (11, BLOCKER)

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

## `rust:S3776` — cognitive complexity 16–17 against the limit of 15 (accepted, not marked)

Left as is: no natural seam to split, and splitting would cost readability for one or two
points. These stay open in Sonar.

- `distinct_sample.rs` `DistinctSample::absorb` (16)
- `streaming.rs` `Streaming::add` (16)
- `recommend.rs` `Rules::float` (16)
- `gcd.rs` `gcd_slice` (17): generic closures on a hot path inflate the score
- `shared.rs` `encode_series` (17): a flat dtype dispatch, one arm per type
