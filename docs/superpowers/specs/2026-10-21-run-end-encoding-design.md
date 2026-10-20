# Run-end encoding for DataFusion — decision (ToDo 1a)

Status: approved 2026-10-21.  Sub-project C of ToDo 1a–g.  Outcome: **REE is not recommended**; this spec records why,
withdraws the reserved column and keeps the evidence re-runnable.

## 1. Question

ToDo 1a: recommend Arrow run-end encoding (REE) for the DataFusion target where the runs test shows long runs —
"recommend REE only where it is queryable".  Polars has no REE type; ClickHouse has none either (its run-oriented
codecs are ToDo 3's concern).

## 2. Evidence (DataFusion 54.0.0 Python, pyarrow 25.0.1, 2026-10-21)

A probe registered a table with run-end-encoded `Int64` and `Utf8` columns (with nulls) next to the same data plain,
ran the queries below on both, and compared the results.

| operator | REE column |
|---|---|
| `SELECT *`, projection; the column survives as REE in results | works |
| filters `=`, `BETWEEN`, `IN`, `LIKE`, `IS NULL`, string functions (`upper`) | works, results identical |
| `SUM`, arithmetic, `CAST`, `COUNT(*)`, `COUNT(DISTINCT)` | works, results identical |
| `GROUP BY`, `ORDER BY`, window `OVER (PARTITION BY …)`, `UNION ALL` | works, results identical |
| equi-join on the column | works, result identical |
| reading REE from an Arrow IPC file (`register_arrow`) | works |
| **`MIN` / `MAX` aggregates** | **fails**: "Internal error: Min/Max accumulator not implemented for type RunEndEncoded(…)" |

## 3. Decision

`MIN` / `MAX` are among the most common aggregates; an REE recommendation would make them fail on every recommended
column.  Therefore neither recommender ever recommends REE, and the reserved `datafusion_encoding` column (spec
2026-10-18 §2.1) is **withdrawn** (not emitted; removed from the skeleton).  The runs-test technique is unchanged: run
structure stays a metric for the ordering recommender (ToDo 8) and the ClickHouse delta codecs (ToDo 3).

**Re-evaluation trigger:** when DataFusion supports `MIN` / `MAX` natively on REE columns, REE is added in a new spec
(the reserved column re-introduced then).  `tests/performance/spike_datafusion_ree.py` re-runs this probe on the
installed DataFusion and prints the table above.

## 4. Changes

- `tests/performance/spike_datafusion_ree.py` (new; standalone, never collected by pytest): the probe — REE and plain
  tables, the queries of §2, per operator "works / fails (message) / results differ", plus the IPC read; prints the
  DataFusion and pyarrow versions.
- Python: the `OneShotRecommender` and `StreamingRecommender` class docstrings gain a "Run-end encoding" paragraph
  stating the decision, the reason and the spike to re-run.
- Rust: a doc comment where the candidate rules are enumerated (`recommenders/engine/candidates.rs`) stating that no
  REE candidate exists, why, and pointing to this spec.
- Spec 2026-10-18 §2.1: the `datafusion_encoding` reserved column removed from the DataFusion row, with a note
  pointing here.
- ToDo.md 1a: **Closed — not recommended** (spec 2026-10-21), one line with the reason and the trigger.
- CLAUDE.md: the recommender paragraph and the "Not implemented" line state REE is not recommended and why; Current
  Focus moves to sub-project D (range queries, ToDo 1g).

## 5. Testing

No behaviour changes.  The existing schema tests already assert the exact output columns (no `datafusion_encoding`).
The spike is run once and its output checked against §2.
