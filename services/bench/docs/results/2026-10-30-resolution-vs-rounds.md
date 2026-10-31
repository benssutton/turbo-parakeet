# Resolution against the number of rounds, 2026-10-30

> Dated record. Commit `0538f13` (the branch head when measured), ClickHouse 26.8.31.2 in Docker (`chbench-ch`),
> Windows power scheme **Balanced** (the 2026-10-29 runs used High performance), the same laptop. Run from
> `services/bench/` on a fresh store `results-resolution.db` (not committed):
>
> ```
> sed 's/samples: 20/samples: 10/' suites/replicates.yaml > replicates10.yaml
> python -m chbench.cli run replicates10.yaml --store results-resolution.db                   # 10 rounds
> python -m chbench.cli report --suite replicates --store results-resolution.db --baseline skip_index=minmax --csv rep10.csv
> python -m chbench.cli run replicates10.yaml --store results-resolution.db --samples-add 10  # 20 rounds
> python -m chbench.cli report --suite replicates --store results-resolution.db --baseline skip_index=minmax --csv rep20.csv
> python -m chbench.cli run replicates10.yaml --store results-resolution.db --samples-add 10  # 30 rounds
> python -m chbench.cli report --suite replicates --store results-resolution.db --baseline skip_index=minmax --csv rep30.csv
> ```
>
> `replicates10.yaml` is `suites/replicates.yaml` with `samples: 10` (the committed suite has 20). The three runs took
> 39 s, 39 s and 40 s.

The question (ToDo 3): how wide is a paired ratio interval at 10, 20 and 30 rounds, so that `--samples-add` can be
chosen per question. The suite is the replicates check: three Int32 tables that differ only in their skip index
(`minmax`, `set(100)`, `bloom_filter`) and cannot be told apart by any query. The baseline is `skip_index=minmax`, as on
2026-10-29, so there are 6 non-baseline rows (2 tables x full scan / filter 0.05 / filter 0.20). Width is
`ratio_hi - ratio_lo` of the 95% interval, and the median is over those 6 rows.

| rounds | median CPU-ratio width | median server-ratio width | smallest resolvable difference ≈ width / 2 (CPU) | all intervals contain 1? |
|---|---|---|---|---|
| 10 | 0.177 | 0.327 | ≈ 9% | yes (6 of 6 CPU, 6 of 6 server) |
| 20 | 0.078 | 0.161 | ≈ 4% | yes (6 of 6, 6 of 6) |
| 30 | 0.059 | 0.083 | ≈ 3% | yes (6 of 6, 6 of 6) |

The widest single CPU interval was 0.651 at 10 rounds, 0.142 at 20 and 0.104 at 30.

Quietness, as `run` printed it (host CPU over 3 s, before / after): 10 rounds 36% / 27% with 1 ClickHouse merge active
before; +10 rounds 33% / 34%; +10 rounds 30% / 24%. Every run warned that host CPU was over 10%, so these are
*quieted*, not idle runs, like the 2026-10-29 ones.

Reading. The CPU interval narrows from 0.177 to 0.078 to 0.059, and the server-wall interval from 0.327 to 0.161 to
0.083: about 2.3x and 3x narrower at 20 and 30 rounds for CPU, 2x and 4x for wall, faster than the 1/√n that 20 and
30 rounds would give by themselves (1.4x and 1.7x). The figures are cumulative on one store, so rounds 1-10 are in all
three. They ran under the highest load (host CPU 36% before, one merge active), so the faster narrowing is at least
partly those noisy early rounds being diluted by later ones, not independent measurements. The 6 rows behind each
median are also correlated, because they share the minmax baseline, and the 10-round set holds a 0.651 outlier. Treat
the three widths as the scale on this laptop for this suite, not as a law. The 10-round width (0.177) is well below the
0.33 that 2026-10-29 reported for the codec suites. That comparison is confounded: the codec runs used High performance
power and compared variants whose costs differ, this one used Balanced and near-identical tables. The 0.33 is the
safer figure to plan with. The server clock is about twice as wide as CPU at every n, so a question that needs a
small difference should be asked of `cpu_ratio`. All 6 intervals per clock contain 1 at every n: the replicates check
holds at 30 rounds too.
