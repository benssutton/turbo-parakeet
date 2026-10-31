# Engine notes

- [clickhouse.md](clickhouse.md): ClickHouse's optimisation mechanisms (aide-memoire).
- [versions.yaml](versions.yaml): the engine versions the calibrations and spikes last ran on, including the pinned
  ClickHouse image (tag + digest). `chbench run` warns when the server differs.

## After an engine upgrade

Re-run what depends on the upgraded engine, compare with the previous results, record the outcome, then bump
`versions.yaml` in the same commit. Use a quiet machine for anything timed (services/bench/CLAUDE.md, Measurement notes).
Commands run from the repository root unless a directory is given.

**ClickHouse** (recreate the container from the new image, then; the spikes write their TSVs next to the script, over
the committed ones, so `git diff` is the comparison):
- From `services/bench/`: `python spikes/spike_ch_vocabulary.py`: types, codecs and skip rules
  (`spikes/spike_ch_vocabulary_results.tsv`).
- From `services/bench/`: `python spikes/spike_ch_bloom_size.py`: bloom_filter sizes
  (`spikes/spike_ch_bloom_size_results.tsv`).
- From `services/bench/`: `python -m spikes.spike_decimal32_zstd` (server run, `spikes/spike_decimal32_zstd_results.tsv`)
  and `python -m spikes.spike_decimal32_zstd --emulator` (no server, `spikes/spike_decimal32_zstd_emulator.tsv`):
  Decimal32 vs Decimal64 under ZSTD levels.
- `python tests/performance/spike_clickhouse_delta.py`: the on-disk Delta / DoubleDelta layout the size emulator
  models (prints a report; nothing is committed).
- From `services/bench/`, the example suites `replicates`, `int_encodings`, `float_codecs`, `decimal32_codecs`
  (`python -m chbench.cli run suites/<name>.yaml --store <new store>`); `replicates` must still have every pairwise
  ratio interval containing 1.
- The loop: `python -m evaluate run calibration` and `python -m evaluate run query_time`.

**DataFusion / Polars / pyarrow** (after the `requirements.txt` bump is installed):
- `python -m pytest tests/test_engine_canaries.py`: a canary that fails names its follow-up.
- `python tests/performance/spike_datafusion_ree.py`: the full REE operator survey.
- `python scripts/check.py check`: the recommender tests, streaming parity included (the recommenders measure sizes in
  Polars' and Arrow's layouts).
- `python -m evaluate run calibration`: predicted sizes against the new layouts.
- The in-process bench suites (ToDo 5), once they exist.

**Comparing.** `python -m evaluate compare <previous run> <new run>` for the loop; `chbench report --baseline` within
one store; `git diff` of a spike's committed TSV after re-running it.

**Recording.** A dated note in the relevant results directory (`services/bench/docs/results/`,
`services/evaluate/docs/results/`, or `services/bench/spikes/FINDINGS.md` for a spike) saying what changed and what did
not, then the `versions.yaml` bump, in one commit.
