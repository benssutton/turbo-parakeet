# Synthetic data generation: feasibility findings

Retained record of what was tried when designing the generator, what worked,
what did not, and the gotchas. Scripts live alongside this file and are
re-run from the repository root: `python spikes/<script>.py`. Each script
puts the repository root on `sys.path` itself, resolved from its own
`__file__` rather than from the current directory, so running one from inside
`spikes/` (`python spike_entropy.py`) works too.

Conventions: rows = 100 000, seeds 0/1/2 unless stated. "Achieved" is the
value measured by `spikes/_common.py` on the generated array.

## 1. Entropy + cardinality (spike_entropy.py)

Full output (n = 100 000, seeds 0/1/2, `k` in {1 000, 10 000}, targets in
{0.5, 2.0, 4.0, 8.0, 12.0} bits):

```
     k  target        s  achieved    card     ms  status
  1000    0.50    4.603     0.500  0.0100     16  ok
  1000    0.50    4.603     0.500  0.0100     11  ok
  1000    0.50    4.603     0.500  0.0100      9  ok
  1000    2.00    2.241     2.000  0.0100      7  ok
  1000    2.00    2.241     2.000  0.0100     10  ok
  1000    2.00    2.241     2.000  0.0100     10  ok
  1000    4.00    1.562     4.000  0.0100      8  ok
  1000    4.00    1.562     4.000  0.0100      7  ok
  1000    4.00    1.562     4.000  0.0100      8  ok
  1000    8.00    0.924     8.000  0.0100      9  ok
  1000    8.00    0.924     8.000  0.0100      9  ok
  1000    8.00    0.924     8.000  0.0100      8  ok
  1000   12.00    0.000     9.966  0.0100      3  clamped: max reachable 9.966
  1000   12.00    0.000     9.966  0.0100      3  clamped: max reachable 9.966
  1000   12.00    0.000     9.966  0.0100      3  clamped: max reachable 9.966
 10000    0.50   30.000     1.798  0.1000      4  clamped: min reachable 1.798
 10000    0.50   30.000     1.798  0.1000      4  clamped: min reachable 1.798
 10000    0.50   30.000     1.798  0.1000      4  clamped: min reachable 1.798
 10000    2.00    5.169     2.000  0.1000     24  ok
 10000    2.00    5.169     2.000  0.1000     22  ok
 10000    2.00    5.169     2.000  0.1000     22  ok
 10000    4.00    1.957     4.000  0.1000     24  ok
 10000    4.00    1.957     4.000  0.1000     25  ok
 10000    4.00    1.957     4.000  0.1000     24  ok
 10000    8.00    1.205     8.000  0.1000     32  ok
 10000    8.00    1.205     8.000  0.1000     30  ok
 10000    8.00    1.205     8.000  0.1000     30  ok
 10000   12.00    0.758    12.000  0.1000     35  ok
 10000   12.00    0.758    12.000  0.1000     36  ok
 10000   12.00    0.758    12.000  0.1000     36  ok
```

**Reachable range.** For `n = 100 000`: k = 1 000 reaches **0.180 to 9.966
bits**; k = 10 000 reaches **1.798 to 13.288 bits**. The max in both cases is
exactly `log2(k)` (the `s = 0` uniform profile), as expected. The min is
*not* 0 or close to it, because the profile is required to give every one of
the k support values at least 1 occurrence (`powerlaw_counts` adds `+ 1` to
every rank after the largest-remainder allocation of the remaining `n - k`
rows). Even at the most extreme exponent tested (`s = 30`, effectively all
of the `n - k` extra rows going to rank 1), the heavy rank still contributes
its own entropy term, plus the `k - 1` singleton values at probability `1/n`
each. The floor is `H_floor = -p_heavy*log2(p_heavy) - (k-1)*(1/n)*log2(1/n)`,
where `p_heavy = (n-k+1)/n` is the heavy rank's probability once it has
absorbed all `n - k` extra rows plus its own guaranteed 1. For k = 1 000:
p_heavy = 99001/100000 = 0.99001, giving `-p_heavy*log2(p_heavy)` = 0.01434
bits from the heavy rank and `999 * (1/100000) * log2(100000)` = 0.16593
bits from the 999 stragglers (each at p = 1e-5 against log2(1e5) ≈ 16.6 bits
of surprisal), summing to 0.01434 + 0.16593 = 0.18027 bits — matching the
script's reported floor. For k = 10 000: p_heavy = 90001/100000 = 0.90001,
giving 0.13679 bits from the heavy rank and
`9999 * (1/100000) * log2(100000)` = 1.66080 bits from the 9 999 stragglers,
summing to 0.13679 + 1.66080 = 1.79759 bits — again matching. The heavy
rank's own term is not negligible: it is about 8% of the total in both
cases, so dropping it (i.e. approximating the floor as just the stragglers'
contribution) systematically understates the true floor. This floor grows
with k, so for a fixed `n` the achievable *low* end of the entropy range
shrinks as cardinality rises — the target profile can never look "almost
constant" once cardinality is high enough that the mandatory singletons
dominate the entropy sum.

**Bisection.** Converged for every non-clamped target: achieved matched
target to 3 decimals in the printed table, and a closer check showed the
residual is on the order of 1e-5 bits (e.g. k = 1000, target = 8.0 →
achieved 8.000033468; k = 10000, target = 2.0 → achieved 2.000050987). 60
iterations over the `[0, 30]` bracket shrink the interval to ~2.6e-17, far
below float64 resolution, so bisection itself is not the limiting factor —
the residual comes from `largest_remainder`'s integer rounding of counts,
which can only hit a discrete set of achievable entropies near the target,
not the target exactly. Clamped targets (k = 1000/target = 12.0,
k = 10000/target = 0.5) correctly reported the reachable bound instead of
pretending to converge. Runtime per (k, target, seed) row was 3-36 ms,
comfortably under the "well under 1 s" bar; the `rng.choice(9_000_000, ...) +
1_000_000` form (used in place of the brief's
`rng.choice(np.arange(1_000_000, 10_000_000), ...)`) kept the k = 10 000
rows in the same tens-of-ms range as k = 1 000 despite drawing 10x more
distinct values.

**`entropy_bits(x)` vs `entropy_of_counts(counts)`.** Not exactly bit-identical:
diffs were 0 in some rows and ~4.4e-16 to 8.9e-16 in others (e.g. k=1000,
target=2.0: 1.9999892803545598 vs 1.9999892803545603). This is floating-point
summation-order noise, not a bug: `entropy_of_counts` sums over `counts` in
rank order, while `entropy_bits` sums over `np.unique(x)`'s sorted-by-value
order after the values have been shuffled and relabelled to random integers
in `expand()` — the same multiset of probabilities added in a different
order accumulates rounding differently at the ULP level. This is roughly
100x smaller than the ~1e-5 bisection residual above, so it never affects
the printed 3-decimal agreement and is not a concern for tolerance purposes.

**Use-case-2 discrepancy.** The brief notes the spec's use case 2 states
"cardinality ratio 0.1 (i.e. up to 1000 unique values)". For `n = 100 000`,
a cardinality ratio of 0.1 means k = 10 000 (0.1 x 100 000), not 1 000 — the
"1000 unique values" phrasing corresponds instead to a ratio of 0.01. This
run tested both k = 1 000 and k = 10 000 so the discrepancy doesn't block
anything here, but the spec text is internally inconsistent and whoever
resolves it needs to say explicitly which value (ratio 0.1 → k = 10 000, or
"1000 unique values" → k = 1 000, ratio 0.01) is the intended target for use
case 2 before the real generator is built; this spike does not resolve which
one is "correct", only that they disagree.

**Tolerance recommendation.** ±0.01 bits (as the brief suggested) is safe and
has ample margin: observed bisection residuals were ~1e-5 bits, three orders
of magnitude tighter than that tolerance, across every non-clamped (k,
target) combination tested. Recommend the real generator's acceptance check
use ±0.01 bits, and additionally check for the clamped/"ok" status string so
that a request for < 0.18 bits at k = 1 000 (or < 1.798 bits at k = 10 000
for this n) is reported as an achievable-bound violation rather than a near
miss.

## 2. Runs z-score (spike_runs.py)

Full output (n = 100 000, seeds 0/1/2, targets in
{-0.25, -5.0, 3.0, -50.0, 50.0, -400.0}):

```
 target z seed  R target   R got  achieved z multiset ok
    -0.25    0     43380   43380      -0.248        True
    -0.25    1     49935   49935      -0.250        True
    -0.25    2     46819   46819      -0.253        True
    -5.00    0     42728   42728      -4.997        True
    -5.00    1     49184   49184      -5.002        True
    -5.00    2     46116   46116      -4.997        True
     3.00    0     43826   43826       3.001        True
     3.00    1     50449   50449       3.003        True
     3.00    2     47301   47301       3.000        True
   -50.00    0     36550   36550     -49.999        True
   -50.00    1     42073   42073     -50.000        True
   -50.00    2     39448   39448     -50.000        True
    50.00    0     50278   50278      49.999        True
    50.00    1     57876   57876      50.000        True
    50.00    2     54265   54265      50.000        True
  -400.00    0         2       2    -316.223        True
  -400.00    1         2       2    -316.223        True
  -400.00    2         2       2    -316.223        True
```

`R got` equals `R target` on every row without exception, including the
clamped -400 rows, so `layout_runs` always hits the integer run count it
is asked for.

**Achieved precision.** Restricting to the 15 non-clamped rows (the -400
target is a deliberate out-of-range probe, not a precision test — see
below), the largest `|target − achieved|` observed was **0.00301**, on the
z = -5.0, seed = 0 row (n1 = 31 852, n2 = 68 148, sd = 137.283, achieved
z = -4.996988). That row's per-run z step is `1/sd = 1/137.283 = 0.007284`;
0.00301 is 41% of that step, comfortably under the theoretical worst case of
half a step (0.00364) that rounding `mu + z*sd` to the nearest integer R can
produce. The other seeds in the run had larger sd (148.2, 158.0) and smaller
per-run steps (0.00675, 0.00633) with correspondingly smaller max errors
(≤0.00288). So the achieved z's distance from the target is bounded by
`0.5/sd`, because `target_run_count` rounds the continuous target run count
to the nearest achievable integer and `layout_runs` then hits that integer
exactly (confirmed by `R got == R target` on every row) — the only source of
error is that one rounding step, not the block-composition machinery.

**Reachable z range.** For n = 100 000 with a roughly balanced dichotomy
(seed 1: n1 = 48 849, n2 = 51 151, mu = 49 974.5, sd = 158.03), the reachable
range is **z = -316.22 at R = 2** to **z = +302.00 at R = r_max = 97 699**
(r_max = 2*min(n1,n2)+1 since n1 != n2). The -400 target sits well outside
this, so `target_run_count`'s clamp to R = 2 is exercised exactly as
intended, and the achieved z lands at -316.22 (seed 1) / -316.22 (all three
seeds, since the clamp is always to R = 2 and mu/sd happen to scale so the
three seeds' `(2-mu)/sd` values agree to 3 decimals: -316.2225, -316.2230,
-316.2228). This confirms the brief's expectation: **the achieved z at the
-400 target sits at the clamp (~-316), not anywhere near -400**, and the
reachable range this implies is roughly ±300 for a near-balanced split at
this n — a target z outside that band is simply not achievable by block
layout and must be reported as a clamped/reachable-bound violation rather
than treated as a near miss.

**The band is asymmetric, and ±300 is only the near-balanced case.** The
negative bound is essentially fixed, because R = 2 is reachable for any
split: `(2 - mu)/sd` is -316.22 on all three seeds. The positive ceiling is
`(r_max - mu)/sd` with `r_max = 2*min(n1,n2) + 1`, so it shrinks as the split
moves away from 50/50. Computed from this run's own group sizes: **+302.00 at
seed 1's 48 849/51 151 split, +189.42 at seed 2's 37 461/62 539, and only
+147.80 at seed 0's 31 852/68 148** — under half the ±300 figure on the most
lopsided of the three seeds. So ±300 is the best case at this n, not the
typical one, and both bounds must be computed from the measured `n1, n2`
rather than quoted as ±300.

**Multiset preservation.** `multiset ok` (sorted-array equality AND identical
`entropy_bits`) is `True` on all 18 rows, with no exceptions, including the
extreme -400 clamp case. `layout_runs` only ever slices and concatenates the
two group arrays (`a_vals`, `b_vals`), which are themselves permutations
(`rng.permutation`) of `x[b]` and `x[~b]`; every element of `x` appears
exactly once in the output. This is the load-bearing result for the
two-stage generator design: **the block layout is confirmed to be a pure
permutation of the multiset — it does not add, drop, or alter any value's
frequency.** Stage two (layout) can be built on `layout_runs` without
touching the value-frequency contract that stage one establishes.

**Median dichotomy with heavy ties.** The synthetic multiset is heavily
skewed (~1000 support values with weights `rank^-1.2`), so the single most
common value can account for a large fraction of the rows sitting exactly at
the median. Observed tie counts: seed 0 had 23 108 of 100 000 rows equal to
the median (5 613 871), seed 2 had 22 992 equal to its median, and seed 1
had only 6 209 (the median happened to land on a less populous value for
that seed's random support draw, not because the underlying weight profile
differs — all three seeds use the same `rank^-1.2` weights). Because
`dichotomise`'s `kind="median"` uses strict `x > median` — putting every
tied value in the
"below" (False) group — this pushes those ties entirely into n2 and produces
a lopsided split rather than the ~50/50 the runs test assumes: seed 0 was
31 852 / 68 148 (32%/68%), seed 2 was 37 461 / 62 539 (37%/63%), while seed 1
(fewer ties) was a near-balanced 48 849 / 51 151 (49%/51%). This matters
because `runs_expected`'s mu and sd (and hence r_max and the reachable z
range) depend on n1*n2, which shrinks as the split moves away from 50/50 —
seed 0's r_max was correspondingly smaller than seed 1's despite equal n.
Any downstream tolerance or reachable-range calculation must use the actual
post-dichotomy n1/n2 for the specific multiset, not an assumed 50/50 split,
because heavy ties at the median (common for skewed integer/categorical
data) can shift the balance substantially.

**Refine pass.** Not needed for this metric. `R got` matched `R target`
exactly on every single row (18/18), and the resulting achieved z was always
within one rounding step of the requested target (≤0.00301, well under half
of the largest per-run step of 0.00728 seen in this run). `layout_runs`'s
one-shot construction (dichotomise, invert mean/sd for R, lay out R
compositions) is exact and needs no iterative correction, unlike the
entropy spike's bisection search.

## 3. Inter-arrival CV (spike_interarrival.py)

Full output (n = 100 000, seeds 0/1/2, cardinality ratio 0.75, cv targets in
{0.5, 1.0, 2.0}):

```
--- inter-arrival alone ---
 mode target cv seed   runs achieved cv   runs_z
 free      0.50    0  21226       0.288   -78.85
 free      0.50    1  21246       0.289   -79.16
 free      0.50    2  21257       0.290   -78.20
 free      1.00    0  21226       0.554   -77.93
 free      1.00    1  21246       0.550   -79.30
 free      1.00    2  21257       0.553   -78.91
 free      2.00    0  21226       1.099   -78.05
 free      2.00    1  21246       1.086   -80.29
 free      2.00    2  21257       1.092   -78.16
start      0.50    0  21226       0.468   -78.09
start      0.50    1  21246       0.471   -79.11
start      0.50    2  21257       0.472   -80.41
start      1.00    0  21226       0.797   -78.21
start      1.00    1  21246       0.791   -80.06
start      1.00    2  21257       0.795   -77.83
start      2.00    0  21226       1.342   -78.84
start      2.00    1  21246       1.318   -78.85
start      2.00    2  21257       1.321   -78.55
--- combined with runs z ---
 target z target cv seed   runs achieved cv achieved z
    -0.25      0.50    0  11576       0.771      -0.25
    -0.25      0.50    1  11535       0.771      -0.25
    -0.25      0.50    2  11499       0.766      -0.25
    -0.25      1.00    0  11571       0.876      -0.25
    -0.25      1.00    1  11630       0.880      -0.25
    -0.25      1.00    2  11582       0.879      -0.25
    -0.25      2.00    0  11505       1.096      -0.25
    -0.25      2.00    1  11566       1.086      -0.25
    -0.25      2.00    2  11567       1.085      -0.25
   -20.00      0.50    0  12146       0.754     -20.00
   -20.00      0.50    1  12211       0.752     -20.00
   -20.00      0.50    2  12230       0.744     -20.00
   -20.00      1.00    0  12136       0.873     -20.00
   -20.00      1.00    1  12167       0.884     -20.00
   -20.00      1.00    2  12272       0.891     -20.00
   -20.00      2.00    0  12169       1.090     -20.00
   -20.00      2.00    1  12208       1.098     -20.00
   -20.00      2.00    2  12188       1.113     -20.00
```

All `assert np.array_equal(np.sort(y), np.repeat(support, counts))` (alone) and
`assert np.array_equal(np.sort(x), np.sort(y))` (combined) checks passed on
every one of the 36 rows; `layout_interarrival` and `layout_combined` are
confirmed pure permutations of their input multisets.

**Target vs achieved CV — which mode is closer.** Averaging the achieved CV
over the three seeds for each (mode, target) cell:

```
mode  target   mean achieved   mean |error|   relative error
free    0.50          0.289          0.211           42.1%
free    1.00          0.553          0.448           44.8%
free    2.00          1.092          0.908           45.4%
start   0.50          0.471          0.030            5.9%
start   1.00          0.794          0.206           20.6%
start   2.00          1.327          0.673           33.6%
```

`start` is closer than `free` at every target, and the margin is largest at
the low end: at target 0.5, `start`'s relative error is 5.9% versus `free`'s
42.1% — a 7x improvement. `free`'s relative error is nearly constant
(42–45%) across all three targets because it draws the *free*-gap length
directly from the gamma and never accounts for the run itself occupying part
of the start-to-start gap; since mean run length is a roughly fixed fraction
of the mean start-to-start gap at this cardinality, ignoring it produces a
roughly constant proportional undershoot regardless of target. `start`
instead draws the start-to-start gap, subtracts the run length, and clips
negative results to 0 (`np.maximum(gaps - run_len, 0.0)`); at low target CV
the gamma draws are tightly clustered around the mean so almost none go
negative and the correction is nearly exact (5.9% miss), but as target CV
rises the shape parameter `1/cv^2` shrinks (0.25 at cv=2.0), producing a
much more heavy-tailed/skewed gamma where a large fraction of draws fall
below the run length and get clipped to exactly 0 — this truncates the left
tail of the gap distribution, suppresses variance, and drives the achieved
CV further under target as the target rises (33.6% relative miss at
cv=2.0). Both modes always *undershoot*; neither ever overshoots the
target, in any of the 18 alone rows.

**Repeat-run count and mean gap at cardinality 0.75.** The number of
*repeat-runs* (m — maximal stretches of one identical value, length ≥ 2 — not
the Wald-Wolfowitz boolean run count R that the `runs_z` column is computed
from) is a property of the multiset alone (values with count ≥ 2), not of the
layout method or the CV target: for a given seed it is identical across both
modes and all three CV targets — 21 226 (seed 0), 21 246 (seed 1), 21 257
(seed 2).
Direct measurement of the achieved gaps confirms the mean start-to-start gap
is essentially constant at **4.711** rows across every (mode, cv) cell
(4.7109–4.7112, agreeing with `n/m = 100000/21226 = 4.711`), while only the
*spread* of the gaps changes with mode/target (e.g. `start`, cv=0.5: std
2.203, min 2, max 20; `start`, cv=2.0: std 6.324, min 2, max 108). This
confirms the brief's premise: at this cardinality the mean gap is only
about 5 rows, and every gap is an integer ≥ 2 (the smallest possible gap is
one run length). A CV target near 0 would require nearly every gap to equal
the same integer near 4–5; the discreteness alone puts a nonzero floor on
the achievable low end (e.g. alternating gaps of 4 and 5 already gives
`std/mean = 0.5/4.7 ≈ 0.11`), and both modes' undershoot at the tested low
target (0.5) is consistent with that floor pulling achieved CV down from
below rather than being a defect specific to one mode.

**Note on the "alone" `runs_z` column.** The brief's step 2 expected this to
sit "near 0 (random with respect to the median)"; instead it is consistently
around -78 to -80 across every mode/target/seed. This was checked and is
not a bug: a control run that applies a fully free `rng.permutation` to the
same multiset (no forced adjacency of repeated values) gives z = -0.34, in
line with the brief's expectation. `layout_interarrival` necessarily forces
every occurrence of a repeated value into one contiguous block — that is
the entire point of the spike — and doing so removes internal alternation
opportunities that a free permutation of the same n rows would have had.
`runs_expected`'s mu/sd formula assumes n freely-reorderable individual
rows, but the actual number of independently placed units here is only
`m + S + 1` (≈ 75 000, not 100 000, at this cardinality), so the achieved
Wald-Wolfowitz run count falls far below the naive per-row expectation even
though the block order itself is randomised. This is the same "forced adjacency"
mechanism that later explains the combined section's run-splitting effect,
just observed from the other direction.

**Combined results.** `runs_z` is essentially undisturbed by the interarrival
layering: for a given (z, seed), the achieved z is identical to 4 decimal
places across all three CV targets (e.g. z=-0.25 → achieved -0.2530 for
cv ∈ {0.5, 1.0, 2.0}; z=-20.00 → achieved -19.9983 for all three), because
`layout_combined` computes `pattern = dichotomise(layout_runs(x, z, rng))`
*before* consuming any CV-dependent randomness, so the block pattern (and
hence z) is fixed independently of what `layout_interarrival` later does
inside it. The residual miss (0.0030 at z=-0.25, 0.0017 at z=-20.00) is
in the same ≤0.003 range reported for the runs spike alone (section 2), so
layering does not degrade z.

The interarrival CV, however, is degraded substantially by block splitting.
Compared to `start`-mode alone (mean achieved 0.471 / 0.794 / 1.327 for
target 0.5 / 1.0 / 2.0), the combined achieved CV (averaged over z ∈
{-0.25, -20.00} and seeds) is 0.759 / 0.881 / 1.095 for the same three
targets — every value moves *toward 1* relative to its alone counterpart:
+0.288 for target 0.5 (0.471 → 0.759) and −0.232 for target 2.0
(1.327 → 1.095). The surviving repeat-run count also drops sharply: alone
m ≈ 21 226–21 257, combined observed repeat-runs ≈ 11 499–12 272, a 42–46%
reduction. The mechanism is that `out[pattern] = layout_interarrival(...)`
fills the scattered `True`
positions of the runs-test block pattern with one contiguous interarrival
sequence, in left-to-right order; wherever a block boundary of `pattern`
falls in the middle of what would have been a repeat-run in that sequence,
the run is physically split apart in the final array (the two halves are no
longer adjacent). Checking the actual numbers for this multiset (n1≈n2≈
50 000, mu≈50 001, sd≈158.1): the *unconstrained maximum* Wald-Wolfowitz run
count is r_max≈99 999–100 000, reached at z≈+316 — which is the magnitude of
section 2's **negative** bound (-316.22 at R = 2), not section 2's positive
ceiling of +302.00. The difference is the dichotomy balance, not an empirical
ceiling being reached: section 2's seed 1 split 48 849/51 151, so its
`r_max = 2*min(n1,n2)+1 = 97 699` fell short of n, whereas this section's
balanced split puts r_max at n itself. Either way it is nowhere near either z
tested here. The two tested targets,
z=-0.25 and z=-20.00, give R=49 961 and R=46 839 respectively: both sit at
or just below the *random* level (mu), not near the maximum, giving mean
block lengths of n/R = 2.002 and 2.135. A more negative z means *fewer*
runs, hence *longer* blocks, not shorter ones — the direction is the
opposite of "aggressive z gives short blocks". Since both mean block
lengths (2.00–2.14) are still well under the ~4.7-row mean gap between
repeat-runs, nearly every repeat-run's *interior* has a block boundary
running through it at either z, and since most runs have length 2,
splitting one in half destroys it entirely (both halves become singletons,
below `repeat_runs`'s length-≥2 threshold). This is confirmed directly by
the table: *more* repeat-runs survive at the longer-block z=-20.00
(12 136–12 272) than at the shorter-block z=-0.25 (11 499–11 630) — longer
blocks cut fewer repeat-runs, consistent with the corrected direction, not the
inverse. Because survival is largely decoupled from the CV-arranged gap
structure, it pulls
the observed gap statistic of the *survivors* toward the memoryless CV≈1
value regardless of which side of 1 the target sat on, and the more
negative z (longer blocks) shows mildly *less* distortion (e.g. target 0.5:
0.769 at z=-0.25 vs 0.750 at z=-20.00, averaged over seeds — see the
per-z, per-cv table above). Quantitatively, at z=-0.25 — already the
near-random default, not an extreme — a target of 0.5 is pulled up by
about +63% relative (0.471→0.769) and a target of 2.0 is pulled down by
about −18% relative (1.327→1.089). Large *positive* z (which shortens
blocks further, below the already-damaging ~2-row length) was not tested
here and would be expected to make the splitting worse, not better.

**Is a refine pass needed?** Yes, for CV specifically — unlike entropy
(bisection, section 1) and runs z (exact one-shot, section 2), the gamma
method has no closed-form inverse here because of the compounding integer
rounding (`largest_remainder`) and, for `start` mode, the negative-clipping
of `gaps - run_len`. But the miss is *monotonic* in the input CV parameter
(higher input cv reliably produces higher output cv within a mode, as seen
across 0.5→1.0→2.0 for both modes above), so the same bisection strategy
used for entropy would work: treat the gamma's input CV as the search
variable, bisect against `interarrival_cv` of the generated output the same
way spike 1 bisects the power-law exponent against `entropy_bits`. This
spike does not implement that bisection (the brief calls for measuring the
miss, not correcting it), but the monotonicity observed here is the
evidence that such a pass would converge. A rescaling shortcut (e.g.
dividing the constant ~44% miss out of `free` mode) would not generalize,
because `start` mode's miss is not constant — it grows from 5.9% to 33.6%
across the tested range — so only an iterative/bisection correction, not a
single multiplicative fudge factor, is safe to recommend.

**Tolerance recommendation.** No fixed absolute tolerance is safe across
this cardinality range: `start` mode alone misses by 0.03 absolute (5.9%)
at cv=0.5 but by 0.67 absolute (33.6%) at cv=2.0, and combined with a runs
pattern the miss is dominated by block-splitting rather than the gamma
method itself. Recommend: (1) always run a bisection refine pass on the
input CV parameter before accepting, mirroring the entropy spike; (2) after
refinement, a relative tolerance of **±10%** of target CV is a reasonable
acceptance band for the alone case (the refine pass should comfortably beat
the 5.9%–33.6% raw misses measured here); (3) when CV is combined with a
runs z target, do not expect the same tolerance to hold in the *default*
case — mean block length is roughly n/R, which is ≈2.0–2.1 rows at any
near-random z (R near mu), and that is already below the ~4.7-row mean
inter-run gap, so run-splitting distortion (18%–63% relative, measured at
z=-0.25, the near-random level) is the common case, not an edge case
reached only at extreme settings. Large negative z lengthens blocks and
mildly *reduces* the distortion (the surviving-run counts and achieved-CV
values are both slightly better at z=-20.00 than at z=-0.25 in the table
above); large positive z shortens blocks further and would be expected to
*worsen* it, though that direction was not tested here. Flag this
combination explicitly: the real generator should either accept degraded
CV fidelity whenever CV and runs z are combined at this cardinality, warn
most strongly for positive/near-random z rather than negative z, or
document that the two targets trade off against each other at high
cardinality regardless of how moderate the z target looks.

## 4. Lognormal shape + index of dispersion (spike_lognormal.py)

Full output after the fix described below (n = 100 000, k = 75 000,
mean = 4.0, shift = -1.0, clip [-1, 10], 8 dp, seeds 0/1/2):

```
--- shape scan ---
 sigma seed    mean clipmean      ID    card  clipped% shortfall
  0.25    0   2.915   2.9997   0.277  0.7499      0.00        13
  0.25    1   2.911   2.9939   0.276  0.7499      0.00         7
  0.25    2   2.914   2.9980   0.275  0.7500      0.00         4
  0.50    0   2.615   2.9727   1.085  0.7409      1.20       909
  0.50    1   2.608   2.9602   1.083  0.7413      1.16       874
  0.50    2   2.615   2.9694   1.089  0.7414      1.14       860
  1.00    0   1.447   2.4983   3.268  0.7016      6.45      4845
  1.00    1   1.432   2.4767   3.261  0.7020      6.40      4804
  1.00    2   1.442   2.4924   3.266  0.7016      6.45      4840
--- solve sigma for target index of dispersion ---
target ID seed   sigma  achieved  status
     0.50    0  0.3297    0.5000  ok
     0.50    1  0.3317    0.5000  ok
     0.50    2  0.3309    0.5000  ok
     1.00    0  0.4756    1.0000  ok
     1.00    1  0.4765    1.0000  ok
     1.00    2  0.4752    1.0000  ok
     2.00    0  0.7359    2.0000  ok
     2.00    1  0.7348    2.0000  ok
     2.00    2  0.7351    2.0000  ok
     5.00    0  1.2586    5.0000  ok
     5.00    1  1.2587    5.0000  ok
     5.00    2  1.2562    4.9999  ok
```

**A defect found and fixed in `solve_sigma`.** The shape scan above is
unchanged from the first run, but the solve section is not: the brief's
literal code (`id_lo, id_hi = id_at(s_lo), id_at(s_hi)`, then clamp if
`target_id >= id_hi`) reported every one of the 12 (target, seed)
combinations above as "clamped: max reachable" at a *negative* achieved
value (-2.15 to -2.18), even for a target as small as 0.5. That was not a
real reachable-bound result; it was a defect in the bracket, and it is
fixed in the committed script (`spikes/spike_lognormal.py`), not just
reported.

*Root cause.* `raw = lognormal(mu_log, sigma_log) + shift` is strictly
greater than `shift = -1` for every draw (a lognormal variate is strictly
positive), so the clip's lower bound is structurally unreachable and all
clipping mass comes from the upper bound (`hi = 10`) alone. As sigma rises,
both a growing share of the mass being capped at `hi` and, more strongly at
low-to-moderate sigma, the density-weighted count allocation inside
`lognormal_multiset` (detailed in the "Clipped fraction and mean drift"
bullet below) pull the *final multiset mean* down — clipping alone is not
the dominant driver except at the largest sigmas. A fine sigma scan at
seed 0 shows the (final, multiset) mean falling monotonically from 3.0 at
sigma → 0 through 2.915 (sigma 0.25), 1.447 (sigma 1.0), 0.220 (sigma
1.75), 0.0064 (sigma 1.95), crossing zero between sigma = 1.95
(mean = +0.0064) and sigma = 1.98 (mean = -0.0211) — call this the
*mean-crossing sigma*, roughly 1.95-1.98 depending on seed. Over that same
range the variance stays positive and of comparable magnitude throughout
(3.2-4.7). Since `index_of_dispersion = var / mean`, this makes ID diverge
to +infinity as sigma approaches the crossing from below (measured values:
16.9 at sigma 1.75, 59.6 at 1.90, 516.9 at 1.95), then flip to a large
*negative* number just past the crossing (-154.5 at sigma 1.98) and rise
(become less negative) again from there (-81.97, -11.35, -5.52, -3.23,
-2.15 at sigma 2.0/2.25/2.5/2.75/3.0). So `index_of_dispersion(sigma)` is
monotone increasing on `(0, ~1.95)` and, separately, monotone increasing
again on `(~1.98, 3.0)` — but these are two disjoint branches separated by
a sign-flipping discontinuity at the mean-crossing, not one monotone
function over the brief's default bracket `[0.01, 3.0]`. Because the
brief's `s_hi = 3.0` lands on the far (negative) side of that crossing,
`id_hi` comes out negative, so `target_id >= id_hi` is trivially true for
every positive target and the code reports "clamped" without ever
bisecting — a false clamp, not a genuine reachable-bound result.

*The fix.* `solve_sigma` now detects this directly and principledly: before
bisecting on ID, it computes the data mean at `s_lo` and `s_hi` (via a new
internal `stats_at` helper that returns both mean and ID from the same
generated multiset). If the mean at `s_lo` is already non-positive, the
bracket is invalid from the start and the function returns immediately
with an `"invalid bracket: ..."` status instead of guessing. If the mean at
`s_hi` is non-positive (the case that broke the brief's literal code),
the function bisects *on the sign of the data mean itself* — not on ID —
to find the largest sigma below the caller's `s_hi` where the mean is
still positive, and uses that as the effective upper bracket before
proceeding to the ordinary ID-target bisection. This has no hardcoded
upper-bound literal (no "1.7", no fixed backoff margin): the narrowed
bound is derived from `mean`, `shift`, `lo`, `hi` and `seed` exactly as
provided by the caller, so it stays correct for other parameter
combinations where the crossing sits at a different sigma. If even this
narrowing fails to find a positive-mean sigma (defensive branch, not
expected to fire for this task's parameters), the function again returns
an `"invalid bracket: ..."` status rather than fabricating a clamp.

*Result.* With the fix, all four targets (0.5, 1.0, 2.0, 5.0) now report
status `"ok"` and solve to within 0.0001 of target across all three seeds
(see the corrected table above), exactly matching the brief's expected
outcome ("hits targets 0.5, 1.0, 2.0 to within 0.01"). None of these four
targets is genuinely out of range for this configuration — separately
verified (not part of the committed script's `main()`, which still only
exercises the brief's four targets) by calling `solve_sigma` directly with
deliberately extreme targets: `target_id = 1e6` solves normally (status
`"ok"`) for seeds 0 and 1 but genuinely clamps for seed 2
(`"clamped: max reachable 629230.853 at sigma=1.9537 ..."`, since that
seed's mean-crossing sigma happens to sit slightly lower); `target_id =
1e12` genuinely clamps for all three seeds, with reported bounds ranging
629 231 to 2 598 948 depending on seed. This confirms the three statuses
are each independently reachable and behave correctly: a real "ok" solve
(the four brief targets), a real "clamped" result at a true bound (the
1e6/1e12 probes), and a real "invalid bracket" result (confirmed with a
separate pathological parameter set, `shift = -10, lo = -20, hi = 20`,
where the mean is already negative at `s_lo`, returning
`"invalid bracket: mean already <= 0 at s_lo=0.01 (mean=-6)"`).

**"Mean 4" interpretation.** Per the brief, `mean` is the mean of the
lognormal variate *before* `shift` is applied: `mu_log = ln(mean) -
sigma_log**2 / 2`, so that `exp(mu_log + sigma_log**2/2) == mean` exactly
(the standard lognormal mean identity), and the shift is applied afterward
to the drawn values, not folded into `mu_log`. With `shift = -1`, the *raw
draws'* mean is therefore centred near `mean - 1 = 3`, not 4 — the sample
mean of lognormal draws is an unbiased estimator of the lognormal mean, so
the raw (unclipped) draws are not skewed low: measured directly at seed 0,
`raw.mean()` is 2.9997, 2.9995, and 3.0014 at sigma = 0.25, 0.5, and 1.0
respectively, all within sampling noise of 3.0, not below it. The **final
multiset mean** the shape scan prints (2.915/2.911/2.914 at sigma = 0.25,
falling further at higher sigma) is measurably lower than that, but the gap
is *not* pre-clip skew — it is produced downstream, by clipping and then by
the density-weighted count allocation; see the "Clipped fraction and mean
drift" bullet below for the mechanism and its measured decomposition. This
interpretation and formula (`mu_log = ln(mean) - sigma_log**2/2`, applied
to the variate *before* `shift`) should be recorded verbatim in the real
generator's docstring, since "mean 4" is otherwise ambiguous between the
pre-shift and post-shift variate — but the docstring must not imply that
the achieved multiset mean is simply `mean - 1`; it is not.

**Clipped fraction and mean drift.** The clip is `[-1, 10]` and the shift is
`-1`, so `raw = lognormal(...) + shift` satisfies `raw > -1` for *every*
draw, always (a lognormal variate is strictly positive, so `raw + shift`
can equal `-1` only in the zero-probability limit of a zero draw). The
**lower bound of the clip is structurally unreachable** in this
configuration — all clipping mass comes from the upper bound alone. Direct
inspection at seed 0 confirms this: at sigma = 0.25, 0.5, 1.0 exactly 1,
901, and 4840 raw draws respectively exceeded `hi = 10` and zero draws fell
below `lo = -1`. Clipped fraction rises from 0.001% (sigma 0.25) to 1.2%
(sigma 0.5) to 6.45% (sigma 1.0), and the final multiset mean falls
monotonically (2.915 → 2.615 → 1.447 at seed 0), but **clipping is not the
dominant cause of that fall except at the largest sigmas tested** — an
earlier version of this finding attributed the whole drift to clipping and
that attribution is wrong, corrected here with a measured decomposition.

`lognormal_multiset` now records `info["clipped_draw_mean"]` (the mean of
the clipped-but-not-yet-deduplicated draws) alongside the returned
`support` array, which makes the following three-way decomposition
reproducible directly from the retained script. For each sigma, starting
from the raw (unclipped) draw mean and ending at the final multiset mean,
there are three distinct steps, each independently measured at seed 0:

1. **Clipping** (`raw.mean() - clipped_draw_mean`): capping draws above
   `hi = 10` before rounding.
2. **Deduplication** (`clipped_draw_mean - support.mean()`): `np.unique`
   collapses every draw that rounds to the same 8 dp value — overwhelmingly
   the pile-up of draws exceeding `hi`, which all round to the single value
   `10.00000000` — down to one representative support point, discarding the
   rest of that pile-up's weight from a simple average of the *support*
   values.
3. **Density-weighted count allocation** (`support.mean() - final
   multiset mean`): `counts = largest_remainder(density, n - k) + 1`
   distributes the `n - k` non-unique rows in proportion to the lognormal
   PDF evaluated at each support point. That density is concentrated near
   the distribution's *mode*, which for a right-skewed lognormal sits below
   its mean, so the extra replication weight is disproportionately assigned
   to below-mean support values — this systematically pulls the multiset
   mean below the (unweighted) support mean, with **no clipping required at
   all**.

Measured values (seed 0, n = 100 000, k = 75 000):

| sigma | raw draw mean | clipped draw mean | support mean | multiset mean | clip effect | dedup effect | weight effect |
|---|---|---|---|---|---|---|---|
| 0.25 | 2.9997 | 2.9997 | 2.9997 | 2.9152 | 0.0000 | 0.0000 | 0.0845 |
| 0.50 | 2.9995 | 2.9727 | 2.8874 | 2.6149 | 0.0268 | 0.0853 | 0.2725 |
| 1.00 | 3.0014 | 2.4983 | 1.9810 | 1.4468 | 0.5031 | 0.5173 | 0.5342 |
| 1.95 | 3.0706 | 1.0353 | 0.3916 | 0.0064 | 2.0353 | 0.6437 | 0.3852 |

At sigma = 0.25 the clip and dedup effects are both ~0 (clipped_frac is
only 0.001%), so the **entire 0.0845 deficit is the density-weighting
effect** — i.e. even with essentially zero clipping, the multiset mean
(2.9152) sits measurably below the raw draw mean (2.9997) purely because of
how `largest_remainder` allocates replication weight by density. At
sigma = 1.0 the total drift (raw 3.0014 → multiset 1.4468 = 1.5546) splits
close to evenly across all three mechanisms: clip effect 0.5031, dedup
effect 0.5173, weight effect 0.5342 — clipping is a real contributor here,
but not the sole one, and not even the largest of the three. Only at
sigma = 1.95, where clipped_frac has grown much larger, does the clip
effect (2.0353) clearly dominate the other two (0.6437 and 0.3852) — this
is also the sigma at which the multiset mean (0.0064) is on the verge of
crossing zero (see the defect/fix discussion above). So the relative
importance of the three mechanisms shifts with sigma: density-weighting
dominates at low sigma where clipping is negligible, and clipping only
takes over as the dominant term once clipped_frac becomes large. A reader
predicting the achieved mean as "the shifted distribution mean minus a
clip correction" would be wrong by about 3% at sigma = 0.25 (where clipping
contributes essentially nothing) and by around 70% at sigma = 1.95 (where
the multiset mean is nearly zero); the density-weighting effect must be
accounted for at every sigma, not just introduced as a correction once
clipping becomes visible.

**Reachable ID range under the clip.** With the fix in place, the true
reachable range for ID in the physically meaningful regime (positive data
mean, sigma below the mean-crossing) is **(0, +infinity)**, diverging as
sigma approaches ~1.9-1.98 (seed-dependent) where the final multiset mean
(driven by clipping, deduplication, and density-weighting together, not
clipping alone — see the mean-drift decomposition above) crosses zero —
there is no finite practical ceiling of the kind the
entropy and runs spikes (sections 1-2) found; the ceiling that matters
here is the crossing point itself, past which the metric stops meaning
anything for this clip and shift, not a fixed maximum ID value. Practically,
`solve_sigma`'s narrowing bisection (40 iterations by default) gets close
enough to the crossing that the computed "reachable maximum" is a large but
finite number in roughly the 6x10^5 to 2.6x10^6 range (seed-dependent, per
the 1e6/1e12 probes above) — this number is not a fundamental property of
the distribution, only of how many bisection iterations are spent narrowing
toward the crossing, and callers should not treat it as a fixed constant.
For any target below roughly 10-15 (comfortably below the crossing, reached
at sigma < ~1.3 for all four brief targets), the range is dense and smooth:
bisection converges to within 0.0001 of target.

**Cardinality.** Held close to the 0.75 target only while clipping is
negligible: card = 0.7499-0.7500 at sigma = 0.25 (clipped_frac ≈ 0.001%),
but degrades to 0.7409-0.7414 at sigma = 0.5 and 0.7016-0.7020 at
sigma = 1.0. The support shortfall (`k - len(support)`) is *not* ≈ 0 once
clipping is material: at sigma = 0.25 the shortfall of 13 (seed 0) is
essentially all ordinary 8 dp rounding collisions among the 74 999
unclipped draws (verified directly: 13 natural duplicate pairs, max
duplicate count 2, with the 1 clipped draw contributing nothing extra since
it is the sole draw at that boundary). But at sigma = 0.5 the shortfall of
909 is overwhelmingly the clip collapse: 901 draws exceed `hi` and all
collapse onto the single support value 10.00000000, costing `901 - 1 = 900`
of the 909, with only ~9 residual natural collisions. At sigma = 1.0, 4840
clipped draws cost `4840 - 1 = 4839` of the 4845 shortfall. So the
mechanism is: **clipping, not 8 dp rounding, is what erodes cardinality**
— every draw that lands past `hi` merges into one shared support point
instead of remaining distinct, and the erosion grows directly with
clipped_frac. The density-weighted profile does *not* keep cardinality
pinned at 0.75 once clipped_frac exceeds roughly 1%; the real generator
must either accept a lower achieved cardinality at high sigma for this
clip, or compensate by requesting a larger `k` when it detects a non-
trivial `clipped_frac`.

**Bisection behaviour and monotonicity.** `index_of_dispersion(sigma)` is
*empirically* monotone increasing across the scanned sigma grid on
`(0, ~1.95)` and, separately, monotone increasing again (across negative
values) on `(~1.98, 3.0)` — the evidence is the fine scan reported above, not
a proof — but these are
two disjoint monotone branches separated by a sign-flipping discontinuity
at the mean-zero crossing, not one monotone function over the full
`[0.01, 3.0]` bracket the brief specifies as the default. The fixed
`solve_sigma` handles this by detecting a non-positive mean at `s_hi` and
narrowing the bracket to the positive-mean branch before bisecting on ID;
with that narrowing, bisection converges correctly and precisely (targets
matched to within 0.0001) for every target tested. No non-monotonicity was
found *within* the safe branch itself from the seed-fixed resampling —
`stats_at` is deterministic per (sigma, seed) since the RNG is freshly
seeded on every call, and repeated bisection runs on the same inputs land
on the same converged sigma. The only source of non-monotonicity is the
multiset-mean-crossing sign flip described above, not resampling noise, and it is now
detected and worked around rather than silently mishandled.

**Must solving use the final clipped/rounded multiset, not the analytic
formula?** Yes, decisively. The analytic lognormal variance (pre-clip,
pre-shift) is `var = (exp(sigma^2) - 1) * mean^2`, giving, against the
post-shift mean of 3, analytic ID = 0.344 (sigma 0.25), 1.515 (sigma 0.5),
9.164 (sigma 1.0) — versus the actually measured (clipped, rounded) ID of
0.277, 1.085, 3.268 at the same sigmas. The measured value is already 80%,
72%, and just 36% of the analytic prediction at sigma 0.25/0.5/1.0
respectively, a gap that widens sharply with sigma because clipping removes
exactly the heavy right tail that dominates the lognormal's variance. Above
this the analytic formula does not even have the sign-flip pathology the
clipped version has (the analytic mean is pinned at the constant `mean - 1
= 3` for every sigma, so analytic ID grows without a pole). Solving against
the analytic formula would therefore not just be quantitatively wrong but
qualitatively wrong — it would miss the mean-crossing pole entirely and
give no warning that very large sigma makes the clipped, rounded
multiset's ID behave nothing like the textbook lognormal. Bisection must
run against `index_of_dispersion` measured on the actual
`np.repeat(support, counts)` array, with the bracket confined to a single
monotone branch as the fixed `solve_sigma` now does.

**Tolerance recommendation.** For targets on the safe branch (roughly
ID < 10-15 for this clip/shift/mean combination, reached at sigma < ~1.3
for the tested targets), recommend an absolute tolerance of **±0.01** on
the index of dispersion — the fixed bisection matched all four tested
targets (0.5, 1.0, 2.0, 5.0) to within 0.0001 across all three seeds, two
orders of magnitude tighter. This tolerance is only safe in combination
with the bracket-narrowing guard now in `solve_sigma`: the real generator
must (1) never assume a fixed `s_hi` is on the correct side of the
mean-crossing sigma for an arbitrary mean/shift/clip combination — that
crossing point moves with all three parameters, so it must be detected
per-call, not hardcoded; (2) treat `"invalid bracket"` as a hard error
distinct from a genuine clamp, since it means the request has no valid
solution space at all; and (3) treat a reported "clamped: max reachable"
value as approximate rather than a hard physical ceiling when the target
sits near a mean-crossing, since (per the reachable-range discussion above)
the true supremum there is +infinity and the reported number only reflects
how far the bracket-narrowing bisection was run. Separately, recommend
flagging any request with `clipped_frac` above roughly 1% as "cardinality
at risk": the 0.75 cardinality target degrades measurably (0.7409 at
sigma 0.5, already 1.2% clipped) well before ID approaches its pole, so a
generator tuning purely for ID could silently blow the cardinality target.

## 5. ClickHouse round-trip and codec applicability (spike_ch_roundtrip.py)

Server: ClickHouse **26.8.2.7**, HTTP port 8123, localhost. Client library:
**clickhouse-connect 1.8.0** (pyarrow 24.0.0, numpy 2.3.5). Database `bench`,
100 000 rows, `rng = np.random.default_rng(0)`, `v = np.round(rng.uniform(-1,
10, 100_000), 8)`.

**Revision note.** This section was revised after a review found the
storage-stat query used the brief's literal SQL, which reads *part-level*
totals (repeated identically across every column of the part) rather than
true per-column figures, and that the codec/index rejection reasons were
truncated to 110 characters in a way that could not be reproduced from the
committed script alone. Both are fixed in `spikes/spike_ch_roundtrip.py`
and the numbers/quotes below are from the corrected script's own output,
not from an out-of-band probe. The `type x codec` / `type x skip index`
applicability matrices (the `ok`/`REJECT` grids) are **unchanged** by
either fix, as expected — the fixes only changed what gets printed
alongside them, not what ClickHouse accepts or rejects.

Full script output after the fixes:

```
version: 26.8.2.7
clickhouse_connect: 1.8.0

--- arrow insert: spike_f64 (Float64) ---
spike_f64: inserted (arrow float64) in 154 ms
  spike_f64.v: compressed=805721 uncompressed=800000 marks_bytes=119 rows=100000 parts=1
  spike_f64.row_id: compressed=400518 uncompressed=800000 marks_bytes=114 rows=100000 parts=1

--- arrow insert: spike_d64 (Decimal64(8)) ---
spike_d64: inserted (arrow decimal64(18,8)) in 150 ms
  spike_d64.v: compressed=617602 uncompressed=800000 marks_bytes=117 rows=100000 parts=1
  spike_d64.row_id: compressed=400518 uncompressed=800000 marks_bytes=114 rows=100000 parts=1

--- arrow insert: spike_d32 Part A (Decimal32(8), real data -- expected to fit) ---
Decimal32(8) representable range: -9.99999999 to +9.99999999 (9 significant digits, scale 8). rng.uniform(-1, 10) draws values strictly < 10, but rounding to 8 dp sends any draw in [9.999999995, 10) to exactly 10.00000000, which overflows: the range check must be applied to the rounded values, not the draw bounds.
spike_d32: inserted (arrow decimal32(9,8)) in 91 ms
  spike_d32.v: compressed=401912 uncompressed=400000 marks_bytes=135 rows=100000 parts=1
  spike_d32.row_id: compressed=400518 uncompressed=800000 marks_bytes=114 rows=100000 parts=1

--- arrow insert: spike_d32 Part B (Decimal32(8), single value 10.0 -- expected overflow) ---
spike_d32_overflow: arrow cast/insert of 10.0 failed: ArrowInvalid: Cannot convert 10.000000 to Decimal32(precision = 9, scale = 8): overflow
spike_d32_overflow: plain INSERT VALUES (10.0) failed: DatabaseError: Received ClickHouse exception, code: 69, server response: Code: 69. DB::Exception: Decimal value is too big: 2 digits were read: '10'e0. Expected to read decimal with scale 8 and precision 9:  at row 0: While executing WaitForAsyncInsert. (ARGUMENT_OUT_OF_BOUND) (for url http://localhost:8123)

decimal insert path used: decimal64=arrow_decimal  decimal32=arrow_decimal

--- timed query + query_log ---
  cache-drop attempts:
    SYSTEM DROP MARK CACHE: ok
    SYSTEM DROP UNCOMPRESSED CACHE: ok
    SYSTEM DROP QUERY CONDITION CACHE: ok
cold=True client_ms=60.4 query_log=[(13, 100000, 800000, 5042065, 13, 1, 1, 0, 0, 0, 0)]
cold=False client_ms=58.5 query_log=[(11, 100000, 800000, 5040305, 13, 1, 1, 0, 0, 0, 0)]

--- EXPLAIN indexes = 1 ---
  Output: count(), sum(v)
  
  Aggregating
  │  Keys:
  │  Aggregates: count(), sum(v)
  │  Skip merging: 0
  └──Filter ((WHERE + Change column names to column identifiers))
     │  Filter column: v >= 2 AND v <= 3
     └──ReadFromMergeTree (bench.spike_f64)
           Read type: Default
           Parts: 1 | Granules: 13
           Output: v

--- DDL applicability matrix: type x codec ---
type                           Delta DoubleDelta     Gorilla         T64         FPC
    Float64 + T64: Received ClickHouse exception, code: 431, server response: Code: 431. DB::Exception: T64 codec is not supported for specified type Float64. (ILLEGAL_SYNTAX_FOR_CODEC_TYPE) (for url http://localhost:8123)
Float64                           ok          ok          ok      REJECT          ok
    Decimal32(8) + Gorilla: Received ClickHouse exception, code: 36, server response: Code: 36. DB::Exception: The combination of compression codecs Gorilla(4), LZ4 is meaningless, because it does not make sense to apply a floating-point time series codec to non-floating-point columns (Note: you can enable setting 'allow_suspicious_codecs' to skip this check). (BAD_ARGUMENTS) (for url http://localhost:8123)
    Decimal32(8) + FPC: Received ClickHouse exception, code: 36, server response: Code: 36. DB::Exception: Codec 'FPC' is not applicable for Decimal(9, 8) because the data type is not Float*. (BAD_ARGUMENTS) (for url http://localhost:8123)
Decimal32(8)                      ok          ok      REJECT          ok      REJECT
    Decimal64(8) + Gorilla: Received ClickHouse exception, code: 36, server response: Code: 36. DB::Exception: The combination of compression codecs Gorilla(8), LZ4 is meaningless, because it does not make sense to apply a floating-point time series codec to non-floating-point columns (Note: you can enable setting 'allow_suspicious_codecs' to skip this check). (BAD_ARGUMENTS) (for url http://localhost:8123)
    Decimal64(8) + FPC: Received ClickHouse exception, code: 36, server response: Code: 36. DB::Exception: Codec 'FPC' is not applicable for Decimal(18, 8) because the data type is not Float*. (BAD_ARGUMENTS) (for url http://localhost:8123)
Decimal64(8)                      ok          ok      REJECT          ok      REJECT
    Int32 + Gorilla: Received ClickHouse exception, code: 36, server response: Code: 36. DB::Exception: The combination of compression codecs Gorilla(4), LZ4 is meaningless, because it does not make sense to apply a floating-point time series codec to non-floating-point columns (Note: you can enable setting 'allow_suspicious_codecs' to skip this check). (BAD_ARGUMENTS) (for url http://localhost:8123)
    Int32 + FPC: Received ClickHouse exception, code: 36, server response: Code: 36. DB::Exception: Codec 'FPC' is not applicable for Int32 because the data type is not Float*. (BAD_ARGUMENTS) (for url http://localhost:8123)
Int32                             ok          ok      REJECT          ok      REJECT
    UInt32 + Gorilla: Received ClickHouse exception, code: 36, server response: Code: 36. DB::Exception: The combination of compression codecs Gorilla(4), LZ4 is meaningless, because it does not make sense to apply a floating-point time series codec to non-floating-point columns (Note: you can enable setting 'allow_suspicious_codecs' to skip this check). (BAD_ARGUMENTS) (for url http://localhost:8123)
    UInt32 + FPC: Received ClickHouse exception, code: 36, server response: Code: 36. DB::Exception: Codec 'FPC' is not applicable for UInt32 because the data type is not Float*. (BAD_ARGUMENTS) (for url http://localhost:8123)
UInt32                            ok          ok      REJECT          ok      REJECT
    String + Delta: Received ClickHouse exception, code: 36, server response: Code: 36. DB::Exception: Codec Delta is not applicable for String because the data type is not of fixed size. (BAD_ARGUMENTS) (for url http://localhost:8123)
    String + DoubleDelta: Received ClickHouse exception, code: 36, server response: Code: 36. DB::Exception: Codec DoubleDelta is not applicable for String because the data type is not numeric. (BAD_ARGUMENTS) (for url http://localhost:8123)
    String + Gorilla: Received ClickHouse exception, code: 36, server response: Code: 36. DB::Exception: Codec Gorilla is not applicable for String because the data type is not of fixed size. (BAD_ARGUMENTS) (for url http://localhost:8123)
    String + T64: Received ClickHouse exception, code: 431, server response: Code: 431. DB::Exception: T64 codec is not supported for specified type String. (ILLEGAL_SYNTAX_FOR_CODEC_TYPE) (for url http://localhost:8123)
    String + FPC: Received ClickHouse exception, code: 36, server response: Code: 36. DB::Exception: Codec 'FPC' is not applicable for String because the data type is not Float*. (BAD_ARGUMENTS) (for url http://localhost:8123)
String                        REJECT      REJECT      REJECT      REJECT      REJECT
    LowCardinality(String) + Delta: Received ClickHouse exception, code: 36, server response: Code: 36. DB::Exception: Codec Delta is not applicable for String because the data type is not of fixed size. (BAD_ARGUMENTS) (for url http://localhost:8123)
    LowCardinality(String) + DoubleDelta: Received ClickHouse exception, code: 36, server response: Code: 36. DB::Exception: Codec DoubleDelta is not applicable for String because the data type is not numeric. (BAD_ARGUMENTS) (for url http://localhost:8123)
    LowCardinality(String) + Gorilla: Received ClickHouse exception, code: 36, server response: Code: 36. DB::Exception: Codec Gorilla is not applicable for String because the data type is not of fixed size. (BAD_ARGUMENTS) (for url http://localhost:8123)
    LowCardinality(String) + T64: Received ClickHouse exception, code: 431, server response: Code: 431. DB::Exception: T64 codec is not supported for specified type String. (ILLEGAL_SYNTAX_FOR_CODEC_TYPE) (for url http://localhost:8123)
    LowCardinality(String) + FPC: Received ClickHouse exception, code: 36, server response: Code: 36. DB::Exception: Codec 'FPC' is not applicable for String because the data type is not Float*. (BAD_ARGUMENTS) (for url http://localhost:8123)
LowCardinality(String)        REJECT      REJECT      REJECT      REJECT      REJECT
    Date + Gorilla: Received ClickHouse exception, code: 36, server response: Code: 36. DB::Exception: The combination of compression codecs Gorilla(2), LZ4 is meaningless, because it does not make sense to apply a floating-point time series codec to non-floating-point columns (Note: you can enable setting 'allow_suspicious_codecs' to skip this check). (BAD_ARGUMENTS) (for url http://localhost:8123)
    Date + FPC: Received ClickHouse exception, code: 36, server response: Code: 36. DB::Exception: Codec 'FPC' is not applicable for Date because the data type is not Float*. (BAD_ARGUMENTS) (for url http://localhost:8123)
Date                              ok          ok      REJECT          ok      REJECT
    DateTime64(3) + Gorilla: Received ClickHouse exception, code: 36, server response: Code: 36. DB::Exception: The combination of compression codecs Gorilla(8), LZ4 is meaningless, because it does not make sense to apply a floating-point time series codec to non-floating-point columns (Note: you can enable setting 'allow_suspicious_codecs' to skip this check). (BAD_ARGUMENTS) (for url http://localhost:8123)
    DateTime64(3) + FPC: Received ClickHouse exception, code: 36, server response: Code: 36. DB::Exception: Codec 'FPC' is not applicable for DateTime64(3) because the data type is not Float*. (BAD_ARGUMENTS) (for url http://localhost:8123)
DateTime64(3)                     ok          ok      REJECT          ok      REJECT
    Nullable(Int32) + Gorilla: Received ClickHouse exception, code: 36, server response: Code: 36. DB::Exception: The combination of compression codecs Gorilla(4), LZ4 is meaningless, because it does not make sense to apply a floating-point time series codec to non-floating-point columns (Note: you can enable setting 'allow_suspicious_codecs' to skip this check). (BAD_ARGUMENTS) (for url http://localhost:8123)
    Nullable(Int32) + FPC: Received ClickHouse exception, code: 36, server response: Code: 36. DB::Exception: Codec 'FPC' is not applicable for Int32 because the data type is not Float*. (BAD_ARGUMENTS) (for url http://localhost:8123)
Nullable(Int32)                   ok          ok      REJECT          ok      REJECT

--- DDL applicability matrix: type x skip index ---
type                            minmax  bloom_filter      set(100)
Float64                             ok            ok            ok
    Decimal32(8) + bloom_filter: Received ClickHouse exception, code: 44, server response: Code: 44. DB::Exception: Unexpected type Decimal(9, 8) of bloom filter index: When validating secondary index `ix`. (ILLEGAL_COLUMN) (for url http://localhost:8123)
Decimal32(8)                        ok        REJECT            ok
    Decimal64(8) + bloom_filter: Received ClickHouse exception, code: 44, server response: Code: 44. DB::Exception: Unexpected type Decimal(18, 8) of bloom filter index: When validating secondary index `ix`. (ILLEGAL_COLUMN) (for url http://localhost:8123)
Decimal64(8)                        ok        REJECT            ok
Int32                               ok            ok            ok
UInt32                              ok            ok            ok
String                              ok            ok            ok
LowCardinality(String)              ok            ok            ok
Date                                ok            ok            ok
DateTime64(3)                       ok            ok            ok
Nullable(Int32)                     ok            ok            ok

--- cleanup ---
  dropped bench.spike_f64 (if existed)
  dropped bench.spike_d64 (if existed)
  dropped bench.spike_d32 (if existed)
  dropped bench.spike_d32_overflow (if existed)
  dropped bench.spike_probe (if existed)
```

**Storage stats are per-column now, and the fix required forcing Wide part
format.** The brief's literal SQL summed `data_compressed_bytes` /
`data_uncompressed_bytes` from `system.parts_columns` — those are **part-
level totals repeated identically on every column's row**, not per-column
figures (confirmed by the original run: `spike_f64.v` and `spike_f64.row_id`
reported the identical 1 206 123 / 1 600 000 bytes, which is the sum of
*both* columns — 800 000 (Float64, 8 bytes/row) + 800 000 (UInt64
`row_id`, 8 bytes/row) = 1 600 000 uncompressed, exactly). The true
per-column columns are `column_data_compressed_bytes`,
`column_data_uncompressed_bytes`, and `column_marks_bytes`, and
`report_storage()` now queries those instead.

Switching to the per-column columns alone was not sufficient: on the
first rerun every value came back **0** for all three tables. The cause is
that MergeTree stores a part as **Compact** (all columns interleaved in one
file, no per-column byte accounting at all) unless the part is at or above
`min_bytes_for_wide_part` / `min_rows_for_wide_part` (`system.parts` showed
`part_type = 'Compact'` for these ~1.6 MB, 100 000-row tables, which is
comfortably under the 10 MiB default `min_bytes_for_wide_part`), and
`column_data_compressed_bytes` etc. are only populated for **Wide** parts. This
is itself a finding for the harness: **per-column storage stats are silently
zero, not an error, for any MergeTree table small enough to land in Compact
format** — a benchmark comparing per-column codec effectiveness at a small
row count could easily read all-zero columns and misinterpret it as "no
difference" rather than "wrong part format for this measurement." The
script now creates the three insert tables with `SETTINGS
min_bytes_for_wide_part=0, min_rows_for_wide_part=0` to force Wide format so
the per-column query returns real numbers; the design/harness should force
Wide format (or use a large enough dataset that ClickHouse would choose it
anyway) whenever it needs genuine per-column storage comparisons.

Verified per-column storage, restated (compressed / uncompressed / marks_bytes,
100 000 rows, 1 part each):

| table | column | codec | compressed | uncompressed | marks_bytes |
|---|---|---|---|---|---|
| spike_f64 | v (Float64) | Gorilla, ZSTD | 805 721 | 800 000 | 119 |
| spike_f64 | row_id (UInt64) | default | 400 518 | 800 000 | 114 |
| spike_d64 | v (Decimal64(8)) | Delta, LZ4 | 617 602 | 800 000 | 117 |
| spike_d64 | row_id (UInt64) | default | 400 518 | 800 000 | 114 |
| spike_d32 | v (Decimal32(8)) | Delta, LZ4 | 401 912 | 400 000 | 135 |
| spike_d32 | row_id (UInt64) | default | 400 518 | 800 000 | 114 |

Sanity checks: uncompressed bytes per `v` column match the raw type width
exactly — 800 000 = 8 bytes x 100 000 rows for both Float64 and
Decimal64(8), and 400 000 = 4 bytes x 100 000 rows for Decimal32(8). The
`row_id` column (`UInt64`, no explicit codec, identical data and default
codec in all three tables) reports the *same* compressed (400 518) and
uncompressed (800 000) bytes across all three tables regardless of what
codec its sibling `v` column uses — exactly as expected once the figures
are genuinely per-column, and a useful cross-check that the fixed query is
now correct (the original part-level query could never have shown this,
since it always summed both columns together).

**The three-way storage comparison in the original write-up was confounded
and is withdrawn; only Decimal32 vs Decimal64 supports a comparison.**
`spike_f64.v` uses `Gorilla, ZSTD` while both `spike_d64.v` and
`spike_d32.v` use `Delta, LZ4` — a different codec *and* a different
underlying type family, so any Float64-vs-Decimal difference in the
figures mixes codec choice with type choice and cannot be attributed to
raw width alone. That three-way "cheapest of the three" claim from the
original write-up is withdrawn. What *does* survive, because
`spike_d64.v` and `spike_d32.v` share both the same codec pair (`Delta,
LZ4`) and the same `row_id` sibling column: **Decimal32(8) compresses to
401 912 bytes vs Decimal64(8)'s 617 602 bytes for the same values under
the same codec** — Decimal32 is cheaper here, consistent with its raw
per-row width being half of Decimal64's (4 vs 8 bytes) before compression
even applies. This narrower comparison, not a three-way ranking including
Float64, is the safe takeaway: **Decimal32(8) accommodates this project's
use-case-1 data (see the overflow discussion below) and costs less than
Decimal64(8) to store it, under matching codecs** — no claim is made here
about Decimal vs Float64, which would require an apples-to-apples codec
choice this spike did not run.

**The design spec's own storage-stat column list has the same part-level-
vs-per-column defect and needs the same fix.** `docs/superpowers/specs/
2026-09-13-clickhouse-benchmark-framework-design.md` line 213 reads
"Storage stats from `system.parts_columns`: `data_compressed_bytes`,
`data_uncompressed_bytes`, `marks`, `rows`." All three of
`data_compressed_bytes`, `data_uncompressed_bytes`, and `marks` are
part-level columns (repeated identically on every column's row of the
same part), not a per-column breakdown, so as written the spec would have
the real framework compute the same silently-wrong per-column attribution
this spike's original script did. The spec should be amended to name
`column_data_compressed_bytes`, `column_data_uncompressed_bytes`, and
`column_marks_bytes` in place of those three (`rows` is fine as-is — it is
already per-column in `system.parts_columns`), and should also note the
Wide-vs-Compact part-format dependency above, since a benchmark run at a
small enough scale would otherwise silently read zeros from those columns
too.

**Decimal32(8) is not an overflow case for this project's data — the
brief's premise was wrong, and the corrected two-part probe confirms why.**
`Decimal32(8)` is 9 significant digits at scale 8, so its representable
range is exactly **-9.99999999 to +9.99999999**. `rng.uniform(-1, 10)`
draws values on `[-1, 10)` — strictly less than 10 — but the draw bound is
not on its own sufficient: rounding to 8 dp sends any draw in
`[9.999999995, 10)` to exactly `10.00000000`, which *does* overflow
Decimal32(8). That window is 5e-9 wide against a uniform density of 1/11, so
the expected number of such values in 100 000 draws is about 4.5e-5
(arithmetic from the draw bounds, not a measurement; the run above contained
none, which is why Part A inserted cleanly). **The validator must therefore
range-check the *rounded* values against the type's representable bound, not
assume the draw bounds suffice.** Part A confirms the real dataset fits
empirically: the full dataset, cast to Arrow `decimal32(9, 8)` and
inserted via `insert_arrow`, succeeded in 91 ms with one active part.
**Decimal32(8) comfortably accommodates the project's use-case-1 data**,
and (per the corrected comparison above) is the cheaper of the two Decimal
widths for it under matching codecs — this is itself a load-bearing
finding for the design, which should recommend Decimal32(8) over
Decimal64(8) wherever the value range is known to stay inside
+/-9.99999999.

Part B isolates the true overflow boundary with a single deliberately
out-of-range value, `10.0`, in a dedicated `spike_d32_overflow` table so
that Part A's success is never at risk from Part B's expected failure. Both
the client-side Arrow cast and the server-side plain `INSERT VALUES` were
tried, independently, and **both reject the value**, with two different
but consistent error texts:

- Arrow cast (pyarrow, client-side, before any network call):
  `ArrowInvalid: Cannot convert 10.000000 to Decimal32(precision = 9, scale = 8): overflow`
- Plain `INSERT INTO bench.spike_d32_overflow VALUES (10.0)` (server-side,
  the message `validate.py` should surface verbatim or mimic since it is
  the one a real (non-Arrow-preprocessed) insert path would hit):
  `DatabaseError: Received ClickHouse exception, code: 69, server response: Code: 69. DB::Exception: Decimal value is too big: 2 digits were read: '10'e0. Expected to read decimal with scale 8 and precision 9:  at row 0: While executing WaitForAsyncInsert. (ARGUMENT_OUT_OF_BOUND) (for url http://localhost:8123)`

Both fail at the *client/insert* boundary, not silently truncating or
wrapping — this is the safe failure mode the harness's loader can rely on:
an out-of-range Decimal32 value raises before it reaches storage, rather
than being stored corrupted.

**Arrow insert path per decimal width — no fallback was needed.**
`insert_arrow` accepted the Arrow decimal type directly for both widths on
this server/library combination:

| width | arrow type tried | path that worked |
|---|---|---|
| Decimal64(8) | `pa.decimal64(18, 8)` via `pc.cast` | `arrow_decimal` (direct) |
| Decimal32(8) | `pa.decimal32(9, 8)` via `pc.cast` | `arrow_decimal` (direct) |

Neither the `pc.cast(pa.array(v), pa.decimal64/32(...))` step nor the
subsequent `client.insert_arrow(...)` raised for the real 100 000-row
dataset (only the deliberately-overflowing single value 10.0 in Part B
failed, as expected). **The Float64-staging + `toDecimal64`/`toDecimal32`
fallback path specified by correction 3 exists in the committed script
(`insert_decimal()` in `spikes/spike_ch_roundtrip.py`) but was not
exercised on this run** — record this so the harness's loader knows the
direct Arrow-decimal path is the primary one to try first on
pyarrow 24.0.0 / clickhouse-connect 1.8.0 / ClickHouse 26.8.2.7, with the
staging fallback kept only as a defensive path for other version
combinations where the direct cast or insert might reject Arrow decimals.

**DDL applicability matrix: type x codec** (`CODEC(<codec>, LZ4)`, measured,
verbatim from the run above; 25 of the 50 cells accepted, 25 rejected):

```
type                           Delta DoubleDelta     Gorilla         T64         FPC
Float64                           ok          ok          ok      REJECT          ok
Decimal32(8)                      ok          ok      REJECT          ok      REJECT
Decimal64(8)                      ok          ok      REJECT          ok      REJECT
Int32                             ok          ok      REJECT          ok      REJECT
UInt32                            ok          ok      REJECT          ok      REJECT
String                        REJECT      REJECT      REJECT      REJECT      REJECT
LowCardinality(String)        REJECT      REJECT      REJECT      REJECT      REJECT
Date                              ok          ok      REJECT          ok      REJECT
DateTime64(3)                     ok          ok      REJECT          ok      REJECT
Nullable(Int32)                   ok          ok      REJECT          ok      REJECT
```

Every distinct rejection reason, captured **verbatim and complete** —
the committed script now prints the full exception text for every REJECT
cell (no truncation), so every quote below is reproducible from a single
run of `spikes/spike_ch_roundtrip.py` and does not depend on any
out-of-band probe:

- **`T64`** (code 431, `ILLEGAL_SYNTAX_FOR_CODEC_TYPE`) rejects only
  `Float64`, `String`, and `LowCardinality(String)` (3 of 10 types) —
  **not** "non-integer types" in general, since it is *accepted* on both
  Decimal widths, `Int32`, `UInt32`, `Date`, `DateTime64(3)`, and
  `Nullable(Int32)` (7 of 10 types), all of which ClickHouse stores using
  a fixed-width integer-like internal representation. Exact text:
  `T64 codec is not supported for specified type Float64.` (and the
  analogous `... specified type String.` for both String-shaped types).
- **`Gorilla` on every non-floating type tested** (`Decimal32(8)`,
  `Decimal64(8)`, `Int32`, `UInt32`, `Date`, `DateTime64(3)`,
  `Nullable(Int32)`) (code 36, `BAD_ARGUMENTS`): `The combination of
  compression codecs Gorilla(N), LZ4 is meaningless, because it does not
  make sense to apply a floating-point time series codec to
  non-floating-point columns (Note: you can enable setting
  'allow_suspicious_codecs' to skip this check).` The byte-width parameter
  `N` ClickHouse infers from the column's storage type, confirmed from the
  untruncated text for every case tested: **`Gorilla(2)` for `Date`**,
  **`Gorilla(4)` for `Int32`, `UInt32`, `Nullable(Int32)`, and
  `Decimal32(8)`**, **`Gorilla(8)` for `Decimal64(8)` and
  `DateTime64(3)`** — i.e. `N` tracks the type's actual storage width (4
  bytes for `Decimal32`, 8 for `Decimal64`, matching their names), not a
  fixed value. (An earlier version of this section incorrectly guessed
  `Decimal32(8)` maps to `Gorilla(8)`, unsupported by any measured text at
  the time; the measured value is `Gorilla(4)`, corrected here.) This
  rejection is opt-outable via `allow_suspicious_codecs` — a setting the
  harness could expose but should not enable by default, since the
  message itself calls the combination "meaningless."
- **`FPC` on every non-`Float*` type tested** (code 36, `BAD_ARGUMENTS`):
  `Codec 'FPC' is not applicable for <Type> because the data type is not
  Float*.`, where `<Type>` is the resolved scalar type ClickHouse reports
  (`Int32`, `UInt32`, `Date`, `DateTime64(3)`, or, notably, the *decimal
  storage type* rather than the DDL spelling for the two decimal columns:
  `Decimal(9, 8)` for `Decimal32(8)` and `Decimal(18, 8)` for
  `Decimal64(8)`).
- **`Delta` on `String`/`LowCardinality(String)`** (code 36,
  `BAD_ARGUMENTS`): `Codec Delta is not applicable for String because the
  data type is not of fixed size.`
- **`DoubleDelta` on `String`/`LowCardinality(String)`** (code 36,
  `BAD_ARGUMENTS`): `Codec DoubleDelta is not applicable for String because
  the data type is not numeric.` (a different clause than Delta's, even
  though both reject the same type — `validate.py` should not assume one
  canned message covers every codec-on-String rejection.)
- **`Gorilla` on `String`/`LowCardinality(String)`** (code 36,
  `BAD_ARGUMENTS`): `Codec Gorilla is not applicable for String because the
  data type is not of fixed size.`

`LowCardinality(String)` behaves identically to plain `String` for every
codec in this matrix — ClickHouse reports the rejection in terms of the
*underlying* type (`String`), not the `LowCardinality(...)` wrapper, so
`validate.py`'s rule can key off the dictionary-encoded base type directly.

**DDL applicability matrix: type x skip index** (`INDEX ix v TYPE <index>
GRANULARITY 1`, measured, verbatim; 28 of 30 cells accepted, 2 rejected):

```
type                            minmax  bloom_filter      set(100)
Float64                             ok            ok            ok
Decimal32(8)                        ok        REJECT            ok
Decimal64(8)                        ok        REJECT            ok
Int32                               ok            ok            ok
UInt32                              ok            ok            ok
String                              ok            ok            ok
LowCardinality(String)              ok            ok            ok
Date                                ok            ok            ok
DateTime64(3)                       ok            ok            ok
Nullable(Int32)                     ok            ok            ok
```

Only one rejection pattern in this matrix, on both Decimal widths (code 44,
`ILLEGAL_COLUMN`), captured verbatim and complete:

- **`bloom_filter` on `Decimal32(8)`**: `Unexpected type Decimal(9, 8) of
  bloom filter index: When validating secondary index \`ix\`.`
- **`bloom_filter` on `Decimal64(8)`**: `Unexpected type Decimal(18, 8) of
  bloom filter index: When validating secondary index \`ix\`.`

`minmax` and `set(100)` are unconditionally accepted on every type tested,
including both Decimal widths and both String-shaped types — this spike
found no rejections at all for those two index types across this type
list. `bloom_filter` on Decimal is the only reject in the index matrix;
`validate.py`'s skip-index rule should special-case Decimal types (by
their resolved `Decimal(P, S)` form, matching the codec matrix's finding
that ClickHouse reports decimals by resolved precision/scale, not DDL
spelling) as bloom-filter-incompatible while leaving every other tested
type combination unrestricted for all three index types.

**`query_log` fields — three confirmed present and non-zero, two
confirmed present as registered events but not emitted for this query.**
`query_duration_ms`, `read_rows`, `read_bytes`, and `memory_usage` all came
back populated and non-zero for both runs. Of the five requested
`ProfileEvents` keys:

- **`SelectedMarks`, `SelectedRanges`, `SelectedParts`** came back
  non-zero on both the cold and warm run (`13`, `1`, `1` respectively on
  both of the two runs shown), which by itself proves they
  are present under exactly those names in 26.8's `ProfileEvents` map for
  this query.
- **`QueryConditionCacheHits`, `QueryConditionCacheMisses`** came back `0`
  on both runs, which is ambiguous by itself (indexing a `Map` with `[]`
  returns the value type's zero for an absent key, so `0` does not by
  itself distinguish "present with value 0" from "absent"). The script now
  also selects `mapContains(ProfileEvents, 'QueryConditionCacheHits')` and
  `mapContains(ProfileEvents, 'QueryConditionCacheMisses')`, which
  returned **0 (false) on both runs** — so for *this* query, both keys are
  genuinely **absent** from the recorded `ProfileEvents` map, not merely
  zero-valued. Separately, `SELECT name FROM system.events WHERE name
  ILIKE '%QueryConditionCache%'` confirms both `QueryConditionCacheHits`
  and `QueryConditionCacheMisses` **are registered event names on this
  server** (alongside `SelectedParts`, `SelectedPartsTotal`,
  `SelectedRanges`, `SelectedMarks`, `SelectedMarksTotal`). Putting the two
  together: the event names are valid and exist on 26.8, but this spike's
  queries — which explicitly set `use_query_condition_cache: 0` — never
  populate them in `query_log`, consistent with the condition cache being
  disabled outright for the query rather than merely recording zero
  hits/misses of an active cache. A harness that wants to measure the
  condition cache's hit/miss behavior must run with
  `use_query_condition_cache: 1` (or unset) and re-check `mapContains` at
  that setting; this spike did not do so, since exercising the cache in
  that mode was outside the brief's scope.

**Cold vs warm timing at this size.** `query_duration_ms` from
`query_log` (the authoritative server-side figure, not the Python
`client_ms` wall-clock figure, which also includes HTTP/serialization
overhead) was **13 ms cold vs 11 ms warm** on the run reproduced above — a
real but small difference in absolute terms at this row count (100 000
rows, 13 granules, 1 part). The client-side wall-clock times (`client_ms`:
60.4 ms cold, 58.5 ms warm) show the same direction but a much smaller
relative gap, because they are dominated by fixed HTTP/connection overhead
that swamps the few milliseconds of actual server-side cache effect at
this data size. **`SelectedMarks`, `SelectedRanges`, and `SelectedParts`
were identical between cold and warm** (13, 1, 1 both times) — the cache
drop changes *how fast* the same plan executes, not *what* it reads — so
any real benchmark built on this harness needs a much larger dataset than
100 000 rows before cold/warm cache effects become a reliable,
well-separated signal; at this size the effect is measurable but small
enough that noise from other sources (HTTP overhead, OS page cache outside
ClickHouse's own caches) could plausibly dominate it. (The absolute
millisecond values shift slightly run-to-run relative to earlier drafts of
this section, since forcing Wide part format for the storage-stat fix
above changed the granule/mark layout slightly — 13 granules here vs 12 in
an earlier Compact-format run of the same query; this does not change any
conclusion, only the exact numbers quoted.)

**Cache-drop statements.** All three attempted statements succeeded on
this server version, each wrapped in its own try/except per correction 2
so a hypothetically-missing one could not abort the run:

- `SYSTEM DROP MARK CACHE`: **ok**
- `SYSTEM DROP UNCOMPRESSED CACHE`: **ok**
- `SYSTEM DROP QUERY CONDITION CACHE`: **ok**

All three exist and are callable as an unprivileged-enough user on
ClickHouse 26.8.2.7 with this account, so the design's cold-cache
measurement mode can rely on all three without needing a fallback/skip
path for this server version — though the per-statement try/except should
stay in the harness itself, since the brief's own premise (that
`QUERY CONDITION CACHE` might not exist) was a reasonable one to guard
against for portability to other ClickHouse versions even though it did
not fire here.

**`SYSTEM FLUSH LOGS` sufficiency.** A first attempt at this spike queried
`system.query_log` immediately after `SYSTEM FLUSH LOGS` and got **zero
rows back**, even though `client.command("SYSTEM FLUSH LOGS")` had
returned successfully — this was traced to the query_id call-shape defect
described below (the query actually executed under a server-autogenerated
query_id, not the one being searched for), not a genuine `FLUSH LOGS`
timing gap. Once the query_id was passed correctly, `SYSTEM FLUSH LOGS`
alone (no retry loop, no `time.sleep`) was sufficient for the row to appear
on the very next query in both the cold and warm iterations of every run
of this script. **Recommendation: `SYSTEM FLUSH LOGS` alone is sufficient
on 26.8.2.7; a retry loop is not needed** for this harness's query volume
— but this spike ran queries one at a time with no concurrent load, so a
busier harness run should still budget a short defensive retry rather than
assume this holds under contention.

**EXPLAIN indexes = 1 output format.** Returned one row per line as the
brief expected, with box-drawing characters (`│`, `└──`) forming the tree
structure — the same as `EXPLAIN` output would look in the CLI client. One
row was **blank** (a genuinely empty string between the `Output:` line and
`Aggregating`), which callers must tolerate rather than treat as an error
or skip silently — filtering it out naively (e.g. `if not line: continue`
during parsing) is safe, but code that assumes every row carries content
would break. The **surprise this spike hit directly**: those box-drawing
characters are non-ASCII (`U+2502`, `U+2514`, `U+2500`), and printing them
to a Windows console via the default `cp1252` stdout encoding raised
`UnicodeEncodeError` and crashed the script mid-run on the first attempt.
The committed script now calls `sys.stdout.reconfigure(encoding="utf-8")`
defensively at the top of `main()` (wrapped in try/except since not every
stdout stream supports `reconfigure`) — **the harness must do the same, or
otherwise force UTF-8 output, anywhere it prints or logs raw EXPLAIN text
on Windows**, or a later run's report generation could crash the same way.

**Two clickhouse-connect 1.8.0 API surprises, found and fixed while making
this spike run to completion (recorded here because the harness will hit
the same ones):**

1. **`Client.query()` takes no bare `query_id=` keyword.** The brief's
   literal call, `client.query(sql, query_id=qid, settings={...})`, raises
   `TypeError: Client.query() got an unexpected keyword argument 'query_id'`
   immediately — 1.8.0's signature has no such parameter at all.
2. **The seemingly-obvious fix, `transport_settings={"query_id": qid}`,
   silently does *not* work**, even though `"query_id"` is listed in
   `HttpClient.valid_transport_settings`: `transport_settings` values are
   sent as **HTTP headers**, and ClickHouse's HTTP interface only reads
   `query_id` from a **URL query parameter**, not a header, so the server
   silently ignored the header and autogenerated its own random query_id
   instead. This was confirmed directly: `client.query(..., transport_
   settings={"query_id": qid})` returns a `summary["query_id"]` different
   from the `qid` that was sent, and a subsequent `system.query_log` lookup
   by the sent `qid` returns zero rows even after `SYSTEM FLUSH LOGS` —
   this is what first made `FLUSH LOGS` sufficiency look suspect (see
   above) before the real cause was isolated.
3. **The call shape that actually works**: put `query_id` *inside* the
   `settings` dict — `client.query(sql, settings={"query_id": qid,
   "use_query_condition_cache": 0, "use_query_cache": 0})`. Because
   `"query_id"` is in `valid_transport_settings`, the client's
   `_validate_setting` passes it through unfiltered into the settings dict
   that becomes actual URL parameters, which is exactly where ClickHouse's
   HTTP interface expects it. Verified directly: the returned
   `summary["query_id"]` then matches the sent `qid`, and
   `system.query_log` returns the expected `QueryStart`/`QueryFinish` rows
   for it. **`validate.py`/the harness's query-tracking helper must use
   this call shape (`query_id` inside `settings`), not `query_id=` or
   `transport_settings={"query_id": ...}`,** on clickhouse-connect 1.8.0.

**Cleanup.** All probe tables (`spike_f64`, `spike_d64`, `spike_d32`,
`spike_d32_overflow`, `spike_probe`, plus the transient `spike_d64_stage` /
`spike_d32_stage` staging tables the fallback path would create if it were
exercised, and the extra `spike_probe2`/`spike_probe3` tables used only for
standalone verification of the untruncated error text and the
Compact-vs-Wide part-format behavior described above, outside `main()`)
were dropped by the script's own cleanup step or by hand after each probe.
`SELECT name FROM system.tables WHERE database = 'bench'` returned an
empty result after the final run — **no tables were deliberately left
behind in `bench`.**

**Addendum, 2026-09-16 (Plan 3, Task 3): nullable + primary key is
rejected at DDL time.** Executing the harness's rendered DDL against
26.8.2.7 for a `Nullable(Float64)` column with `primary_key: true` fails
with code 44 and the verbatim message `Sorting key contains nullable
columns, but merge tree setting `allow_nullable_key` is disabled`; a
`LowCardinality(Nullable(X))` column as sorting key fails the same way.
Nullable + projection *without* a primary key is accepted. The harness
records the pair as a validation skip carrying that message
(`validate.py`), rather than enabling `allow_nullable_key` in the DDL,
because turning the setting on is a table-behaviour decision the design
never made and it would change what the `nullable` axis measures.

**Addendum, 2026-09-16 (Plan 3, Task 7): a bare `count()` over a
primary-key range does not read the column.** The design's original
filter query, `SELECT count() FROM t WHERE v BETWEEN a AND b`, is answered
on 26.8 through the exact-count projection: `EXPLAIN indexes = 1` shows
`AggregatingProjection` over `ReadFromPreparedSource
(_exact_count_projection)`, which counts the fully-covered granules from
the primary index and reads only the boundary granules. Measured against a
200 000-row `Float64` table sorted by `v` with a 10 percent range filter
(20 000 matched rows): the plan selected `Granules: 4/25`, but
`SelectedMarks` was 2, `SelectedRanges` 2 and `read_rows` 16 385, so the
query timed index lookup rather than column decompression and flattered
every `primary_key: true` variant against the same codec without a key.
Appending the predicate to the full-scan aggregate instead
(`SELECT count(), sum(toFloat64(v)) FROM t WHERE ...`) forced every
matched granule to be read: `SelectedMarks` 4, `read_rows` 32 768. That is
what `queries.filter_sql` now renders (spec §5.3).

**Addendum, 2026-09-17 (Plan 3, Task 11): `LowCardinality` is accepted on
`String` only.** The applicability matrix above measured the
`LowCardinality` wrapper on `String` alone, and the harness inherited a
rule that the wrapper is applicable to any type. The first run of the int
suite found otherwise: all six `LowCardinality(Int32)` and
`LowCardinality(UInt32)` cells failed at `CREATE TABLE`. Re-measured the
same day by executing the harness's DDL for each type in the vocabulary:
`String` and `Nullable(String)` are accepted; `Int32`, `UInt32`,
`Float64`, `Date` (and `Bool`, `Int64`, `UInt8`, outside the vocabulary)
and their `Nullable` forms are refused with code 455, verbatim `Creating
columns of type LowCardinality(Int32) is prohibited by default due to
expected negative impact on performance. It can be enabled with the
`allow_suspicious_low_cardinality_types` setting.
(SUSPICIOUS_TYPE_FOR_LOW_CARDINALITY)`, naming the full column type
(`LowCardinality(Nullable(Int32))` when nullable); `Decimal32(8)`,
`Decimal64(8)` and `DateTime64(3)` and their `Nullable` forms are refused
with a different code, 43, `DataTypeLowCardinality is supported only for
numbers, strings, Date or DateTime, but got Decimal(18, 8).
(ILLEGAL_TYPE_OF_ARGUMENT)`, in the resolved `Decimal(P, S)` spelling and
inside `Nullable(...)` when nullable. The harness now records both as
validation skips carrying the message (ruling R27); the setting is not
enabled, on the same reasoning as the nullable sorting key above.

## 6. Conclusions for the generator design

Sections 1-5 measured what the design document had guessed. This section
states, per metric, whether the constructive first pass sufficed, the
tolerance to default to, which corrective move (if any) is required, and what
interaction a later plan has to handle. "Constructive" here means the one-shot
analytical construction *plus* any bisection that runs inside stage 1 on a
distribution parameter (the power-law exponent, the lognormal sigma) before
any value is laid out; it excludes the design's §4.4 refine pass, which
operates on an already-generated sequence.

**`entropy_bits` (section 1) — constructive with stage-1 bisection; ±0.01
bits; no refine pass.**

- *Constructive method sufficient?* Yes. Bisection on the power-law exponent
  `s` converged on every non-clamped (k, target, seed) combination, with a
  residual of order 1e-5 bits (k = 1 000, target 8.0 → achieved 8.000033;
  k = 10 000, target 2.0 → achieved 2.000051). 60 iterations over the
  `[0, 30]` bracket shrink the interval to ~2.6e-17, so the search is not the
  limiting factor — the residual is `largest_remainder`'s integer rounding of
  counts, which can only land on a discrete set of achievable entropies near
  the target.
- *Tolerance:* **±0.01 bits absolute**, three orders of magnitude looser than
  the measured residual, plus an explicit reachable-bound check (below) so an
  unreachable request is reported as a clamp rather than as a near miss.
- *Refine move:* none. The bisection is part of stage 1, and nothing about the
  sequence layout can change entropy, since stage 2 only permutes.
- *Interaction:* entropy and `cardinality_ratio` constrain each other, because
  every one of the k support values is required to appear at least once. The
  reachable floor is `H_floor = -p_heavy*log2(p_heavy) -
  (k-1)*(1/n)*log2(1/n)` with `p_heavy = (n-k+1)/n`, measured at **0.180 bits
  for k = 1 000** and **1.798 bits for k = 10 000** at n = 100 000, and it
  *rises* with k; the ceiling is `log2(k)` (9.966 and 13.288 respectively). A
  low entropy target and a high cardinality target are therefore mutually
  exclusive at fixed n, and Plan 2 must check both ends of
  `[H_floor, log2 k]` rather than the design's assumed `(0, log2 k]`. The use
  case 2 ambiguity this spike measured both sides of ("cardinality ratio 0.1
  (i.e. up to 1000 unique values)", which disagree at n = 100 000) **was
  resolved by the project owner on 2026-09-14 in favour of 1000 distinct
  values**. At the row count now set for that suite, 10 000 000, the
  `cardinality_ratio` is therefore 0.0001, and the entropy range is
  `[0.0025, 9.966]` bits, so its 4.0-bit target remains comfortably reachable.
  Spec §5.1 carries the corrected suite.

**`cardinality_ratio` (sections 1 and 4) — constructive; ±0.001 absolute
while clipping is negligible; no refine pass, but a clip guard.**

- *Constructive method sufficient?* Yes where the support draw is not clipped:
  achieved 0.7499-0.7500 against a 0.75 target at sigma = 0.25
  (clipped_frac 0.001%), and exactly k/n (0.0100, 0.1000) in the integer
  spike. No once clipping bites: 0.7409-0.7414 at sigma = 0.5
  (clipped_frac 1.2%) and 0.7016-0.7020 at sigma = 1.0 (6.45%). The mechanism
  is clip collapse, not rounding: at sigma = 0.5, 900 of the 909 missing
  support points are the 901 draws above `hi = 10` merging onto the single
  value `10.00000000`; ordinary 8 dp rounding collisions account for only ~9.
- *Tolerance:* **±0.001 absolute on the ratio**, valid only while
  `clipped_frac` stays below roughly 1%. Above that the miss is systematic and
  must be reported with its cause rather than tightened away.
- *Refine move:* none on the sequence — cardinality is fixed at draw time. The
  only levers are to request a larger `k` when a non-trivial `clipped_frac` is
  detected, or to accept the lower achieved ratio; either way the generator
  must flag `clipped_frac` above roughly 1% as "cardinality at risk".
- *Interaction:* with `index_of_dispersion`. A high ID target is reached by
  raising sigma, which raises `clipped_frac`, which erodes cardinality — the
  0.75 target degrades measurably at sigma = 0.5, long before ID approaches
  its pole, so a generator tuning purely for ID can silently blow the
  cardinality target. Also with `entropy_bits`, through the `H_floor(k)`
  relation above.

**`index_of_dispersion` (section 4) — constructive with stage-1 bisection, but
only against the measured multiset and only inside one branch; ±0.01
absolute; no refine pass, a solver requirement instead.**

- *Constructive method sufficient?* Yes, once two things are right. Bisection
  hit targets 0.5, 1.0, 2.0 and 5.0 to within **0.0001** across all three
  seeds. But it must run against `index_of_dispersion` measured on the final
  `np.repeat(support, counts)` array, not the analytic lognormal formula: the
  measured values (0.277 / 1.085 / 3.268 at sigma 0.25 / 0.5 / 1.0) are only
  80%, 72% and 36% of the analytic predictions (0.344 / 1.515 / 9.164), a gap
  that widens with sigma because clipping removes exactly the heavy right tail
  the variance depends on — and the analytic form is *qualitatively* wrong too,
  since its mean is pinned at `mean + shift` and it therefore has no pole.
- *Tolerance:* **±0.01 absolute** on the positive-mean branch (ID below
  roughly 10-15 for use case 1's mean/shift/clip, reached at sigma < ~1.3),
  two orders of magnitude looser than the measured 0.0001.
- *Refine move:* none on the sequence; the requirement lands on the solver.
  `index_of_dispersion(sigma)` is **not monotone** over the default
  `[0.01, 3.0]` bracket: the generated multiset mean falls with sigma and
  crosses zero at a seed-dependent **mean-crossing sigma of ~1.95-1.98**, so
  ID diverges to +infinity below the crossing (16.9 at sigma 1.75, 516.9 at
  1.95) and returns as a large negative number above it (-154.5 at 1.98,
  -2.15 at 3.0). Evaluated at `s_hi = 3.0` the bracket therefore yields a
  *negative* upper ID, and a naive `target_id >= id_hi` clamp test reports
  every positive target — even 0.5 — as unreachable without ever bisecting.
  The solver must first bisect on the *sign of the generated mean* to find the
  largest sigma below `s_hi` whose mean is still positive, use that as the
  effective upper bracket, and only then bisect on ID; it must treat
  `"invalid bracket"` (mean already non-positive at `s_lo`, i.e. no solution
  space at all) as a hard error distinct from a genuine clamp; and it must
  never report a target as unreachable without having checked that its bracket
  is valid. A "max reachable" figure computed near the crossing (measured
  6.3e5 to 2.6e6, seed-dependent) is an artifact of how far the narrowing
  bisection was run, not a physical ceiling — the true supremum on that branch
  is +infinity.
- *Interaction:* with `cardinality_ratio`, as above. Separately, the achieved
  multiset mean must **not** be predicted as "the shifted distribution mean
  minus a clip correction": the drift is dominated by the density-weighted
  count allocation, not by clipping. At sigma = 0.25 the clip and dedup
  effects are both 0.0000 and the **entire 0.0845 deficit** (raw draw mean
  2.9997 → multiset mean 2.9152) is density weighting; at sigma = 1.0 the
  1.5546 total drift splits near-evenly across clip 0.5031, dedup 0.5173 and
  weight 0.5342; only at sigma = 1.95 does clipping (2.0353) dominate. A
  clip-only model is wrong by about 3% at sigma 0.25 and about 70% at
  sigma 1.95.

**`runs_z` (sections 2 and 3) — constructive, exact; ±0.01 absolute; no refine
pass.**

- *Constructive method sufficient?* Yes, and this is the cleanest result of the
  five spikes. `layout_runs` hit the requested integer Wald-Wolfowitz run
  count R on **18 of 18 rows** (`R got == R target` without exception,
  including the clamped probe),
  so the only error is the single rounding of the continuous `mu + z*sd` to
  the nearest integer R. The largest |target − achieved| over the 15
  non-clamped rows was **0.00301** (z = -5.0, seed 0; sd = 137.283, per-run
  step `1/sd` = 0.00728), i.e. 41% of one step and inside the theoretical
  half-step bound of 0.00364. The layout is also confirmed to be a pure
  permutation of the multiset on every row (sorted-array equality and
  identical `entropy_bits`), which is what licenses the two-stage design.
- *Tolerance:* **±0.01 absolute**, roughly 3x the largest half-step
  (`0.5/sd`) seen at n = 100 000. The principled form of the bound is
  `0.5/sd` computed from the *actual* post-dichotomy `n1, n2`.
- *Refine move:* none. The design's "swap two elements across a block boundary
  to change R by ±1" move is not needed and is retired for this metric: the
  construction already hits R exactly, so a ±1 move could only overshoot.
- *Interaction:* three. (a) The median dichotomy's strict `x > median` puts
  every tied value in the below group, and with skewed integer data the ties
  are heavy — observed splits 31 852/68 148 (23 108 rows sitting exactly at
  the median), 37 461/62 539 and 48 849/51 151 for seeds 0/2/1 of the same
  weight profile. Every mu, sd, `r_max` and reachable-range calculation must
  therefore use the measured `n1, n2`, never an assumed 50/50. Only the
  median dichotomy was exercised: every runs and combined row in sections 2
  and 3 used `kind="median"`, so the **top-frequency dichotomy the design
  routes every string and categorical column through is untested** — the
  exact-R and pure-permutation results here are established for the numeric
  path only, and per (b) its reachable band differs substantially, since a
  most-frequent-set split is not near-balanced in general. (b) The
  reachable band is roughly **±300** at n = 100 000 for a near-balanced split
  (z = -316.22 at R = 2 up to +302.00 at `r_max = 97 699`), but it is
  **asymmetric and split-dependent**: the negative bound stays near -316.22
  for any split, while the positive ceiling `(r_max - mu)/sd` shrinks with
  `min(n1, n2)` — computed from the measured group sizes it is +302.00 at
  seed 1's 48 849/51 151, +189.42 at seed 2's 37 461/62 539, and only
  **+147.80 at seed 0's 31 852/68 148**, under half the ±300 figure. The -400
  probe clamped to R = 2 and landed at -316.22, so out-of-band targets must be
  reported as reachable-bound violations, against bounds computed from the
  measured `n1, n2` rather than against ±300. (c) When combined with
  `interarrival_cv`, `runs_z` is the metric that survives: achieved z was
  identical to 4 decimal places across all three CV targets, because
  `layout_combined` fixes the block pattern before consuming any CV-dependent
  randomness. The pair is not symmetric — z wins, CV pays.

**`interarrival_cv` (section 3) — constructive method insufficient; ±10%
relative after refinement, alone only; bisection refine pass required; degrades
whenever combined with `runs_z` at high cardinality.**

- *Constructive method sufficient?* No — the only metric of the five for which
  it is not. Both gap modes **always undershoot and never overshoot**, on all
  18 alone rows. Mean relative error: `free` 42.1% / 44.8% / 45.4% at targets
  0.5 / 1.0 / 2.0; `start` 5.9% / 20.6% / 33.6% for the same targets. Use
  `start` mode (draw the start-to-start gap, subtract the run length, clip
  negatives to 0): it is closer at every target, and 7x closer at 0.5. The
  residual miss comes from integer rounding compounded with that
  negative-clipping, which bites harder as the target rises because the gamma
  shape `1/cv^2` shrinks to 0.25 at cv = 2.0 and a large share of draws then
  fall below the run length and are clipped to 0.
- *Tolerance:* **±10% relative** to the target CV, and only for the alone
  case, and only *after* the refine pass. No absolute tolerance is safe: the
  raw miss is 0.03 absolute at cv = 0.5 but 0.67 absolute at cv = 2.0.
- *Refine move:* **bisection on the gamma's input CV parameter**, measured
  against `interarrival_cv` of the generated output — structurally the same
  search spike 1 runs on the power-law exponent. Justified by the monotonicity
  measured here: within each mode, higher input CV reliably produced higher
  output CV across 0.5 → 1.0 → 2.0. A simpler multiplicative rescaling
  correction is explicitly **rejected**: it would work only for `free`, whose
  miss is near-constant at 42-45%, and `start` — the mode we actually want —
  has a miss that grows from 5.9% to 33.6% across the same range, so no single
  factor fits it. The design's "relocate one run to lengthen or shorten a gap"
  move is not what the evidence supports; a parameter bisection is.
- *Interaction:* the constraint Plan 2 most needs to carry. Combining CV with
  a `runs_z` target splits repeat-runs across block boundaries and pulls the
  achieved CV toward the memoryless value 1 regardless of which side of 1 the
  target sat on: against `start`-alone means of 0.471 / 0.794 / 1.327, the
  combined means are **0.759 / 0.881 / 1.095** — target 0.5 pulled *up* by
  about 63% relative, target 2.0 pulled *down* by about 18%, measured at
  z = -0.25. Surviving *repeat-run* count — maximal stretches of one identical
  value, length ≥ 2, not the Wald-Wolfowitz run count R, which is 46 839 to
  49 961 at the two z targets tested — drops 42-46% (21 226-21 257 alone → 11
  499-12 272 combined). This is the **default case, not an edge case**: mean
  block length is `n/R` ≈ 2.0-2.1 rows at any near-random z, already well
  below the ~4.7-row mean inter-run gap, so nearly every repeat-run has a
  block boundary through its interior, and splitting a length-2 repeat-run
  destroys it entirely. Stated plainly, and not as a caveat: **at cardinality
  0.75 a suite that sets both `runs_z` and `interarrival_cv` cannot
  meaningfully hit its CV
  target**, at any z tested including the near-random default. The refine pass
  is not obviously a way out: the distortion is inflicted after the gaps are
  laid out, and survival of a run is largely decoupled from the arranged gap
  structure, so bisecting the gamma's input parameter against the *combined*
  output is not expected to recover the target — though that was not tested
  here, since this spike measures the miss rather than correcting it. Plan 2
  therefore faces a real design
  choice — whether to forbid the combination, to redesign the interarrival
  layout to respect the block boundaries, or to accept and document degraded CV
  fidelity whenever both are set. This spike measures the effect and takes no
  position on which of those is right; the choice belongs to whoever plans the
  generator. **That choice has since been made: the project owner confirmed the
  third option on 2026-09-14**, and the design records it in §2 and implements
  it in §4.3, which warns that the two targets trade off and records the
  achieved CV without applying the alone-case tolerance to it, and §4.4, which
  skips the CV refine pass entirely when both ordering targets are set. Plan 2
  should implement that behaviour rather than reopen the question. The
  direction is the opposite of intuition: *more negative* z
  means fewer, longer blocks and mildly *less* distortion (12 136-12 272
  surviving runs at z = -20 versus 11 499-11 630 at z = -0.25); large positive
  z shortens blocks further and would be expected to be worse, though that was
  not tested. Separately, the CV floor is set by discreteness: the mean
  start-to-start gap is fixed by the multiset at `n/m` = 4.711 and every gap
  is an integer ≥ 2, so alternating gaps of 4 and 5 already give CV ≈ 0.11 and
  targets near 0 are unreachable by construction. Finally, a note for whoever
  reads the alone-mode `runs_z` column: it sits at -78 to -80, not near 0,
  because forcing every repeated value into one contiguous block leaves only
  `m + S + 1` ≈ 75 000 independently placed units rather than n = 100 000
  rows, while `runs_expected`'s mu/sd assume the latter. A free permutation of
  the same multiset gives z = -0.34, as expected.

**Default tolerances, collected.** These are the values to put in
`DatasetSpec.tolerances`:

| metric | default tolerance | basis | measured achievement | caveat |
|---|---|---|---|---|
| `entropy_bits` | ±0.01 bits | recommended in section 1 | ~1e-5 bits | plus a `[H_floor, log2 k]` reachable-bound check |
| `cardinality_ratio` | ±0.001 absolute | **derived here**; no section recommends a number: 10x the 0.0001 miss measured where clipping is negligible | 0.0001 at clipped_frac 0.001% | only while `clipped_frac` < ~1%; flag above that |
| `index_of_dispersion` | ±0.01 absolute | recommended in section 4 | 0.0001 | positive-mean branch only (ID below ~10-15 for use case 1) |
| `runs_z` | ±0.01 absolute | **derived here**; section 2 recommends no number: ~3x the measured step from rounding to the integer run count (largest miss 0.00301, worst-case `0.5/sd` = 0.00364) | ≤0.00301 | the bound is `0.5/sd` from the measured `n1, n2` |
| `interarrival_cv` | ±10% relative | recommended in section 3 | 5.9-33.6% raw, before refinement | after the bisection refine; alone only — not when combined with `runs_z` |

**Decisions carried into the design document.**

1. §4.1 gains the tolerance table above (sections 1-4 recommendations).
2. §4.2 step 2 gains the measured entropy floor `H_floor` and its formula, in
   place of the assumed reachable range `(0, log2 k]` (section 1).
3. §4.2 step 3 gains the bracket requirement for the dispersion solve
   (positive-mean branch, detected per call, `"invalid bracket"` distinct from
   a clamp), the requirement to solve against the measured multiset rather
   than the analytic formula, the `mu_log = ln(mean) - sigma^2/2` reading of
   "mean", and the clip-erodes-cardinality guard (section 4).
4. §4.2 step 4 records that the density-weighted allocation itself pulls the
   multiset mean below the support mean, with no clipping required
   (section 4).
5. §4.3 step 1 records that the median dichotomy's tie handling can move the
   split far from 50/50, so all runs-test arithmetic uses the measured
   `n1, n2`, and the reachable z band — ±300 for a near-balanced split, but
   asymmetric: the negative bound holds near -316.22 whatever the split while
   the positive ceiling falls with `min(n1, n2)`, to +147.80 at the
   31 852/68 148 split (section 2).
6. §4.3 step 2 records `start` gap mode as the choice, the `n/m` mean-gap and
   integer-gap floor, and the combined-target degradation as a stated
   constraint on the generator (section 3).
7. §4.4 is rewritten: `runs_z` needs no refine pass, `interarrival_cv` is the
   only metric that does, and its move is a bisection on the gamma's input CV
   parameter rather than per-run relocation; §4.4 also records that this pass
   does not run when both ordering targets are set, the achieved CV being
   recorded instead (sections 2 and 3).
8. §5.2's validation-rules paragraph is replaced by the two measured
   applicability matrices with their rejection reasons, and the Decimal32
   overflow rule is corrected (section 5).
9. §5.2's storage-stat column list is corrected to the per-column names and
   gains the Wide-part-format requirement (section 5).
10. §5.2 and the `tables` row in §6.1 record the marks field as a byte size:
    `column_marks_bytes` / `marks_bytes`, in place of the design's original
    `marks`, which implied a mark count (section 5).
11. §5.2 records the direct Arrow decimal insert path as the primary one on
    pyarrow 24.0.0 / clickhouse-connect 1.8.0 / ClickHouse 26.8.2.7, with the
    Float64-staging + `toDecimal32/64` fallback kept only for other version
    combinations (section 5).
12. §2's decisions table and `CLAUDE.md` are aligned with the amended
    refinement rules: the bounded refine pass is for `interarrival_cv` alone,
    `runs_z` being exact by construction (sections 2 and 3).
13. §4.2 step 1's shortfall remedy is replaced. Clip collapse, not 8 dp
    rounding, is the dominant cause (900 of the 909 missing support points at
    sigma 0.5), and redrawing cannot recover it, so the remedy is to request a
    larger `k` on a non-trivial clipped fraction or accept the lower achieved
    ratio; the bounded redraw is kept for rounding collapse alone (section 4).
14. §5.4 gains three harness facts (section 5): on clickhouse-connect 1.8.0
    the `query_id` must be passed inside the settings mapping — a bare keyword
    raises and `transport_settings` fails silently under a server-autogenerated
    id — stdout and log encoding must be forced to UTF-8 wherever EXPLAIN text
    is printed, and cold versus warm server-side timing differs by only about
    2 ms at 100 000 rows, a caveat on §5.1's own example row count.
15. §5.2's Decimal32 rule is corrected a second time: the range check applies
    to the *rounded* values, since a draw in `[9.999999995, 10)` rounds to
    exactly 10 and overflows Decimal32(8) (section 5).

**Findings that invalidate a design-document assumption.** Kept to the
assumptions that actually changed:

1. **Storage-stat column names (design line 213).**
   `data_compressed_bytes`, `data_uncompressed_bytes` and `marks` from
   `system.parts_columns` are **part-level totals repeated identically on
   every column's row**, not a per-column breakdown — proved by `spike_f64.v`
   and `spike_f64.row_id` reporting the identical 1 206 123 / 1 600 000 bytes,
   which is the sum of both columns. The per-column names are
   `column_data_compressed_bytes`, `column_data_uncompressed_bytes` and
   `column_marks_bytes`; the last is a byte size, not the count the design's
   `marks` label implies. (`rows` was already per-column and is fine.)
2. **Per-column storage stats are not always available.** They are populated
   only for **Wide** parts, and MergeTree stores a part as **Compact** below
   `min_bytes_for_wide_part` / `min_rows_for_wide_part` — including the
   ~1.6 MB, 100 000-row tables of the design's own examples, which returned
   **0, silently, not an error**, for all three per-column figures. Forcing
   Wide (`min_bytes_for_wide_part=0, min_rows_for_wide_part=0`) is a
   requirement of the harness's table creation, not a footnote.
3. **Codec applicability was guessed wrong in both directions.** The design
   said "Gorilla: floats and ints only; T64: ints/dates only". Measured:
   Gorilla is accepted on **Float64 only** and rejected on every integer,
   Decimal, Date, DateTime64 and Nullable(Int32) tested ("the combination ...
   is meaningless"); T64 is accepted on **7 of the 10 types** — both Decimal
   widths, Int32, UInt32, Date, DateTime64(3) and Nullable(Int32) — and
   rejected only on Float64 and the two String-shaped types. So the design was
   simultaneously too permissive about Gorilla and too restrictive about T64.
4. **Decimal32 is not an overflow case for use case 1.** The design's rule
   "value range must fit the target type given scale" was written assuming
   `[-1, 10]` at 8 dp overflows Decimal32. It does not: Decimal32(8)
   represents -9.99999999 to +9.99999999, `rng.uniform(-1, 10)` draws strictly
   below 10, and the full 100 000-row dataset inserted cleanly in 91 ms. Only
   the exact value 10 overflows — and it is rejected at the insert boundary by
   both the Arrow cast and plain `INSERT VALUES`, never stored truncated.
   Decimal32(8) is also the *cheaper* of the two widths for this data under
   matching codecs (401 912 versus 617 602 compressed bytes).
5. **Entropy's reachable range is not `(0, log2 k]`.** The floor is strictly
   positive and grows with k — 0.180 bits at k = 1 000, 1.798 bits at
   k = 10 000 for n = 100 000 — because every support value is guaranteed at
   least one occurrence.
6. **A bounded refinement pass is not needed for both ordering metrics.** The
   design describes one for `runs_z` and `interarrival_cv` alike. `runs_z`
   needs none (the construction hits R exactly, 18/18 rows).
   `interarrival_cv` needs one, but not the design's per-run relocation move —
   a bisection on the input CV parameter.
7. **The dispersion solve is not a plain bisection over a fixed bracket.** The
   design says ID is "solved by bisection on the distribution's spread
   parameter". Over the default `[0.01, 3.0]` bracket ID is not monotone at
   all: it has a sign-flipping pole where the generated mean crosses zero
   (sigma ~1.95-1.98), and the literal bisection reported every target,
   including 0.5, as unreachable.
8. **The mean drift is not a clipping effect.** A reader of the design would
   predict the achieved mean from the distribution mean, the shift and a clip
   correction; the dominant term at low-to-moderate sigma is instead the
   density-weighted count allocation, which at sigma = 0.25 accounts for the
   entire 0.0845 deficit while clipping contributes 0.0000.

Two design assumptions the spikes **confirmed**, recorded because later work
leans on them: stage 2 is a pure permutation of the multiset (verified on all
54 layout rows across sections 2 and 3, by sorted-array equality and identical
entropy), and all three `SYSTEM DROP ... CACHE` statements the cold-cache mode
needs exist and succeed on ClickHouse 26.8.2.7.

**Addendum, 2026-09-15: the rank-to-support assignment under an entropy
target is shuffled.** The design's §4.2 step 2 reads literally as
`f_i ∝ rank_i^(-s)` over the k support values, which, with the support held
sorted, would always give the smallest value the heaviest count and the
largest value a count of one. The Plan 2 generator (`multiset.build_multiset`)
instead permutes the power-law counts with the spec's seed before pairing
them with the sorted support, so the value-to-count pairing is random and
differs from the design's literal text. This matches the construction this
spike's `expand` used, which drew the support with `rng.choice` in random
order and paired it with the counts in rank order, so the entropy and
cardinality results in section 1 were measured under a shuffled pairing all
along. Entropy and cardinality are order-free, so neither changes; what does
change is any metric that reads the values, which is why `validate()` rejects
`entropy_bits` together with `index_of_dispersion` (the pairing reorders with
sigma while the count permutation stays fixed, so the dispersion objective is
not monotone in sigma).

## 7. Top-frequency dichotomy (spike_topfreq.py)

Section 2 measured `layout_runs` only against the median dichotomy on skewed
numeric data (32/68 at its most lopsided). Spec §2/§4.3 route every string
and categorical column through the *top-frequency* dichotomy instead —
membership of the most-frequent values covering half the rows — which this
spike exercises directly for the first time outside its one ten-element unit
test.

Full output (n = 200 000, `categorical_multiset` with k in {50, 1000}, power-law
exponent s in {0.0, 0.8, 1.2, 2.0}, targets in {-0.25, -20.0, 20.0}, seed 0 for
the multiset / seed 1 for the layout):

```
    s     k       n1       n2  split%      z lo      z hi  target   R want    R got  achieved  multiset
  0.0    50   100000   100000    50.0    -447.2     447.2   -0.25    99945    99945    -0.250      True
  0.0    50   100000   100000    50.0    -447.2     447.2  -20.00    95529    95529   -19.999      True
  0.0    50   100000   100000    50.0    -447.2     447.2   20.00   104473   104473    19.999      True
  0.0  1000   100000   100000    50.0    -447.2     447.2   -0.25    99945    99945    -0.250      True
  0.0  1000   100000   100000    50.0    -447.2     447.2  -20.00    95529    95529   -19.999      True
  0.0  1000   100000   100000    50.0    -447.2     447.2   20.00   104473   104473    19.999      True
  0.8    50   104514    95486    52.3    -447.2     408.6   -0.25    99741    99741    -0.252      True
  0.8    50   104514    95486    52.3    -447.2     408.6  -20.00    95334    95334   -20.001      True
  0.8    50   104514    95486    52.3    -447.2     408.6   20.00   104260   104260    19.999      True
  0.8  1000   100311    99689    50.2    -447.2     444.4   -0.25    99944    99944    -0.251      True
  0.8  1000   100311    99689    50.2    -447.2     444.4  -20.00    95528    95528   -20.000      True
  0.8  1000   100311    99689    50.2    -447.2     444.4   20.00   104472   104472    19.999      True
  1.2    50   102881    97119    51.4    -447.2     422.2   -0.25    99862    99862    -0.251      True
  1.2    50   102881    97119    51.4    -447.2     422.2  -20.00    95450    95450   -19.998      True
  1.2    50   102881    97119    51.4    -447.2     422.2   20.00   104386   104386    19.998      True
  1.2  1000   103301    96699    51.7    -447.2     418.6   -0.25    99836    99836    -0.251      True
  1.2  1000   103301    96699    51.7    -447.2     418.6  -20.00    95425    95425   -19.999      True
  1.2  1000   103301    96699    51.7    -447.2     418.6   20.00   104359   104359    19.999      True
  2.0    50   123037    76963    61.5    -447.2     279.7   -0.25    94641    94641    -0.250      True
  2.0    50   123037    76963    61.5    -447.2     279.7  -20.00    90459    90459   -20.001      True
  2.0    50   123037    76963    61.5    -447.2     279.7   20.00    98929    98929    20.001      True
  2.0  1000   121052    78948    60.5    -447.2     291.7   -0.25    95516    95516    -0.249      True
  2.0  1000   121052    78948    60.5    -447.2     291.7  -20.00    91295    91295   -20.001      True
  2.0  1000   121052    78948    60.5    -447.2     291.7   20.00    99843    99843    20.000      True

--- ceiling as the split gets lopsided (n = 200 000) ---
 top set %       n1       n2         sd       z lo       z hi
        50   100000   100000     223.61     -447.2      447.2
        60   120000    80000     214.66     -447.2      298.1
        75   150000    50000     167.70     -447.2      149.1
        90   180000    20000      80.50     -447.2       49.7
        99   198000     2000       8.85     -447.2        4.5
```

Ran identically from the repository root and from inside `spikes/`
(`python spike_topfreq.py`), confirming the `sys.path` fix resolves
`chbench` and `spikes.spike_runs` correctly from both working directories —
both runs produced byte-identical tables, as expected from fixed seeds.

**Split as a function of frequency skew.** `topfreq`'s own definition —
accumulate ranks by descending count until the cumulative total reaches
`n/2`, then take that set — keeps the split much closer to 50/50 than the
median dichotomy's tie-driven imbalance (32/68 at its worst, section 2) ever
did, across every exponent this spike tested: s = 0.0 gives an exact 50.0%
(uniform weights, no rank dominates enough to overshoot the `n/2` cutoff);
s = 0.8 nudges to 52.3% (k=50) / 50.2% (k=1000); s = 1.2 is similar, 51.4% /
51.7%. The split is still within 2 points of even at s = 1.2. It only becomes
clearly lopsided at the highest exponent tested, s = 2.0, reaching **61.5%
(k=50) / 60.5% (k=1000)** — the point at which it stops being near-balanced
for this construction. Even that is far short of the "far more than half"
scenario the task brief worried about: because `topfreq` targets covering
exactly half the rows by construction, only the single rank that straddles
the `n/2` cutoff can push the split past even, and its overshoot is bounded
by that one rank's count. Reaching a genuinely lopsided top-frequency split
(80-99% coverage) needs either a much higher exponent or, more realistically,
a small number of ranks each individually holding a large share of `n` (e.g.
one dominant status code) — which is why the script's second experiment
constructs the lopsided splits directly (by fixing `n1` as a fraction of `n`)
rather than trying to reach them through `categorical_multiset`'s power law.

**Reachable band at each split, and the 90% ceiling.** At the balanced 50/50
split (sd = 223.61) the band is symmetric, **z = -447.2 to +447.2**. The
negative bound holds at essentially the same **-447.2** at every split tested,
from 50/50 down to the most lopsided 99/1 (n1 = 198 000, n2 = 2 000, sd =
8.85, mu = 3 961, (2-3961)/8.85 = -447.29) — reproducing, for the
top-frequency dichotomy, the same near-split-invariance of the floor that
section 2 found for the median dichotomy, because `R = 2` is always reachable
regardless of which dichotomy produced `n1, n2`. The positive ceiling, by
contrast, collapses as the split moves away from even: **+298.1 at 60%,
+149.1 at 75%, and explicitly, at a 90% top set, +49.7** — a 9.0x reduction
from the balanced case's +447.2 (447.2 / 49.7 = 9.00) — falling further to
**+4.5 at 99%** (a 99.4x reduction). The 24-row sweep's own lopsided rows
corroborate this from the categorical-multiset side: the 60.5-61.5% splits at
s = 2.0 give ceilings of +279.7/+291.7, consistent with the directly
engineered 60% row's +298.1.

**Target run count and multiset preservation.** `R got` equals `R want`
exactly on all 24 rows, with no exceptions, including the most lopsided
tested split (61.5%/38.5%, s = 2.0, k = 50). `multiset` (sorted-array
equality, identical `entropy_bits`, and identical `cardinality_ratio`) is
`True` on all 24 rows. `layout_runs` never inspects which dichotomy produced
its boolean mask — it only permutes and reslices `x[b]` and `x[~b]` — so this
confirms it is dichotomy-agnostic, exactly as the brief expected and as
section 2 already established for the median dichotomy.

**Which targets clamped.** None of the three tested targets (-0.25, -20.0,
+20.0) were clamped on any of the 24 rows: the achieved z landed within
0.003 of the requested target in every case (e.g. 19.999, -19.998, 20.001 —
the same ≤0.00301 rounding-step precision section 2 measured for the median
dichotomy), because `categorical_multiset` at the tested exponents never
produced a split lopsided enough to push +20.0 near a ceiling — even the most
lopsided row's ceiling (+279.7 at 61.5%/38.5%) left more than an order of
magnitude of headroom above the +20.0 target. But the ceiling sweep shows
clamping is a real risk once real data is more concentrated than these power
laws produced: a target of +20.0 already exceeds the +4.5 ceiling at a 99%
top set, and a suite requesting a larger positive `runs_z` (the kind of
target section 2 tested up to +50/+400) would clamp well before reaching 99%
— at a 90% top set the ceiling is only +49.7, and at 75% it is +149.1. **A
suite that sets `runs_z` on a string column cannot assume the reachable band
measured for a numeric column with the median dichotomy; it must compute
`runs_z_bounds` from the actual post-`dichotomise(kind="topfreq")` `n1, n2`
of that specific dataset**, because a genuinely skewed categorical (a
dominant status code, a Zipfian ID column) can produce a top-set coverage far
beyond the 61.5% ceiling this spike's power-law construction reached.

**Recommendation for Task 8.** `sequence.py`'s runs-layout step needs **no
categorical-specific code beyond passing `kind` through** to `dichotomise`,
`target_run_count`, and `runs_z_bounds` — `layout_runs` is confirmed
dichotomy-agnostic (exact `R got == R want` and full multiset preservation on
all 24 rows), matching section 2's finding for the median dichotomy. The
generator **should warn** whenever the requested `runs_z` falls outside
`runs_z_bounds(n1, n2)` computed from the *measured* post-dichotomise split
of the specific column — not from an assumed 50/50 or from the modest skew
this spike's exponents produced — because the positive ceiling for `topfreq`
collapses sharply and asymmetrically as the top set grows past even (9.0x
down at 90%, 99.4x down at 99%), and a categorical column's real-world skew
is not bounded by the power-law exponents tested here. No runs-z
clamp-reporting path exists yet to model this on: `chbench/datagen/`
currently has only `metrics.py` and `solve.py`, and while `solve.py`
already defines the status vocabulary (`SolveStatus`: ok / clamped-low /
clamped-high / invalid-bracket), its docstring names only the entropy
solve, the dispersion solve, and the CV refinement as callers — not the
runs z-score, which the design's own conclusion says needs no bisection
because it hits the target run count exactly (section 2, confirmed again
here for `topfreq`). The only clamp anywhere today is the bare, unreported
`np.clip` inside `target_run_count` in the phase one spike. The sequence
module would therefore be adding the *first* runs-z clamp-reporting path,
reusing `SolveStatus` for the report but driving it from `runs_z_bounds`
against the measured post-dichotomise group sizes rather than from a
bisection search.

## 8. Vectorising the inter-arrival layout (spike_vectorise_interarrival.py)

**Question.** Task 13's profile of `generate()` with `interarrival_cv=1.0` at
10 000 000 rows and cardinality 0.75 put 96.0 percent of runtime (814.6 of
849.0 profiled seconds) in `sequence.layout_interarrival`, whose Python loop
does one `np.repeat` and two list appends per repeat-run, 2.5 million times
per layout, 55 layouts per generate. The deferred-findings note named this as
the first place a Rust extension would earn its keep. Can numpy alone remove
the loop, and does it produce the same array?

**Method.** Build the same multiset the package builds for that spec
(`build_multiset`, seed 42), then lay it out twice from identical generator
state (`default_rng(7)`): once with the package function, once with a copy
whose draws are verbatim up to `free_int` and whose loop is replaced by run
start offsets from two cumulative sums, one `np.repeat` over the run values,
a scatter into a preallocated array, and the singles into the remaining
positions. Time both, check `np.array_equal`.

Full output (2026-09-15, i5-1235U, Python 3.12.13, numpy 2.3.5):

```
multiset at 10,000,000 rows: 1.14s; support 7,500,000, repeat-runs 2,500,000, singles 5,000,000
package layout_interarrival (Python loop): 10.89s
vectorised layout (numpy only):            0.94s
np.array_equal(package, vectorised): True; speedup 11.5x
```

**Conclusion.** The loop is removable in numpy with no change to the draws
and a byte-identical result, eleven times faster per layout. The hot spot is
Python overhead, not a numeric kernel numpy lacks, so it does not justify a
Rust extension; the recommendation in the deferred-findings note is to lift
this function into `chbench/datagen/sequence.py` in Plan 3 and lower the
refine cap, and to revisit Rust only if a profile after that still shows a
Python-bound function above about a third of runtime.


## 9. The recommended-type vocabulary (spike_ch_vocabulary.py)

**Question.** The sibling project turbo-parakeet recommends, per column, a
ClickHouse type and codec chain, and writes chbench suites over the real column.
Its types include ones chbench never measured (narrow integers, `Float32`,
`Bool`, `Decimal128`, `Date32`, `DateTime`, `DateTime64` with other precisions
and time zones, `FixedString`, `Enum8` / `Enum16`, `Time64`), and the `GCD`
codec. Which of them does the server accept, in which combinations, and what
does `GCD` do on disk?

**Method.** Server ClickHouse **26.8.31.2** (the container's current build; §5
ran on 26.8.2.7), `clickhouse-connect` 1.8.0. For each type below, in chbench's
DDL form (`v <type> CODEC(...)`, Wide parts forced, `LowCardinality` as the type
wrapper, `ORDER BY tuple()`), a `CREATE TABLE` and then an `INSERT ... SELECT`
of 1 000 representative values (every seventh a NULL when nullable), recording
the stage that rejected (every rejection below is at `create`; no accepted DDL
failed its insert). Two matrices, factored like §5.2's, each over nullable
false / true: codec x compression (`LZ4`, `ZSTD(3)`) with no skip index, and
skip index (`minmax`, `set(100)`, `bloom_filter`) with codec `none`,
compression `LZ4`. 1 140 cells; every cell is in
`spikes/spike_ch_vocabulary_results.tsv`, and rerunning
`python spikes/spike_ch_vocabulary.py` regenerates it. A second probe (inline,
not committed) inserted Arrow columns through `insert_arrow` into each type and
read them back, see "Arrow insert" below.

**Types accepted by the DDL and the insert** (all of them, with the
combinations below): `UInt8`, `UInt16`, `UInt64`, `Int8`, `Int16`, `Int64`,
`Float32`, `Bool`, `Decimal128(4)`, `Date32`, `DateTime`, `DateTime('UTC')`,
`DateTime64(0)`, `DateTime64(6)`, `DateTime64(9, 'UTC')`, `FixedString(3 | 4 |
16)`, `Enum8('a' = 1, 'b' = 2)`, `Enum16(...)` with 200 values, `Time64(3)`.
Nothing was rejected at the type level. **`Time64(3)` is accepted with the
default settings and with `enable_time_time64_type = 1` alike: the two runs
give identical cells**, so no setting is needed on 26.8.31.2.

**Codec matrix** (`CODEC(<codec>, LZ4)`; identical with `ZSTD(3)` except for
the compression name inside the Gorilla message; identical nullable and not
except where noted). `ok` = accepted by DDL and insert; the number is the
rejection's error code.

```
type                         GCD   Delta DoubleDelta Gorilla  T64   FPC   LowCardinality
Int8/16/32/64, UInt8/16/32/64 ok     ok      ok       36      ok    36    455
Bool                          ok     ok      ok       36      ok    36    455
Float32, Float64              36     ok      ok       ok      431   ok    455
Decimal32(8), Decimal64(8)    ok     ok      ok       36      ok    36    43
Decimal128(4)                 ok     36      36       36      431   36    43
Date, Date32, DateTime        ok     ok      ok       36      ok    36    455
DateTime64(p[, tz])           ok     ok      ok       36      ok    36    43
Enum8, Enum16                 36     ok      ok       36      ok    36    43
Time64(3)                     36     ok      ok       36      ok    36    43
FixedString(4)                36     ok      36       36      431   36    ok
FixedString(3), (16)          36     36      36       36      431   36    ok
String                        36     36      36       36      431   36    ok
```

`FixedString(n)` was measured at n = 3, 4 and 16 only. Delta / Gorilla accept
the sizes 1, 2, 4 and 8 bytes, which the server's own message states; n = 1, 2,
8 follow from it but were not run.

Exact messages (code, symbolic name, text; `<T>` is the type as the server
prints it: `Decimal(P, S)` for decimals, otherwise the DDL spelling, an
`Enum8('a' = 1, ...)` listing in full; never wrapped in `Nullable` except in
the LowCardinality and bloom filter messages):

- `GCD` outside Int*, UInt*, Decimal*, Date*, DateTime* (code 36,
  `BAD_ARGUMENTS`): `Codec GCD cannot be applied to column <T> because it can
  only be used with Int*, UInt*, Decimal*, Date* or DateTime* types.` Seen on
  Float32, Float64, String, FixedString(n), Enum8 / Enum16, Time64. **`Bool`,
  `Date`, `Date32` and `DateTime64` are inside the set**: accepted.
- `Delta` / `Gorilla` on `Decimal128` and on `FixedString(n)`, n not in
  {1, 2, 4, 8} (code 36): `Codec Delta is only applicable for data types of
  size 1, 2, 4, 8 bytes. Given type <T>.` (and `Codec Gorilla is ...`). The
  String messages of §5 are unchanged.
- `DoubleDelta` on `Decimal128` (code 36): `Codec DoubleDelta is only
  applicable for data types of size 1, 2, 4, 8 bytes. Given type Decimal(38,
  4).` On `FixedString(n)`, any n: `Codec DoubleDelta is not applicable for
  FixedString(n) because the data type is not numeric.` (the String clause).
- `Gorilla` on a fixed-size non-float of 1, 2, 4 or 8 bytes (code 36): `The
  combination of compression codecs Gorilla(N), <compression> is meaningless,
  because it does not make sense to apply a floating-point time series codec
  to non-floating-point columns (Note: you can enable setting
  'allow_suspicious_codecs' to skip this check).` N is the storage width: 1
  for UInt8 / Int8 / Bool / Enum8, 2 for 16-bit ints, Enum16 and Date, 4 for
  Date32, DateTime (with or without time zone) and FixedString(4), 8 for
  Int64 / UInt64, Time64 and every DateTime64. **The `(Note: ...)` suffix is
  new on 26.8.31.2**; §5's message ended after "columns". Gorilla is accepted
  on `Float32` as on `Float64`.
- `T64` (code 431, `ILLEGAL_SYNTAX_FOR_CODEC_TYPE`): `T64 codec is not
  supported for specified type <T>.` on Float32 / Float64, Decimal128,
  FixedString(n), String. Accepted on every integer, Bool, Date32, DateTime,
  DateTime64, Enum8 / Enum16, Time64 (and Decimal32 / Decimal64 as in §5).
- `FPC` (code 36): `Codec 'FPC' is not applicable for <T> because the data
  type is not Float*.` Accepted on Float32.
- `LowCardinality` wrapper: **accepted on `String` and `FixedString(n)`
  only**. Numbers, Bool, Date, Date32, DateTime (code 455,
  `SUSPICIOUS_TYPE_FOR_LOW_CARDINALITY`): `Creating columns of type
  LowCardinality(<T>) is prohibited by default due to expected negative
  impact on performance. It can be enabled with the
  `allow_suspicious_low_cardinality_types` setting.` Decimals (any width),
  DateTime64, Enum8 / Enum16, Time64 (code 43, `ILLEGAL_TYPE_OF_ARGUMENT`):
  `DataTypeLowCardinality is supported only for numbers, strings, Date or
  DateTime, but got <T>.` `<T>` carries the `Nullable(...)` wrapper when the
  column is nullable.

**Skip-index matrix** (`INDEX ix v TYPE <index> GRANULARITY 1`, codec `none`,
`LZ4`): `minmax` and `set(100)` are accepted on every type above, nullable or
not. `bloom_filter` is accepted on every type except **`Decimal32`,
`Decimal64`, `Decimal128`, `Date32` and `Time64`** (code 44, `ILLEGAL_COLUMN`):
`Unexpected type <T> of bloom filter index: When validating secondary index
`ix`.` with `<T>` = `Decimal(P, S)`, `Date32`, `Time64(3)`, and
`Nullable(<T>)` when the column is nullable (§5's Decimal message omitted the
wrapper; its nullable form was not measured). `Date`, `DateTime`,
`DateTime64`, `Bool`, `Float32`, `FixedString`, `Enum8` and `Enum16` accept it.

**Arrow insert** (`insert_arrow`, a one-column table of the type, with a
null). Every type round-trips, the value read back through `toString`: `uint8`
/ `int8` / `float` / `bool`; `decimal128(38, 4)`; `date32`; `timestamp[s]` into
`DateTime` and `DateTime64(0)`, `timestamp[ms | us | ns]` into `DateTime64(3 |
6 | 9)`; a time zone on the Arrow timestamp (`tz=UTC`, `tz=Europe/Paris`) into
`DateTime('tz')` / `DateTime64(p, 'tz')` keeps the instant (23:15:23 Paris for
22:15:23 UTC); `fixed_size_binary[4]` and `string` into `FixedString(4)` (a
shorter string is zero-padded; a longer one raises code 131,
`TOO_LARGE_STRING_SIZE`); `string` into `Enum8` (a name outside the enum
raises code 691); `time32[s | ms]` and `time64[us | ns]` into `Time64(0 | 3 |
6 | 9)` (negative times and hours past 24 are stored). `Date32` holds -719162
(0001-01-01).

**The GCD codec on disk.** `Int64`, 1 000 000 rows, one part after `OPTIMIZE
FINAL`; `column_data_compressed_bytes` / `column_data_uncompressed_bytes` of
column `v`. "x" is `intHash64(number) % 10^9` (random, < 10^9); "x*1000" the
same values times 1 000 (< 10^12); "sorted" is `number` and `number * 1000`.

```
input                      codec          compressed  uncompressed
x*1000 (random)            LZ4               6968860       8000000
x*1000 (random)            GCD, LZ4          6054590       8000000
x*1000 (random)            ZSTD(3)           5812455       8000000
x*1000 (random)            GCD, ZSTD(3)      4284001       8000000
x      (random)            LZ4               5941270       8000000
x      (random)            GCD, LZ4          6054691       8000000
x      (random)            ZSTD(3)           4280194       8000000
x      (random)            GCD, ZSTD(3)      4284056       8000000
x*1000 sorted              LZ4               5019262       8000000
x*1000 sorted              GCD, LZ4          4008690       8000000
x*1000 sorted              ZSTD(3)           1872291       8000000
x*1000 sorted              GCD, ZSTD(3)      1033374       8000000
x      sorted              LZ4               4004915       8000000
x      sorted              GCD, LZ4          4008570       8000000
x      sorted              ZSTD(3)           1029794       8000000
x      sorted              GCD, ZSTD(3)      1033787       8000000
```

Observations, not interpretations. The uncompressed size is 8 000 000 in
every row: the column stays 8 bytes wide, so GCD does not narrow the type. On
multiples of 1 000, `GCD, LZ4` is 13.1 percent smaller than `LZ4` (random)
and 20.1 percent smaller (sorted); `GCD, ZSTD(3)` is 26.3 and 44.8 percent
smaller. `GCD, <c>` on the multiples lands within 1.9 percent (LZ4, random),
0.1 percent (LZ4, sorted), 0.1 percent (ZSTD, random) and 0.4 percent (ZSTD,
sorted) of plain `<c>` on the divided values, i.e. it reproduces the size of
the column of quotients. On data whose GCD is 1 the codec costs a little:
`GCD, LZ4` is 1.9 percent larger than `LZ4` on random x (6054691 against
5941270) and 0.09 percent larger on sorted x; `GCD, ZSTD(3)` is 0.09 percent
larger (random) and 0.4 percent larger (sorted). That is consistent with the
model "same width plus a small per-block header"; the random-x LZ4 gap
(113 421 bytes) is larger than a header alone would suggest and was not
investigated.

**Consequences for chbench** (ruling R36, spec §5.2): the vocabulary gains
`UInt8`, `UInt16`, `UInt64`, `Int8`, `Int16`, `Int64`, `Float32`, `Bool`,
`Decimal128`, `Date32`, `DateTime`, `FixedString`, `Enum8`, `Enum16`, `Time64`
and the codec `GCD`; the rules above become skip rules carrying the server's
messages; the parameterised types take their parameters from the dataset.

## 10. Skip-index sizes (spike_ch_bloom_size.py)

ClickHouse 26.8.31.2. `Int64` column `v` (`CODEC(LZ4)`), 1 000 000 rows,
`ORDER BY tuple()`, Wide parts, `index_granularity = 8192`, one part after
`OPTIMIZE ... FINAL`, `INDEX ix v TYPE <t> GRANULARITY G`; sizes from
`system.data_skipping_indices` (`data_compressed_bytes`,
`data_uncompressed_bytes`, `marks_bytes`). Index blocks: ceil(1e6 / (8192 G))
= 123 (G = 1) or 31 (G = 4), the last one partial. Shapes: `distinct` (every
value distinct), `clustered` (100 distinct values per granule, each value in
one granule), `ten` (10 distinct values, pseudo-random order). D = distinct
values per index block, measured from the stored data (8 192 / 32 768,
100 / 400, 10; the last block fewer). Raw cells:
`spikes/spike_ch_bloom_size_results.tsv`.

**bloom_filter, uncompressed bits per index-block distinct value**
(uncompressed bytes x 8 / sum of D over blocks):

| shape (G) | fpr 0.001 | 0.01 | 0.025 | 0.05 |
|---|---|---|---|---|
| distinct (1) | 15.00 | 10.00 | 8.00 | 7.00 |
| distinct (4) | 15.00 | 10.00 | 8.00 | 7.00 |
| clustered (1) | 15.12 | 10.08 | 8.08 | 7.12 |
| clustered (4) | 15.04 | 10.04 | 8.04 | 7.04 |
| ten (1) | 16.00 | 11.20 | 8.80 | 8.00 |
| ten (4) | 16.00 | 11.20 | 8.80 | 8.00 |

Per row (uncompressed bits / 1e6
rows): `distinct` 15.0 / 10.0 / 8.0 / 7.0 at both G; `clustered` 0.186 /
0.124 / 0.099 / 0.088 (G = 1); `ten` 0.0197 / 0.0138 / 0.0108 / 0.0098
(G = 1) and 0.0050 / 0.0035 / 0.0027 / 0.0025 (G = 4). Bits per row differ by
two orders of magnitude across shapes at a fixed fpr; bits per distinct value
are constant: the filter is sized by the distinct values of the block, not
by its rows.

**Exact model.** Per block, uncompressed bytes = ceil(D x b / 8) + v(D), with
b = 15, 10, 8, 7 for fpr 0.001, 0.01, 0.025, 0.05 and v(D) = 1 byte for
D < 128, 2 for D < 16 384, 3 above. This reproduced all 24 bloom cells
(3 shapes x 2 G x 4 fpr) to the byte, including the partial last block
(`distinct`: 576 rows, `clustered`: 300 distinct at G = 4).

**Compression** (compressed / uncompressed): `distinct` 1.004 to 1.0044
(compression adds 0.4 percent: random bits); `clustered` 1.005 to 1.007; `ten` 0.026 to 0.040 (G = 1; 65 compressed bytes for 2 460) and 0.09 to
0.15 (G = 4). In `ten` every block holds the same 10 values, so all blocks are
byte-identical and LZ4 collapses them; the ratio belongs to that shape, not to
the filter.

**minmax and set(125)** (uncompressed bytes per block, G = 1 and G = 4):

| shape | minmax | set(125) |
|---|---|---|
| distinct (D 8 192 / 32 768) | 16 | 8 (overflow: stored empty) |
| clustered, G = 1 (D = 100) | 16 | 808 = 8 + 100 x 8 |
| clustered, G = 4 (D = 400) | 16 | 8 (overflow) |
| ten (D = 10) | 16 | 88 = 8 + 10 x 8 |

`minmax` is 16 B per block (an `Int64` minimum and maximum): 1 968 B at 123
blocks, 496 B at 31. Compressed: `distinct` 1 287 / 367 B, `clustered` 1 061 /
309, `ten` 53 / 47. `set(125)` compressed: 49 680 B for `clustered` G = 1
(0.50 of 99 384 uncompressed), 3 691 for `ten` G = 1 (0.34), 986 for `ten`
G = 4 (0.36), 36 to 39 for the overflowed cases. `marks_bytes` of the
indexes was 121 to 305 B in every cell, varying with the cell (not modelled).

Observations only. Scope: Int64 values; the 8-byte value size in the `set`
row is that of `Int64` and was not varied; the per-block constant of the
bloom filter's byte count (v(D)) was fitted to D, and was not separated from
the row count of the block (the `ten` shape fixes D at 10 with 8 192 rows).
