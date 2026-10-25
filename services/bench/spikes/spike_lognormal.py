"""Spike 4: lognormal support values with density-weighted frequencies, and
solving the log-sigma for a target index of dispersion (var/mean).

Interpretation: `mean` is the mean of the lognormal variate before `shift`;
mu_log = ln(mean) - sigma^2 / 2. Values are clipped to [lo, hi] then rounded.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
from scipy.stats import lognorm

# Resolve the repository root from this file, not the cwd, so the script runs
# correctly from any working directory (including spikes/ itself).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from spikes._common import cardinality_ratio, index_of_dispersion, largest_remainder


def lognormal_multiset(
    n: int,
    k: int,
    mean: float,
    sigma_log: float,
    shift: float,
    lo: float,
    hi: float,
    dp: int,
    seed: int,
):
    rng = np.random.default_rng(seed)
    mu_log = np.log(mean) - sigma_log**2 / 2
    raw = rng.lognormal(mu_log, sigma_log, size=k) + shift
    clipped = np.clip(raw, lo, hi)
    clipped_frac = float(((raw < lo) | (raw > hi)).mean())
    support = np.unique(np.round(clipped, dp))
    density = lognorm(s=sigma_log, scale=np.exp(mu_log)).pdf(support - shift)
    density = np.where(density > 0, density, density[density > 0].min())
    counts = largest_remainder(density, n - len(support)) + 1
    # clipped_draw_mean is the mean of the clipped-but-not-yet-deduplicated
    # draws: recording it lets a reader decompose the gap between it and the
    # final multiset mean (x.mean()) into the deduplication step (collapsing
    # repeated clipped values onto one support point) and the density-weighted
    # count allocation step below, rather than attributing the whole gap to
    # clipping alone (see FINDINGS.md section 4).
    info = {
        "clipped_frac": clipped_frac,
        "support_shortfall": k - len(support),
        "clipped_draw_mean": float(clipped.mean()),
    }
    return support, counts, info


def solve_sigma(
    n, k, mean, shift, lo, hi, dp, target_id, seed, s_lo=0.01, s_hi=3.0, iters=40
):
    """Bisect sigma_log against the multiset's index of dispersion.

    Assumes the final multiset mean (clipped, rounded, and density-weighted
    by lognormal_multiset -- not just the clipped draws) is monotone in
    sigma across [s_lo, s_hi], i.e. it changes sign at most once. That holds
    for the configurations this spike tested, where increasing sigma only
    ever pushes more mass past the upper clip bound. It is not guaranteed in
    general: with a lower clip bound that actually bites, raising sigma
    could push mass off the *low* end instead and raise the mean back up,
    giving more than one sign change. Callers with such a configuration
    should not rely on the narrowing below finding a single valid crossing.
    """

    def stats_at(s):
        support, counts, _ = lognormal_multiset(n, k, mean, s, shift, lo, hi, dp, seed)
        x = np.repeat(support, counts)
        return float(x.mean()), index_of_dispersion(x)

    mean_lo, id_lo = stats_at(s_lo)
    if mean_lo <= 0:
        return (
            s_lo,
            id_lo,
            f"invalid bracket: mean already <= 0 at s_lo={s_lo} (mean={mean_lo:.4g})",
        )

    mean_hi, id_hi = stats_at(s_hi)
    if mean_hi <= 0:
        # index_of_dispersion = var/mean is only a monotone, meaningful function
        # of sigma while the final multiset mean stays positive: once sigma is
        # large enough, the clipped, rounded, and density-weighted multiset
        # mean (not merely the mean of the clipped draws, which can still be
        # comfortably positive here) crosses zero, and var/mean flips sign and
        # diverges instead of continuing to rise, which breaks the assumption
        # bisection relies on. Detect that directly (bisect on the sign of the
        # multiset mean, not on ID) and narrow s_hi to the largest sigma below
        # the caller's s_hi where that mean is still positive, rather than
        # assuming s_hi is usable.
        pos, neg = s_lo, s_hi
        for _ in range(iters):
            mid = (pos + neg) / 2
            m_mid, _ = stats_at(mid)
            if m_mid > 0:
                pos = mid
            else:
                neg = mid
        s_hi = pos
        mean_hi, id_hi = stats_at(s_hi)
        if mean_hi <= 0:
            return (
                s_lo,
                id_lo,
                "invalid bracket: no positive-mean sigma found below s_hi",
            )

    if target_id <= id_lo:
        return s_lo, id_lo, f"clamped: min reachable {id_lo:.3f}"
    if target_id >= id_hi:
        return (
            s_hi,
            id_hi,
            f"clamped: max reachable {id_hi:.3f} at sigma={s_hi:.4f} (edge of mean-positive bracket)",
        )

    for _ in range(iters):
        mid = (s_lo + s_hi) / 2
        if stats_at(mid)[1] < target_id:
            s_lo = mid
        else:
            s_hi = mid
    s = (s_lo + s_hi) / 2
    return s, stats_at(s)[1], "ok"


def main() -> None:
    n, k = 100_000, 75_000
    args = dict(mean=4.0, shift=-1.0, lo=-1.0, hi=10.0, dp=8)
    print("--- shape scan ---")
    print(
        f"{'sigma':>6} {'seed':>4} {'mean':>7} {'clipmean':>8} {'ID':>7} {'card':>7} "
        f"{'clipped%':>9} {'shortfall':>9}"
    )
    for sigma in (0.25, 0.5, 1.0):
        for seed in (0, 1, 2):
            support, counts, info = lognormal_multiset(
                n, k, sigma_log=sigma, seed=seed, **args
            )
            x = np.repeat(support, counts)
            print(
                f"{sigma:>6.2f} {seed:>4} {x.mean():>7.3f} {info['clipped_draw_mean']:>8.4f} "
                f"{index_of_dispersion(x):>7.3f} {cardinality_ratio(x):>7.4f} "
                f"{100*info['clipped_frac']:>9.2f} {info['support_shortfall']:>9}"
            )

    print("--- solve sigma for target index of dispersion ---")
    print(f"{'target ID':>9} {'seed':>4} {'sigma':>7} {'achieved':>9}  status")
    for target in (0.5, 1.0, 2.0, 5.0):
        for seed in (0, 1, 2):
            s, got, status = solve_sigma(n, k, target_id=target, seed=seed, **args)
            print(f"{target:>9.2f} {seed:>4} {s:>7.4f} {got:>9.4f}  {status}")


if __name__ == "__main__":
    main()
