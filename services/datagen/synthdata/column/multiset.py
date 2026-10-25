"""Stage 1: which values appear and how often.

Order-free. Everything here is a property of the multiset, so stage 2 cannot
change any of it. Split across three tasks: the support draw (this), the
frequency profile and entropy solve, and the dispersion solve.
"""

from __future__ import annotations

import string
from dataclasses import dataclass, field, replace

import numpy as np
import pyarrow as pa
from scipy.stats import lognorm, norm

from synthdata.column.metrics import (
    entropy_of_counts,
    index_of_dispersion,
    largest_remainder,
)
from synthdata.column.solve import SolveResult, solve_monotone
from synthdata.column.spec import DatasetSpec

CLIP_WARN_FRACTION = 0.01  # FINDINGS §4
REDRAW_CAP = 20
STRING_ALPHABET = np.array(list(string.digits + string.ascii_letters))  # 62 symbols


def element_dtype(spec: DatasetSpec) -> pa.DataType:
    """The scalar type the values are drawn as: list<T> draws T."""
    dtype = spec.arrow_dtype
    return dtype.value_type if pa.types.is_list(dtype) else dtype


def is_lattice(dtype: pa.DataType) -> bool:
    """Types whose values are drawn as int64 on an integer lattice and cast later:
    integers, dates and times (integers in the unit's resolution)
    and bool (the lattice {0, 1})."""
    return (
        pa.types.is_integer(dtype)
        or pa.types.is_temporal(dtype)
        or pa.types.is_boolean(dtype)
    )


def random_strings(length: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """n random strings of `length` alphanumeric characters, as a <U array."""
    idx = rng.integers(0, STRING_ALPHABET.size, size=(n, length))
    return STRING_ALPHABET[idx].view(f"U{length}").ravel()


@dataclass
class SupportDraw:
    values: np.ndarray
    requested_k: int
    clipped_frac: float
    clip_collapsed: int
    round_collapsed: int
    warnings: list[str] = field(default_factory=list)


def _raw_draw(spec, rng, size, sigma):
    """Unclipped, unrounded variates for the DatasetSpec's distribution."""
    d = spec.distribution
    lo, hi = spec.value_range
    if d.kind == "uniform":
        return rng.uniform(lo, hi, size=size)
    if d.kind == "normal":
        if sigma is None:
            raise ValueError(
                "distribution.sigma is None and no sigma override was "
                "supplied; pass sigma explicitly or set index_of_dispersion "
                "so the dispersion solver can supply one"
            )
        mean = d.mean if d.mean is not None else (lo + hi) / 2
        return rng.normal(mean, sigma, size=size) + d.shift
    if d.kind == "lognormal":
        if sigma is None:
            raise ValueError(
                "distribution.sigma is None and no sigma override was "
                "supplied; pass sigma explicitly or set index_of_dispersion "
                "so the dispersion solver can supply one"
            )
        if d.mean is None:
            raise ValueError(
                "distribution.mean is None; a lognormal distribution "
                "requires a positive mean"
            )
        mu_log = np.log(d.mean) - sigma**2 / 2  # mean is BEFORE the shift
        return rng.lognormal(mu_log, sigma, size=size) + d.shift
    raise ValueError(f"unknown distribution kind {d.kind!r}")


def draw_support(
    spec: DatasetSpec, rng: np.random.Generator, sigma: float | None = None
) -> SupportDraw:
    """Draw k distinct support values, counting the two collapse mechanisms apart.

    Every value_range support is drawn *from the distribution*:
    lattice types (integers, dates, times, bool) draw through `_raw_draw`
    like floats and are rounded to the lattice instead of to decimal_places.
    The one exception is the uniform kind on a lattice, which is a sample
    without replacement from the lattice itself. A uniform support under a
    non-uniform distribution would be wrong whenever k is far below the
    lattice span: the density profile that follows can only weight the points
    it is given, so a support spread over the whole range stays spread.

    Clip collapse: every draw past a value_range bound is clipped to that bound and
    then rounds to it, so all of them merge onto one support point. Dominant, and
    NOT recoverable by redrawing, because fresh draws past the bound collapse the
    same way. Remedy is to request a larger k or accept the lower ratio; either way
    warn above CLIP_WARN_FRACTION.

    Rounding collapse: ordinary collisions after rounding to decimal_places (or to
    the lattice). Minor, and recoverable, so redraw for this case only, up to
    REDRAW_CAP.

    Capacity: a request beyond the number of representable points (the lattice
    span, or `(hi - lo) * 10**decimal_places + 1` for floats and decimals) is
    unsatisfiable by construction, not by chance. The draw is clamped to the
    capacity and reported as a `clamp: cardinality_ratio` warning;
    `clip_collapsed + round_collapsed` then account for the shortfall against
    the clamped count, not against `requested_k`. A clipped fraction above
    CLIP_WARN_FRACTION is a `miss: cardinality_ratio: at risk` warning.
    """
    k = spec.support_size

    if spec.categories is not None:
        if pa.types.is_boolean(element_dtype(spec)):
            # bool categories draw on the lattice {0, 1} like a bool value_range.
            values = np.array(sorted(int(c) for c in spec.categories), dtype=np.int64)
        else:
            values = np.array(sorted(spec.categories))
        return SupportDraw(
            values=values,
            requested_k=k,
            clipped_frac=0.0,
            clip_collapsed=0,
            round_collapsed=0,
            warnings=[],
        )

    if spec.string_length is not None:
        return _draw_strings(spec, rng, k)

    lo, hi = spec.value_range
    lattice = is_lattice(element_dtype(spec))
    warnings: list[str] = []

    if lattice:
        lo, hi = int(lo), int(hi)
        capacity = hi - lo + 1
        where = f"the integer lattice [{lo}, {hi}]"
    else:
        capacity = int(round((hi - lo) * 10**spec.decimal_places)) + 1
        where = f"the range [{lo}, {hi}] at {spec.decimal_places} decimal " "places"
    if k > capacity:
        warnings.append(
            f"clamp: cardinality_ratio: {k} distinct values requested "
            f"but {where} holds only {capacity}; the support is clamped to "
            f"{capacity}"
        )
    n = min(k, capacity)

    if lattice and spec.distribution.kind == "uniform":
        values = _uniform_lattice(lo, hi, n, rng)
        return SupportDraw(
            values=values,
            requested_k=k,
            clipped_frac=0.0,
            clip_collapsed=0,
            round_collapsed=0,
            warnings=warnings,
        )

    def to_grid(x: np.ndarray) -> np.ndarray:
        if lattice:
            return np.round(x).astype(np.int64)
        return np.round(x, spec.decimal_places)

    sig = sigma if sigma is not None else spec.distribution.sigma

    raw = _raw_draw(spec, rng, n, sig)
    below = raw < lo
    above = raw > hi
    n_outside = int(np.count_nonzero(below | above))
    clipped_frac = n_outside / n
    values = np.unique(to_grid(np.clip(raw, lo, hi)))

    # Each bound actually hit retains exactly one support point; a draw past
    # only the upper bound collapses onto one point, a distribution wide
    # enough to clip both bounds collapses onto two. Counting a flat "minus
    # one" here overstates clip_collapsed whenever both bounds are hit.
    bounds_hit = int(np.any(below)) + int(np.any(above))
    clip_collapsed = max(0, n_outside - bounds_hit)
    round_collapsed = n - values.size - clip_collapsed

    if round_collapsed > 0 and clipped_frac <= CLIP_WARN_FRACTION:
        missing = round_collapsed
        attempts = 0
        while missing > 0 and attempts < REDRAW_CAP:
            extra_raw = _raw_draw(spec, rng, missing, sig)
            # Drop out-of-range redraws rather than clipping them: clipping
            # here would merge them onto a bound, silently converting
            # rounding collapse into unaccounted clip collapse.
            in_range = extra_raw[(extra_raw >= lo) & (extra_raw <= hi)]
            values = np.union1d(values, to_grid(in_range))
            missing = n - values.size - clip_collapsed
            attempts += 1
        round_collapsed = n - values.size - clip_collapsed

    if clipped_frac > CLIP_WARN_FRACTION:
        warnings.append(
            f"miss: cardinality_ratio: at risk: {clipped_frac * 100:.1f}% of "
            "the support draw clipped"
        )

    values = np.sort(np.unique(values))

    return SupportDraw(
        values=values,
        requested_k=k,
        clipped_frac=clipped_frac,
        clip_collapsed=clip_collapsed,
        round_collapsed=round_collapsed,
        warnings=warnings,
    )


def _uniform_lattice(lo: int, hi: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """n distinct integers uniform on [lo, hi], sorted.

    `rng.choice(span)` overflows once the span exceeds int64, which int64 and
    uint64 at their full range legitimately do; those draw with replacement
    through `rng.integers` at the type's own width and redraw the (vanishingly
    rare) collisions instead. The result is uint64 only when it has to be.
    """
    span = hi - lo + 1
    if span <= 2**63 - 1:
        return np.sort(rng.choice(span, size=n, replace=False) + lo).astype(np.int64)
    dtype = np.uint64 if hi > 2**63 - 1 else np.int64
    values = np.unique(rng.integers(lo, hi, size=n, endpoint=True, dtype=dtype))
    attempts = 0
    while values.size < n and attempts < REDRAW_CAP:
        extra = rng.integers(lo, hi, size=n - values.size, endpoint=True, dtype=dtype)
        values = np.union1d(values, extra)
        attempts += 1
    return values


def _draw_strings(spec: DatasetSpec, rng: np.random.Generator, k: int) -> SupportDraw:
    """k distinct random strings of spec.string_length. Collisions
    are the strings' rounding collapse: redraw the shortfall up to REDRAW_CAP.
    A request beyond the alphabet's capacity is clamped with a warning."""
    length = spec.string_length
    capacity = STRING_ALPHABET.size**length
    warnings: list[str] = []
    if k > capacity:
        warnings.append(
            f"clamp: cardinality_ratio: {k} distinct strings requested but "
            f"{STRING_ALPHABET.size}^{length} = {capacity} exist at "
            f"string_length {length}; the support is clamped to {capacity}"
        )
    n = min(k, capacity)
    values = np.unique(random_strings(length, n, rng))
    attempts = 0
    while values.size < n and attempts < REDRAW_CAP:
        values = np.union1d(values, random_strings(length, n - values.size, rng))
        attempts += 1
    return SupportDraw(
        values=values,
        requested_k=k,
        clipped_frac=0.0,
        clip_collapsed=0,
        round_collapsed=n - values.size,
        warnings=warnings,
    )


def draw_uncapped(
    spec: DatasetSpec, rng: np.random.Generator, sigma: float | None = None
) -> np.ndarray:
    """The uncapped case: no cardinality_ratio and no
    categories, so draw `rows` values straight from the distribution, clipped
    and rounded, and let their multiplicities be the frequency profile.

    There is no profile step on this path, so the distribution must enter at
    the draw itself: lattice types (integers, dates, times, bool) draw through
    `_raw_draw` like everything else and are then rounded to the lattice. The
    capped path can afford a uniform support because `density_counts` shapes
    the frequencies afterwards; here nothing would.
    """
    rows = spec.rows
    if spec.string_length is not None:
        return random_strings(spec.string_length, rows, rng)
    lo, hi = spec.value_range
    sig = sigma if sigma is not None else spec.distribution.sigma
    raw = _raw_draw(spec, rng, rows, sig)
    if is_lattice(element_dtype(spec)):
        return np.clip(np.round(raw), int(lo), int(hi)).astype(np.int64)
    return np.round(np.clip(raw, lo, hi), spec.decimal_places)


def entropy_floor(rows: int, k: int) -> float:
    """Minimum reachable entropy when every one of k support values appears at least
    once. Strictly positive, and rises with k (FINDINGS §1)."""
    if k <= 1:
        return 0.0
    p_heavy = (rows - k + 1) / rows
    return float(-p_heavy * np.log2(p_heavy) - (k - 1) * (1 / rows) * np.log2(1 / rows))


def entropy_ceiling(k: int) -> float:
    return float(np.log2(k))


def powerlaw_counts(rows: int, k: int, s: float) -> np.ndarray:
    """f_i proportional to rank^-s, with one occurrence guaranteed per value."""
    w = np.arange(1, k + 1, dtype=np.float64) ** (-s)
    return largest_remainder(w, rows - k) + 1


def solve_entropy(
    rows: int, k: int, target: float, iters: int = 60, tol: float | None = None
) -> SolveResult:
    """Bisect the power-law exponent. Entropy is monotone DECREASING in s, and the
    reachable range is [entropy_floor, entropy_ceiling]. `tol` is the
    DatasetSpec's entropy tolerance: a landing outside it is NOT_CONVERGED, not ok."""
    return solve_monotone(
        lambda s: entropy_of_counts(powerlaw_counts(rows, k, s)),
        target=target,
        lo=0.0,
        hi=30.0,
        increasing=False,
        iters=iters,
        tol=tol,
    )


def density_counts(
    support: np.ndarray, spec: DatasetSpec, rows: int, sigma: float | None
) -> np.ndarray:
    """Weight each support value by the distribution's own density, allocate
    rows - k by largest remainder, then add one per value so every value appears
    (largest-remainder rounding, so the counts sum exactly to `rows`).

    The lognormal's shift is applied after drawing (see _raw_draw), so its density
    must be evaluated on support - shift, not on support itself. Any density that
    underflows to zero is floored to the smallest positive density observed, so an
    underflowed tail cannot produce an all-zero weight vector. A nan density
    (a sigma at or below zero, a nan support value) raises: validate() rejects
    such a sigma, so nan here is a regression, and masking it to an even split
    as this once did would hide it.
    """
    d = spec.distribution
    if d.kind == "uniform":
        weights = np.ones(support.size, dtype=np.float64)
    elif d.kind == "normal":
        sig = sigma if sigma is not None else d.sigma
        if sig is None:
            raise ValueError(
                "distribution.sigma is None and no sigma override was "
                "supplied; pass sigma explicitly or set index_of_dispersion "
                "so the dispersion solver can supply one"
            )
        if d.mean is not None:
            mean = d.mean
        else:  # validate() guarantees value_range here
            lo, hi = spec.value_range
            mean = (lo + hi) / 2
        weights = norm.pdf(support - d.shift, loc=mean, scale=sig)
    elif d.kind == "lognormal":
        sig = sigma if sigma is not None else d.sigma
        if sig is None:
            raise ValueError(
                "distribution.sigma is None and no sigma override was "
                "supplied; pass sigma explicitly or set index_of_dispersion "
                "so the dispersion solver can supply one"
            )
        if d.mean is None:
            raise ValueError(
                "distribution.mean is None; a lognormal distribution "
                "requires a positive mean"
            )
        mu_log = np.log(d.mean) - sig**2 / 2  # mean is BEFORE the shift
        weights = lognorm.pdf(support - d.shift, s=sig, scale=np.exp(mu_log))
    else:
        raise ValueError(f"unknown distribution kind {d.kind!r}")

    weights = np.asarray(weights, dtype=np.float64)
    if np.isnan(weights).any():
        raise ValueError(
            f"density_counts: the {d.kind} density is nan on the support "
            f"(sigma={sigma if sigma is not None else d.sigma!r}); validate() "
            "should have rejected this spec, so a caller's sigma override or "
            "support is at fault"
        )
    positive = weights[weights > 0]
    floor = float(positive.min()) if positive.size > 0 else 1.0
    weights = np.where(weights > 0, weights, floor)

    k = support.size
    return largest_remainder(weights, rows - k) + 1


@dataclass
class Multiset:
    support: np.ndarray
    counts: np.ndarray
    warnings: list[str] = field(default_factory=list)
    solves: dict[str, SolveResult] = field(default_factory=dict)
    # Support shortfall by mechanism (the shortfall and which mechanism
    # caused it are both recorded). Zero on the uncapped and categories paths.
    clip_collapsed: int = 0
    round_collapsed: int = 0


def build_multiset(
    spec: DatasetSpec, rng: np.random.Generator, sigma: float | None = None
) -> Multiset:
    """Stage 1: the support and how many times each value appears.

    An entropy target picks a power-law exponent by bisection against the ACTUAL
    support size (which may be below spec.support_size after collapse); otherwise
    counts follow the distribution's own density.

    sigma: an explicit spread for the distribution, threaded through to
    draw_support and density_counts. When it is None, spec.distribution.sigma is
    None and spec.index_of_dispersion is set, the dispersion solve (4.2 step 3)
    chooses it first and the real draw then uses the solved value. Passing sigma
    explicitly never solves, which is what lets the solve's trial builds call
    back into this function without recursing.

    The solve's trial builds use a fresh generator seeded from spec.seed, so the
    returned multiset matches the one the solve measured only when rng is at its
    initial state on entry; the recorded achieved is therefore re-measured on the
    multiset actually returned, with a warning if the two disagree.
    """
    warnings: list[str] = []
    solves: dict[str, SolveResult] = {}

    if (
        sigma is None
        and spec.distribution.sigma is None
        and spec.index_of_dispersion is not None
    ):
        result = solve_dispersion(spec, rng, spec.index_of_dispersion)
        solves["index_of_dispersion"] = result
        if not result.ok:
            warnings.append(
                result.warning("index_of_dispersion", spec.index_of_dispersion)
            )
        sigma = result.x

    if spec.support_size is None:
        # Uncapped: the drawn values' own multiplicities are the profile, and
        # the frequency profile and expansion are skipped. entropy_bits cannot be
        # set here; validate() requires a finite support for it.
        support, counts = np.unique(draw_uncapped(spec, rng, sigma), return_counts=True)
        return _finish(support, counts, spec, warnings, solves)

    draw = draw_support(spec, rng, sigma=sigma)
    warnings.extend(draw.warnings)
    # The draw warns on the clipped *fraction* and the capacity clamp; the
    # counts that survive the redraw loop are otherwise silent, and a rounding
    # collapse the cap could not close (a distribution concentrated on few
    # grid points) leaves the achieved cardinality short with no record.
    if draw.clip_collapsed > 0:
        warnings.append(
            f"miss: cardinality_ratio: short by clip collapse: "
            f"{draw.clip_collapsed} of {draw.requested_k} requested support "
            "values merged onto a value_range bound"
        )
    if draw.round_collapsed > 0:
        warnings.append(
            f"miss: cardinality_ratio: short by rounding collapse: "
            f"{draw.round_collapsed} of {draw.requested_k} requested support "
            f"values collided after rounding and could not be redrawn within "
            f"{REDRAW_CAP} attempts"
        )

    k = draw.values.size
    rows = spec.rows

    if spec.entropy_bits is not None:
        result = solve_entropy(
            rows, k, spec.entropy_bits, tol=spec.tolerances.entropy_bits
        )
        solves["entropy_bits"] = result
        if not result.ok:
            warnings.append(result.warning("entropy_bits", spec.entropy_bits))
        # powerlaw_counts puts rank 1 (the heaviest) at index 0 and the support
        # is sorted, so without a permutation the most frequent value would
        # always be the smallest. Entropy and cardinality are order-free, so
        # shuffling the rank-to-value assignment changes neither.
        counts = rng.permutation(powerlaw_counts(rows, k, result.x))
    else:
        counts = density_counts(draw.values, spec, rows, sigma)

    return _finish(
        draw.values,
        counts,
        spec,
        warnings,
        solves,
        clip_collapsed=draw.clip_collapsed,
        round_collapsed=draw.round_collapsed,
    )


def _finish(
    support: np.ndarray,
    counts: np.ndarray,
    spec: DatasetSpec,
    warnings: list[str],
    solves: dict[str, SolveResult],
    *,
    clip_collapsed: int = 0,
    round_collapsed: int = 0,
) -> Multiset:
    """Re-measure a recorded dispersion solve on the multiset actually returned
    (see build_multiset) and assemble the result."""
    if "index_of_dispersion" in solves:
        solved = solves["index_of_dispersion"]
        measured = index_of_dispersion(np.repeat(support, counts))
        if abs(measured - solved.achieved) > spec.tolerances.index_of_dispersion:
            warnings.append(
                f"miss: index_of_dispersion: re-measured on the returned "
                f"multiset as {measured:.6g}, not the solve's "
                f"{solved.achieved:.6g}: the caller's generator was not at its "
                "initial state, so the final draw differs from the one the "
                "solve measured"
            )
        solves["index_of_dispersion"] = replace(solved, achieved=measured)

    return Multiset(
        support=support,
        counts=counts,
        warnings=warnings,
        solves=solves,
        clip_collapsed=clip_collapsed,
        round_collapsed=round_collapsed,
    )


def _generated(spec: DatasetSpec, seed: int, sigma: float) -> np.ndarray:
    """The final expanded multiset at a trial sigma. Deterministic in (seed, sigma),
    which is what makes the bisection valid. Passing sigma explicitly means
    build_multiset never re-enters the solve."""
    m = build_multiset(spec, np.random.default_rng(seed), sigma=sigma)
    return np.repeat(m.support, m.counts)


def solve_dispersion(
    spec: DatasetSpec,
    rng: np.random.Generator,
    target: float,
    lo: float = 0.01,
    hi: float = 3.0,
) -> SolveResult:
    """Bisect the distribution's spread against the MEASURED index of dispersion.

    index_of_dispersion(sigma) is not monotone across a wide bracket: the generated
    multiset mean falls with sigma and crosses zero, so the metric has a pole there
    and returns large and negative beyond it. Guarding on the sign of the generated
    mean narrows the bracket to the branch where the metric is meaningful, which is
    what stops a naive bracket reporting every target as unreachable.

    No sigma bound is hardcoded: the crossing moves with mean, shift and the clip
    bounds, all of which are spec parameters.

    rng is accepted for interface symmetry with the other stage-1 entry points but
    the trial builds each start a fresh generator from spec.seed, because a build
    that consumed rng would make f(sigma) depend on call order.
    """
    seed = spec.seed
    # The guard and the objective probe the same sigmas (the bracket ends):
    # build each trial multiset once and keep its (mean, index of dispersion).
    built: dict[float, tuple[float, float]] = {}

    def trial(s: float) -> tuple[float, float]:
        if s not in built:
            x = _generated(spec, seed, s)
            built[s] = (float(x.mean()), index_of_dispersion(x))
        return built[s]

    return solve_monotone(
        lambda s: trial(s)[1],
        target=target,
        lo=lo,
        hi=hi,
        increasing=True,
        guard=lambda s: trial(s)[0] > 0,
        tol=spec.tolerances.index_of_dispersion,
    )
