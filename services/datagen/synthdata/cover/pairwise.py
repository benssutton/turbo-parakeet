"""The fewest columns covering every allowed pair of factor levels (greedy set cover)."""

from __future__ import annotations

import functools
import itertools

from synthdata.cover.factors import FACTORS, normalise

Pair = tuple[tuple[str, str], tuple[str, str]]
NAMES = tuple(FACTORS)


def pairs(levels: dict) -> set[Pair]:
    """The level pairs a column covers: factors that apply (not None, not "?")."""
    on = [(f, levels[f]) for f in NAMES if levels.get(f) not in (None, "?")]
    return set(itertools.combinations(on, 2))


@functools.cache
def _candidates() -> tuple[tuple, ...]:
    seen, out = set(), []
    for combo in itertools.product(*FACTORS.values()):
        lv = normalise(dict(zip(NAMES, combo)))
        if lv is not None:
            key = tuple(lv[f] for f in NAMES)
            if key not in seen:
                seen.add(key)
                out.append(key)
    return tuple(out)


def candidates() -> list[dict]:
    return [dict(zip(NAMES, key)) for key in _candidates()]


@functools.cache
def _allowed() -> frozenset:
    return frozenset().union(*(pairs(c) for c in candidates()))


def allowed_pairs() -> set[Pair]:
    return set(_allowed())


@functools.cache
def _covering() -> tuple[tuple, ...]:
    pool = [(key, pairs(dict(zip(NAMES, key)))) for key in _candidates()]
    uncovered = set(_allowed())
    chosen = []
    while uncovered:
        # The first maximum: deterministic.
        key, got = max(pool, key=lambda kp: len(kp[1] & uncovered))
        chosen.append(key)
        uncovered -= got
    return tuple(chosen)


def covering() -> list[dict]:
    return [dict(zip(NAMES, key)) for key in _covering()]
