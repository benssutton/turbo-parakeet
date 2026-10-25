import sys
from pathlib import Path

import pytest

import synthdata.scenarios as sc

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "tests"))
old = pytest.importorskip("datagen")  # the root tests/datagen.py, until it is retired

CASES = [
    ("mixed_dtypes", (300,)),
    ("low_cardinality", (2_000, 6)),
    ("describe_mixed", (500,)),
    ("related_frames", (200,)),
    ("containment_pairs", ()),
    ("similar_frames", ()),
    ("near_unique", (2_000,)),
    ("near_constant", (10_000,)),
    ("top_k_streams", (5_000,)),
    ("integer_multiples", (1_000, 3, 7)),
    ("integer_random", (1_000, 2)),
    ("runs_int", (1_000, 2, 10, 50)),
    ("runs_str", (1_000, 1, 10, 5)),
    ("describe_narrow", (1_000,)),
    ("describe_wide", (500, 12)),
    ("describe_nested", (500,)),
]


@pytest.mark.parametrize("name, args", CASES)
def test_identical_to_the_old_fixture(name, args):
    new, prev = getattr(sc, name)(*args), getattr(old, name)(*args)
    if isinstance(new, dict):
        assert new.keys() == prev.keys() and all(new[k].equals(prev[k]) for k in new)
    else:
        assert new.equals(prev)


def test_population_samples_and_measure_frame():
    df, truth = sc.population_samples(1_000, [(100, "uniform"), (50, ("zipf", 1.1))])
    assert truth == {"p0": 100, "p1": 50}
    man = sc.measure_frame(df)
    assert man.columns["p0"]["distinct"] <= 100 and man.rows == 1_000
