import math

import numpy as np
import pyarrow as pa

from synthdata.frame.measure import column_stats, containment, joint_distinct, nmi


def test_stats_of_a_sorted_low_cardinality_column():
    v = pa.array(np.repeat(np.arange(10) * 1000, 100))
    s = column_stats(v, rows=1_100, integer=True)
    assert s["null_count"] == 100 and s["distinct"] == 10
    assert math.isclose(s["entropy_norm"], 1.0)
    assert s["descents"] == 0 and s["rho"] > 0.95
    assert s["gcd"] == 1000 and s["max_step"] == 1000
    assert s["run_ratio"] > 50


def test_stats_of_constant_and_empty():
    c = column_stats(pa.array([5] * 10), rows=10, integer=True)
    assert c["distinct"] == 1 and math.isnan(c["entropy_norm"]) and math.isnan(c["rho"])
    e = column_stats(pa.array([], pa.string()), rows=4, integer=False)
    assert e["distinct"] == 0 and e["null_count"] == 4 and e["gcd"] is None


def test_relation_metrics():
    a = pa.array(["x", "y", "z", None])
    b = pa.array(["x", "x", "y", None])
    assert containment(b, a) == 1.0 and containment(a, b) == 2 / 3
    assert math.isclose(nmi(a, pa.array([1, 2, 3, None])), 1.0)
    assert joint_distinct(pa.array([1, 1, 2]), pa.array([1, 2, 1])) == 3
