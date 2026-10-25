import numpy as np
import pyarrow as pa
import pytest

from synthdata.frame.spec import (
    Clustered,
    ColumnSpec,
    Multiples,
    Random,
    Runs,
    Sorted,
    Walk,
)
from synthdata.frame.values import (
    apply_order,
    base_values,
    dense_ids,
    null_mask,
    scatter,
)


def seq(n=0):
    return np.random.SeedSequence([7, n])


@pytest.mark.parametrize("type_", ["int64", "string", "float64", "date32", "decimal"])
def test_distinct_count_is_exact(type_):
    v, warnings = base_values(ColumnSpec(type_, distinct=37), 2_000, seq())
    assert len(v) == 2_000 and len(set(v.to_pylist())) == 37 and not warnings


def test_distinct_share_and_entropy():
    v, _ = base_values(ColumnSpec("int64", distinct_share=0.5), 1_000, seq())
    assert len(set(v.to_pylist())) == 500
    h, _ = base_values(ColumnSpec("int64", distinct=10, entropy=0.4), 2_000, seq())
    counts = np.unique(h.to_numpy(), return_counts=True)[1] / 2_000
    assert abs(-(counts * np.log2(counts)).sum() / np.log2(10) - 0.4) < 0.02


def test_multiples_and_walk():
    m, _ = base_values(
        ColumnSpec("int64", distinct=50, structure=Multiples(1000)), 500, seq()
    )
    assert all(x % 1000 == 0 for x in m.to_pylist())
    w, _ = base_values(ColumnSpec("int64", structure=Walk(8)), 500, seq())
    assert np.abs(np.diff(w.to_numpy())).max() <= 8


def test_orders():
    rng = np.random.default_rng(0)
    v = pa.array(np.repeat(np.arange(20), 50))
    s = dense_ids(apply_order(v, Sorted(), rng))
    assert (np.diff(s) >= 0).all()
    c = dense_ids(apply_order(v, Clustered(0.1), rng))
    assert 0.8 < np.corrcoef(np.arange(c.size), c)[0, 1] < 1
    r = dense_ids(apply_order(v, Runs(8), rng))
    assert c.size / (1 + np.count_nonzero(np.diff(r))) > 5
    assert sorted(apply_order(v, Random(), rng).to_pylist()) == v.to_pylist()


def test_null_masks():
    rng = np.random.default_rng(0)
    assert null_mask(100, 0, None, rng).sum() == 0
    m = null_mask(1_000, 100, None, rng)
    assert m.sum() == 100
    c = null_mask(1_000, 100, 10.0, rng)
    assert c.sum() == 100 and np.count_nonzero(np.diff(c.astype(int)) == 1) <= 12
    assert null_mask(5, 5, None, rng).all()


def test_scatter_puts_values_in_order_around_nulls():
    out = scatter(pa.array(["a", "b"]), np.array([True, False, True, False]))
    assert out.to_pylist() == [None, "a", None, "b"]
    assert (
        out.buffers()[2].size == 2
    )  # the data buffer holds "a" and "b" only: null slots are empty
    assert scatter(pa.array([], pa.int64()), np.ones(3, bool)).to_pylist() == [None] * 3
