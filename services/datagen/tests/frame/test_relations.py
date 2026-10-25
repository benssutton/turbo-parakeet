import numpy as np
import pyarrow as pa

from synthdata.frame.measure import containment, joint_distinct, nmi
from synthdata.frame.relations import derive
from synthdata.frame.spec import CompositeKey, ForeignKey, NoisyCopy, Relabel, Subset

RNG = lambda: np.random.default_rng(3)  # noqa: E731
PARENT = pa.array(np.repeat(np.arange(50), 20))  # 50 values x 20 rows
UNIQUE = pa.array([f"k{i}" for i in range(1_000)])


def test_noisy_copy():
    out = derive(NoisyCopy("p", 0.2), PARENT, RNG())
    assert len(out) == 1_000 and containment(out, PARENT) == 1.0
    assert 0.7 < (np.asarray(out) == np.asarray(PARENT)).mean() < 0.9


def test_relabel_is_a_bijection():
    out = derive(Relabel("p", "string"), PARENT, RNG())
    assert out.type == pa.string() and abs(nmi(out, PARENT) - 1) < 1e-9


def test_foreign_key():
    out = derive(ForeignKey("p", null_share=0.1, zipf_exponent=1.1), UNIQUE, RNG())
    assert out.null_count == 100 and containment(out, UNIQUE) == 1.0


def test_subset():
    out = derive(Subset("p", 0.5), UNIQUE, RNG())
    assert abs(containment(out, UNIQUE) - 0.5) < 0.05


def test_composite_key():
    out = derive(CompositeKey("p"), PARENT, RNG())
    assert joint_distinct(PARENT, out) == 1_000
    assert len(set(out.to_pylist())) == 20
