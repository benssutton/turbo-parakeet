import numpy as np
import polars as pl

from analytics.gcd.base import Gcd

_NATIVE = {
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
}
_SIGNED_MIN = {
    pl.Int8: -(2**7),
    pl.Int16: -(2**15),
    pl.Int32: -(2**31),
    pl.Int64: -(2**63),
}


def numpy_gcd(series: pl.Series) -> int:
    phys = series.to_physical().drop_nulls()
    if phys.len() == 0:
        return 0
    base = phys.dtype.base_type()
    if base in _NATIVE and phys.min() != _SIGNED_MIN.get(base):
        values = phys.to_numpy()
    else:
        # 128-bit physical values and |MIN| of a signed type do not fit numpy's
        # native ints: fall back to an object array of Python ints.
        values = np.array(phys.to_list(), dtype=object)
    # reduce() of one element returns it unchanged (possibly negative): take abs.
    return abs(int(np.gcd.reduce(values)))


class GcdNumpy(Gcd):
    """numpy.gcd.reduce per column (vectorised, single core); object arrays for
    values outside numpy's native integer range."""

    def _compute(self, frames, combos):
        return self.metrics_frame(
            combos, {"gcd": [numpy_gcd(frames[n][c]) for ((n, c),) in combos]}
        )
