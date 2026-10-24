"""
Does DataFusion query run-end-encoded (REE) columns? (spec
docs/superpowers/specs/2026-10-21-run-end-encoding-design.md §2)

Registers the same data plain and run-end-encoded (Int64 and Utf8 columns with nulls), runs each
query on both and prints, per operator: works (results identical), DIFFERS (results differ) or
FAILS (the error). Also reads REE back from an Arrow IPC file. Re-run after a DataFusion upgrade:
REE is recommended only once MIN / MAX work (the spec's re-evaluation trigger).

Run: python tests/performance/spike_datafusion_ree.py
"""

from __future__ import annotations

import os
import tempfile

import datafusion
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc
from datafusion import SessionContext

QUERIES = {
    "projection": "select k, i, s from t where k in (3, 9, 10, 11) order by k",
    "filter =": "select count(*) from t where i = 2",
    "filter BETWEEN": "select count(*) from t where i between 2 and 3",
    "filter = (string)": "select count(*) from t where s = 'b'",
    "filter IN": "select count(*) from t where s in ('b', 'd')",
    "filter LIKE": "select count(*) from t where s like 'b%'",
    "IS NULL": "select count(*) from t where i is null",
    "string function": "select count(*) from t where upper(s) = 'B'",
    "SUM": "select sum(i) from t",
    "arithmetic": "select sum(i + 1) from t",
    "CAST": "select count(cast(i as varchar)) from t",
    "COUNT DISTINCT": "select count(distinct s) from t",
    "MIN / MAX": "select min(i), max(i) from t",
    "MIN / MAX (string)": "select min(s), max(s) from t",
    "GROUP BY": "select s, count(*) c from t group by s order by s nulls first",
    "ORDER BY": "select i from t order by i nulls first, k limit 5",
    "window": "select k, count(*) over (partition by s) c from t order by k limit 5",
    "join": "select count(*) from t join d on t.i = d.i",
    "UNION ALL": "select count(*) from (select i from t union all select i from d)",
}


def tables() -> tuple[pa.Table, pa.Table, pa.Table]:
    i = pa.array([1, 1, 1, 2, 2, 3, 3, 3, 3, None, None, 4] * 1_000, pa.int64())
    s = pa.array(["a", "a", "b", "b", "b", "c", None, None, "d", "d", "d", "d"] * 1_000)
    k = pa.array(range(12_000), pa.int64())
    plain = pa.table({"i": i, "s": s, "k": k})
    ree = pa.table({"i": pc.run_end_encode(i), "s": pc.run_end_encode(s), "k": k})
    dim = pa.table(
        {
            "i": pa.array([1, 2, 3, 4], pa.int64()),
            "name": ["one", "two", "three", "four"],
        }
    )
    return plain, ree, dim


def context(t: pa.Table, dim: pa.Table) -> SessionContext:
    ctx = SessionContext()
    ctx.register_record_batches("t", [t.to_batches()])
    ctx.register_record_batches("d", [dim.to_batches()])
    return ctx


def decoded(t: pa.Table) -> list[dict]:
    cols = {
        n: (pc.run_end_decode(c) if pa.types.is_run_end_encoded(c.type) else c)
        for n, c in zip(t.column_names, t.columns)
    }
    return pa.table(cols).to_pylist()


def main() -> None:
    print(f"datafusion {datafusion.__version__}, pyarrow {pa.__version__}")
    plain, ree, dim = tables()
    cp, cr = context(plain, dim), context(ree, dim)
    for name, sql in QUERIES.items():
        want = cp.sql(sql).to_arrow_table().to_pylist()
        try:
            got = decoded(cr.sql(sql).to_arrow_table())
        except Exception as e:  # DataFusion raises a generic Exception
            print(f"FAILS    {name}: {str(e).splitlines()[0][:160]}")
            continue
        print(f"{'works' if got == want else 'DIFFERS':<8} {name}")
    # DataFusion keeps the registered file open; on Windows the directory cannot be removed then.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = os.path.join(tmp, "t.arrow")
        with ipc.new_file(path, ree.schema) as w:
            w.write_table(ree)
        ctx = SessionContext()
        try:
            ctx.register_arrow("f", path)
            n = ctx.sql("select count(*) from f where i = 2").to_arrow_table()
            ok = n.column(0)[0].as_py() == 2_000
            print(f"{'works' if ok else 'DIFFERS':<8} Arrow IPC read")
        except Exception as e:
            print(f"FAILS    Arrow IPC read: {str(e).splitlines()[0][:160]}")


if __name__ == "__main__":
    main()
