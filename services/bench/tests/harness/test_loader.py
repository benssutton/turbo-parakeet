import pytest

from chbench.datagen import DatasetSpec, generate
from chbench.harness.ddl import table_name
from chbench.harness.loader import load_table, storage_stats
from chbench.harness.validate import TableVariant


def variant(**kw):
    d = dict(ch_type="Float64", nullable=False, primary_key=False, projection=False,
             codec="none", skip_index="none", compression="LZ4", status="ok",
             skip_reason=None, warnings=())
    d.update(kw)
    return TableVariant(**d)


@pytest.fixture(scope="module")
def small_dataset():
    return generate(DatasetSpec(rows=50_000, arrow_type="float64", value_range=(-1.0, 10.0),
                                decimal_places=8, null_count=0, seed=1, cardinality_ratio=0.5))


@pytest.fixture
def table(ch):
    """Yield a table-name factory; drop every table it handed out, even on failure."""
    names = []

    def make(suffix: str) -> str:
        name = table_name("t_loader", suffix)
        names.append(name)
        return name

    yield make
    for name in names:
        ch.command(f"DROP TABLE IF EXISTS {name}")


@pytest.mark.clickhouse
def test_load_produces_one_wide_part_with_per_column_stats(ch, table, small_dataset):
    name = table("aaaaaaaa00000000")
    r = load_table(ch, name, variant(), small_dataset)
    assert r.status == "ok", r.error
    assert r.error is None and r.insert_ms > 0
    s = r.stats
    assert s.rows == 50_000 and s.parts == 1
    assert s.uncompressed_bytes == 50_000 * 8          # v alone: Float64 is 8 bytes
    assert 0 < s.compressed_bytes and s.marks_bytes > 0
    assert storage_stats(ch, name) == s


@pytest.mark.clickhouse
def test_decimal32_inserts_through_the_arrow_path(ch, table, small_dataset):
    name = table("bbbbbbbb00000000")
    r = load_table(ch, name, variant(ch_type="Decimal32(8)", codec="Delta"), small_dataset)
    assert r.status == "ok", r.error
    assert r.stats.uncompressed_bytes == 50_000 * 4     # Decimal32 is 4 bytes


@pytest.mark.clickhouse
def test_decimal64_inserts_through_the_arrow_path(ch, table, small_dataset):
    name = table("bbbbbbbb00000001")
    r = load_table(ch, name, variant(ch_type="Decimal64(8)", codec="Delta"), small_dataset)
    assert r.status == "ok", r.error
    assert r.stats.uncompressed_bytes == 50_000 * 8     # Decimal64 is 8 bytes


@pytest.mark.clickhouse
def test_string_and_low_cardinality_string_load(ch, table, small_dataset):
    for ch_type, codec, suffix in (("String", "none", "cccccccc00000000"),
                                   ("LowCardinality(String)", "LowCardinality", "cccccccc00000001")):
        name = table(suffix)
        r = load_table(ch, name, variant(ch_type=ch_type, codec=codec), small_dataset)
        assert r.status == "ok", r.error
        assert r.stats.rows == 50_000


def table_exists(ch, name: str) -> bool:
    db, _, t = name.rpartition(".")
    rows = ch.query_rows(
        "SELECT count() FROM system.tables WHERE database = {db:String} AND name = {t:String}",
        parameters={"db": db, "t": t})
    return rows[0][0] == 1


@pytest.mark.clickhouse
def test_a_failing_cast_is_contained_and_the_table_dropped(ch, table, small_dataset):
    """A cast-boundary failure is contained as status="error": the safe Arrow
    cast rejects this dataset's fractional and negative values for UInt32
    client-side, before the server insert. The empty table it created is
    dropped so an errored variant never leaves a table in bench."""
    name = table("dddddddd00000000")
    r = load_table(ch, name, variant(ch_type="UInt32"), small_dataset)
    assert r.status == "error" and r.error
    assert r.error.startswith("ArrowInvalid")
    assert r.stats is None and r.insert_ms is None
    assert not table_exists(ch, name)


@pytest.mark.clickhouse
def test_a_failing_server_insert_is_recorded_not_raised(ch, table, small_dataset):
    """Accepted at DDL time is not proof the insert works (deferred-findings
    note). This one reaches the server: UUID has no Arrow mapping in the cast
    table, so the Float64 array is inserted uncast and the server rejects it
    (code 48, NOT_IMPLEMENTED). Recorded as status="error" naming the server
    exception, nothing raised, and the half-loaded table dropped."""
    name = table("dddddddd00000001")
    r = load_table(ch, name, variant(ch_type="UUID"), small_dataset)
    assert r.status == "error" and r.error
    assert r.error.startswith("DatabaseError") and "ClickHouse exception" in r.error
    assert r.stats is None and r.insert_ms is None
    assert not table_exists(ch, name)


@pytest.mark.clickhouse
def test_load_replaces_an_existing_table(ch, table, small_dataset):
    name = table("ffffffff00000000")
    ch.command(f"CREATE TABLE {name} (x String) ENGINE = MergeTree ORDER BY tuple()")
    r = load_table(ch, name, variant(), small_dataset)
    assert r.status == "ok", r.error
    assert r.stats.rows == 50_000


@pytest.mark.clickhouse
def test_storage_stats_use_the_per_column_fields(ch, table, small_dataset):
    """row_id and v must report DIFFERENT byte counts; the part-level columns
    would report the same total on both rows (FINDINGS s5)."""
    name = table("eeeeeeee00000000")
    load_table(ch, name, variant(ch_type="Decimal32(8)", codec="Delta"), small_dataset)
    rows = ch.query_rows(
        "SELECT column, sum(column_data_uncompressed_bytes) FROM system.parts_columns "
        "WHERE database = 'bench' AND table = {t:String} AND active GROUP BY column ORDER BY column",
        parameters={"t": name.split(".")[1]})
    by_col = dict(rows)
    assert by_col["row_id"] == 50_000 * 8 and by_col["v"] == 50_000 * 4
    assert storage_stats(ch, name).uncompressed_bytes == by_col["v"]


@pytest.mark.clickhouse
@pytest.mark.parametrize("skip_index,nonzero", [("minmax", True), ("set(100)", True),
                                                ("bloom_filter", True), ("none", False)])
def test_index_bytes_recorded(ch, table, small_dataset, skip_index, nonzero):
    name = table("c" + skip_index.replace("(", "").replace(")", "")[:6].ljust(7, "0") + "00000000")
    r = load_table(ch, name, variant(skip_index=skip_index), small_dataset)
    assert r.status == "ok", r.error
    s = r.stats
    if nonzero:
        assert s.index_compressed_bytes > 0 and s.index_uncompressed_bytes > 0
    else:
        assert (s.index_compressed_bytes, s.index_uncompressed_bytes, s.index_marks_bytes) == (0, 0, 0)


@pytest.mark.clickhouse
@pytest.mark.parametrize("skip_index", [
    "minmax:1", "minmax:2", "minmax:4", "minmax:8", "set(125):2", "set(125):8",
    "bloom_filter(0.001):1", "bloom_filter(0.01):4", "bloom_filter(0.025):2", "bloom_filter(0.05):8",
])
def test_granularity_and_fpr_tables_create_and_fill(ch, table, small_dataset, skip_index):
    digits = "".join(c for c in skip_index if c.isalnum())[:7].ljust(7, "0")
    name = table("g" + digits + "00000000")
    r = load_table(ch, name, variant(skip_index=skip_index), small_dataset)
    assert r.status == "ok", r.error
    assert r.stats.index_compressed_bytes > 0
    gran = int(skip_index.rsplit(":", 1)[1])
    got = ch.query_rows("SELECT granularity FROM system.data_skipping_indices "
                        "WHERE database = 'bench' AND table = {t:String}",
                        parameters={"t": name.split(".")[1]})
    assert [tuple(r) for r in got] == [(gran,)]
