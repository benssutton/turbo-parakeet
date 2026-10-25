import uuid

import pytest

from chbench.harness.client import CHClient, Connection


def test_connection_defaults_match_the_container(monkeypatch):
    for k in ("CH_HOST", "CH_PORT", "CH_USER", "CH_PASSWORD", "CH_DATABASE"):
        monkeypatch.delenv(k, raising=False)
    c = Connection.from_env()
    assert (c.host, c.port, c.user, c.password, c.database) == (
        "localhost", 8123, "CLICKHOUSE_USER", "CLICKHOUSE_PASSWORD", "bench")


def test_connection_reads_env(monkeypatch):
    monkeypatch.setenv("CH_PORT", "9000")
    monkeypatch.setenv("CH_DATABASE", "other")
    c = Connection.from_env()
    assert c.port == 9000 and c.database == "other"


@pytest.mark.clickhouse
def test_version_is_a_dotted_string(ch):
    assert ch.version().count(".") == 3


@pytest.mark.clickhouse
def test_query_id_reaches_the_query_log(ch):
    """The clickhouse-connect 1.8.0 gotcha from FINDINGS s5: only the settings
    mapping delivers query_id. If this fails, every sample lookup will be empty."""
    qid = str(uuid.uuid4())
    ch.query_rows("SELECT 1", query_id=qid)
    ch.flush_logs()
    rows = ch.query_rows(
        "SELECT count() FROM system.query_log WHERE query_id = {q:String} AND type = 'QueryFinish'",
        parameters={"q": qid})
    assert rows[0][0] == 1


@pytest.mark.clickhouse
def test_query_id_survives_caller_settings(ch):
    """query_id is merged into a caller-supplied settings mapping, not dropped by it."""
    qid = str(uuid.uuid4())
    ch.query_rows("SELECT 1", query_id=qid, settings={"use_query_cache": 0})
    ch.flush_logs()
    rows = ch.query_rows(
        "SELECT count() FROM system.query_log WHERE query_id = {q:String} AND type = 'QueryFinish'",
        parameters={"q": qid})
    assert rows[0][0] == 1


@pytest.mark.clickhouse
def test_drop_caches_reports_each_statement(ch):
    ok = ch.drop_caches()
    assert set(ok) == {"MARK", "UNCOMPRESSED", "QUERY CONDITION"}
