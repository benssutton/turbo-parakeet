from chbench.harness import quiet
from tests.harness.fakes import FakeClient


def test_snapshot_collects_host_docker_power_and_server(monkeypatch):
    monkeypatch.setattr(quiet.psutil, "cpu_percent", lambda interval: 4.5)
    outputs = {
        "docker": "8 16624816128",
        "powercfg": "Power Scheme GUID: 381b4222-f694-41f0-9685-ff5bb260df2e  (Balanced)",
    }
    monkeypatch.setattr(quiet, "_run", lambda cmd: outputs[cmd[0]])
    monkeypatch.setattr(quiet.sys, "platform", "win32")
    snap = quiet.snapshot(FakeClient(), cpu_interval_s=0)
    assert snap == {
        "host_cpu_pct": 4.5,
        "docker_cpus": 8,
        "docker_mem_bytes": 16624816128,
        "power_scheme": "Balanced",
        "ch_load_average_1": 0.0,
        "ch_active_merges": 0,
    }


def test_snapshot_degrades_to_none_without_docker_or_powercfg(monkeypatch):
    monkeypatch.setattr(quiet.psutil, "cpu_percent", lambda interval: 1.0)
    monkeypatch.setattr(quiet, "_run", lambda cmd: None)
    monkeypatch.setattr(quiet.sys, "platform", "linux")
    snap = quiet.snapshot(FakeClient(), cpu_interval_s=0)
    assert snap["docker_cpus"] is None and snap["power_scheme"] is None


def test_warnings_name_a_busy_host_and_active_merges():
    assert quiet.warnings({"host_cpu_pct": 3.0, "ch_active_merges": 0}) == []
    w = quiet.warnings({"host_cpu_pct": 25.0, "ch_active_merges": 2})
    assert len(w) == 2 and "25%" in w[0] and "2 ClickHouse merge" in w[1]
