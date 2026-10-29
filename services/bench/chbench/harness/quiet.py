"""How quiet the machine is around a run: host CPU, the Docker VM's resources, the power scheme
and the server's own load, recorded on the run row before the first round and after the last.
`warnings` names the two conditions known to disturb timings; it never refuses a run."""

from __future__ import annotations

import re
import subprocess
import sys

import psutil

HOST_CPU_WARN_PCT = 10.0
_POWER_RE = re.compile(r"\(([^)]*)\)\s*$")


def _run(cmd: list[str]) -> str | None:
    try:
        done = subprocess.run(
            cmd, capture_output=True, text=True, timeout=10, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip()


def _docker() -> dict:
    out = _run(["docker", "info", "--format", "{{.NCPU}} {{.MemTotal}}"])
    try:
        cpus, mem = out.split()
        return {"docker_cpus": int(cpus), "docker_mem_bytes": int(mem)}
    except (AttributeError, ValueError):
        return {"docker_cpus": None, "docker_mem_bytes": None}


def _power_scheme() -> str | None:
    if sys.platform != "win32":
        return None
    out = _run(["powercfg", "/getactivescheme"])
    m = _POWER_RE.search(out or "")
    return m.group(1) if m else out


def _server(client) -> dict:
    try:
        load = client.query_rows(
            "SELECT value FROM system.asynchronous_metrics WHERE metric = 'LoadAverage1'"
        )
        merges = client.query_rows("SELECT count() FROM system.merges")
    except Exception:  # noqa: BLE001 - a probe, never a reason to stop
        return {"ch_load_average_1": None, "ch_active_merges": None}
    return {
        "ch_load_average_1": float(load[0][0]) if load else None,
        "ch_active_merges": int(merges[0][0]),
    }


def snapshot(client, *, cpu_interval_s: float = 3.0) -> dict:
    return {
        "host_cpu_pct": float(psutil.cpu_percent(interval=cpu_interval_s)),
        **_docker(),
        "power_scheme": _power_scheme(),
        **_server(client),
    }


def warnings(snap: dict) -> list[str]:
    out = []
    cpu = snap.get("host_cpu_pct")
    if cpu is not None and cpu > HOST_CPU_WARN_PCT:
        out.append(
            f"host CPU {cpu:.0f}% (over {HOST_CPU_WARN_PCT:.0f}%): timings will be noisy"
        )
    merges = snap.get("ch_active_merges")
    if merges:
        out.append(f"{merges} ClickHouse merge(s) active")
    return out
