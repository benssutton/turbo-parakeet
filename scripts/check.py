"""Project checks: the same commands locally and in CI (.github/workflows/ci-cd.yml).

    python scripts/check.py fast        # formatting + lint (seconds)
    python scripts/check.py check       # fast + build the extension + Rust and Python tests
    python scripts/check.py ci          # check + the Java binding tests (what the pipeline runs)
    python scripts/check.py <step> ...  # any steps, in order (see STEPS)

Run it with the project's Python (conda env p312): the tool versions are pinned in
requirements-dev.txt and the Rust toolchain in rust-toolchain.toml.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CRATE = ROOT / "services" / "analytics"
JAVA = CRATE / "bindings" / "java"
PY = sys.executable
WINDOWS = os.name == "nt"
MVNW = str(JAVA / ("mvnw.cmd" if WINDOWS else "mvnw"))


def _env() -> dict[str, str]:
    """The Python the pyo3 crate links against, and (Windows) its DLL directory."""
    env = dict(os.environ, PYO3_PYTHON=PY)
    env["PATH"] = str(Path(PY).parent) + os.pathsep + env.get("PATH", "")
    return env


def _run(cmd: list[str], cwd: Path = ROOT, env: dict[str, str] | None = None) -> None:
    print(f"\n$ ({cwd.relative_to(ROOT) or '.'}) {' '.join(cmd)}", flush=True)
    done = subprocess.run(cmd, cwd=cwd, env=env or _env())
    if done.returncode:
        sys.exit(done.returncode)


def fmt_check() -> None:
    _run(["cargo", "fmt", "--all", "--", "--check"], CRATE)
    _run([PY, "-m", "black", "--check", "."])


def fmt() -> None:
    _run(["cargo", "fmt", "--all"], CRATE)
    _run([PY, "-m", "black", "."])
    _run([PY, "-m", "ruff", "check", "--fix", "."])


def lint() -> None:
    _run(
        ["cargo", "clippy", "--all-features", "--all-targets", "--", "-D", "warnings"],
        CRATE,
    )
    # The C ABI build (Java): no Python.
    _run(
        [
            "cargo",
            "clippy",
            "--no-default-features",
            "--all-targets",
            "--",
            "-D",
            "warnings",
        ],
        CRATE,
    )
    _run([PY, "-m", "ruff", "check", "."])


def build() -> None:
    _run([PY, "-m", "maturin", "develop", "--release"], CRATE)


def test_rust() -> None:
    _run(["cargo", "test", "--lib"], CRATE)


def test_py() -> None:
    _run([PY, "-m", "pytest", "-q"])


def build_capi() -> None:
    # Own target dir: maturin writes the Python-linked library to target/release.
    _run(
        [
            "cargo",
            "build",
            "--release",
            "--no-default-features",
            "--target-dir",
            "target/capi",
        ],
        CRATE,
    )


def test_java() -> None:
    build_capi()
    _run([MVNW, "-B", "test"], JAVA)


def coverage_rust() -> None:
    """What CI uploads to Codecov (needs `cargo install cargo-llvm-cov`)."""
    _run(["cargo", "llvm-cov", "clean", "--workspace"], CRATE)
    _run(
        [
            "cargo",
            "llvm-cov",
            "--all-features",
            "--workspace",
            "--lcov",
            "--output-path",
            "lcov.info",
        ],
        CRATE,
    )


COVERAGE_TARGET = CRATE / "target" / "llvm-cov-target"


def _llvm_cov_env() -> dict[str, str]:
    """cargo-llvm-cov's environment (its `show-env`) for builds and test runs made outside
    `cargo llvm-cov`. Its wrapper instruments the crate without changing cargo's
    fingerprint, so a dedicated target directory keeps instrumented and normal artifacts
    apart (otherwise cargo reuses the uninstrumented ones)."""
    shown = subprocess.run(
        ["cargo", "llvm-cov", "show-env"],
        cwd=CRATE,
        env=_env(),
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    env = _env()
    for line in shown.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            env[key] = value.strip("'")
    target = COVERAGE_TARGET.as_posix()
    env["CARGO_TARGET_DIR"] = target
    env["CARGO_LLVM_COV_TARGET_DIR"] = target
    env["CARGO_LLVM_COV_BUILD_DIR"] = target
    env["LLVM_PROFILE_FILE"] = f"{target}/analytics-%p-%4m.profraw"
    return env


def coverage_rust_full() -> None:
    """Rust coverage from the unit tests AND the Python tests, merged (what CI uploads to
    Codecov; needs cargo-llvm-cov, and also writes coverage.xml for Python). The pytest run drives an instrumented build of the
    extension, which covers python.rs and the rest of what only Python reaches. The
    normal extension in the working tree is put back afterwards."""
    env = _llvm_cov_env()
    for stale in COVERAGE_TARGET.glob("*.profraw"):
        stale.unlink()
    ext = [
        f
        for pattern in ("analytics.*pyd", "analytics*.so")
        for f in (CRATE / "analytics").glob(pattern)
    ]
    backup = Path(tempfile.mkdtemp(prefix="analytics-ext-"))
    for f in ext:
        shutil.copy2(f, backup / f.name)
    try:
        _run(["cargo", "test", "--lib", "--profile", "coverage"], CRATE, env)
        _run([PY, "-m", "maturin", "develop", "--profile", "coverage"], CRATE, env)
        _run([PY, "-m", "pytest", "-q", "--cov=.", "--cov-report=xml"], ROOT, env)
    finally:
        # The instrumented extension would write profile data on every later Python run.
        for f in ext:
            shutil.copy2(backup / f.name, f)
        shutil.rmtree(backup, ignore_errors=True)
    report = ["cargo", "llvm-cov", "report", "--profile", "coverage"]
    _run([*report, "--lcov", "--output-path", "lcov.info"], CRATE, env)
    _run([*report, "--html", "--output-dir", "coverage-html"], CRATE, env)
    _run(report, CRATE, env)


def coverage_py() -> None:
    _run([PY, "-m", "pytest", "--cov=.", "--cov-report=xml", "--cov-report=html", "-q"])


def _sonar_network() -> str:
    """The Docker network of the running local SonarQube (sonar/compose.yml). The scanner
    joins it and talks to `sonarqube:9000` directly: nothing on the Windows host (or a
    flaky `host.docker.internal` route to it) is in the path."""
    compose = ["docker", "compose", "-f", str(ROOT / "sonar" / "compose.yml")]
    found = subprocess.run(
        [*compose, "ps", "-q", "sonarqube"], capture_output=True, text=True
    )
    container = found.stdout.strip()
    if not container:
        sys.exit("SonarQube is not running: docker compose -f sonar/compose.yml up -d")
    nets = subprocess.run(
        [
            "docker",
            "inspect",
            "-f",
            "{{range $n, $_ := .NetworkSettings.Networks}}{{$n}} {{end}}",
            container,
        ],
        capture_output=True,
        text=True,
    ).stdout.split()
    if not nets:
        sys.exit(f"cannot find the Docker network of container {container}")
    return nets[0]


def sonar() -> None:
    """Analyse with the local SonarQube (sonar/README.md): coverage first, then the scanner."""
    token = os.environ.get("SONARQUBE_TOKEN")
    if not token:
        sys.exit("SONARQUBE_TOKEN is not set (see sonar/README.md)")
    network = _sonar_network()  # fail fast, before the coverage run
    coverage_py()
    # The scanner reads SONAR_TOKEN; `-e SONAR_TOKEN` forwards it from this environment
    # (keeps the token off the command line).
    os.environ["SONAR_TOKEN"] = token
    _run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            network,
            "-e",
            "SONAR_HOST_URL=http://sonarqube:9000",
            "-e",
            "SONAR_TOKEN",
            "-v",
            f"{ROOT}:/usr/src",
            "sonarsource/sonar-scanner-cli",
        ]
    )


STEPS = {
    "fmt": fmt,  # rewrites files
    "fmt-check": fmt_check,
    "lint": lint,
    "build": build,
    "test-rust": test_rust,
    "test-py": test_py,
    "test-java": test_java,
    "coverage-rust": coverage_rust,
    "coverage-rust-full": coverage_rust_full,  # unit + pytest, merged
    "coverage-py": coverage_py,
    "sonar": sonar,  # needs the local SonarQube running
}
GROUPS = {
    "fast": ["fmt-check", "lint"],
    "check": ["fmt-check", "lint", "build", "test-rust", "test-py"],
    "ci": ["fmt-check", "lint", "build", "test-rust", "test-py", "test-java"],
}


def main(argv: list[str]) -> None:
    names = [s for a in argv for s in GROUPS.get(a, [a])]
    unknown = [n for n in names if n not in STEPS]
    if not names or unknown:
        print(__doc__)
        print("steps: ", ", ".join(STEPS))
        print("groups:", ", ".join(GROUPS))
        sys.exit(2 if unknown else 0)
    for n in names:
        STEPS[n]()
    print("\nall checks passed:", " ".join(names))


if __name__ == "__main__":
    main(sys.argv[1:])
