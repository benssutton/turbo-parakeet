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
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CRATE = ROOT / "services" / "analytics"
JAVA = CRATE / "bindings" / "java"
PY = sys.executable
# pytest-xdist workers for the Python tests (locally, in CI and for the Sonar coverage run).
PYTEST_WORKERS = "4"
WINDOWS = os.name == "nt"
MVNW = str(JAVA / ("mvnw.cmd" if WINDOWS else "mvnw"))


def _env() -> dict[str, str]:
    """The Python the pyo3 crate links against, and (Windows) its DLL directory."""
    env = dict(os.environ, PYO3_PYTHON=PY)
    env["PATH"] = str(Path(PY).parent) + os.pathsep + env.get("PATH", "")
    return env


def _run(cmd: list[str], cwd: Path = ROOT) -> None:
    print(f"\n$ ({cwd.relative_to(ROOT) or '.'}) {' '.join(cmd)}", flush=True)
    done = subprocess.run(cmd, cwd=cwd, env=_env())
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
    _run([PY, "-m", "pytest", "-q", "-n", PYTEST_WORKERS])


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


def coverage_py() -> None:
    _run(
        [
            PY,
            "-m",
            "pytest",
            "--cov=.",
            "--cov-report=xml",
            "--cov-report=html",
            "-q",
            "-n",
            PYTEST_WORKERS,
        ]
    )


def sonar() -> None:
    """Analyse with the local SonarQube (sonar/README.md): coverage first, then the scanner."""
    token = os.environ.get("SONARQUBE_TOKEN")
    if not token:
        sys.exit("SONARQUBE_TOKEN is not set (see sonar/README.md)")
    coverage_py()
    # The scanner reads SONAR_TOKEN; `-e SONAR_TOKEN` forwards it from this environment
    # (keeps the token off the command line).
    os.environ["SONAR_TOKEN"] = token
    _run(
        [
            "docker",
            "run",
            "--rm",
            "-e",
            "SONAR_HOST_URL=http://host.docker.internal:9000",
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
