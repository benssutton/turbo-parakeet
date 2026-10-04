# Local SonarQube (Community Build)

## Setup

1. Start Docker Desktop, then `docker compose -f sonar/compose.yml up -d`; wait for
   http://localhost:9000 (log in `admin` / `admin`, set a new password).
2. My Account → Security → generate a **user token**; set it for your user, e.g.
   PowerShell: `[Environment]::SetEnvironmentVariable("SONARQUBE_TOKEN", "<token>", "User")`
   (restart the terminal / VS Code afterwards). Never commit the token.
3. Analyse: `python scripts/check.py sonar` (runs the Python tests with coverage (4 pytest-xdist workers, set in `pytest.ini`) and the
   Java tests with JaCoCo, then the `sonarsource/sonar-scanner-cli` container against
   `sonar-project.properties`; Python coverage comes from `coverage.xml`, Java coverage from
   `bindings/java/target/site/jacoco/jacoco.xml`; Rust coverage is not imported).
   The scanner container joins SonarQube's Docker network and talks to `sonarqube:9000`
   directly (not `localhost` / `host.docker.internal`), so the server must be running
   (`docker compose -f sonar/compose.yml up -d`); the step fails fast if it is not.

## Seeing the results

**Dashboard (browser):** http://localhost:9000/dashboard?id=turbo-parakeet — sign in, then:

| Tab / page | Shows |
|---|---|
| Overview | Quality gate (pass / fail), new vs overall code, coverage, duplications |
| Issues | Bugs, vulnerabilities and code smells; filter by language, file, severity or rule; each issue links to the rule explaining why and how to fix it |
| Security Hotspots | Code to review by hand (not counted as issues until reviewed) |
| Measures | Per-file and per-folder metrics, including lines of code per language |
| Code | Browse files with issues and coverage marked line by line |

Each `check.py sonar` run adds an analysis (the Overview keeps the history); the issues you
see are from the latest one. The browser links directly to a single issue.

**Command line:** the scanner log ends with `ANALYSIS SUCCESSFUL` and the dashboard URL.
That means the upload worked, not that the code is clean: open the dashboard for the findings.

**Claude Code:** `.mcp.json` registers the official `mcp/sonarqube` server (Docker image),
which reads `SONARQUBE_TOKEN` from your environment and talks to the local server. Approve
the project-scoped server when Claude Code prompts for it, then ask for example "list the
open SonarQube issues for turbo-parakeet" or "fix the bugs SonarQube found in
`services/analytics/src/…`". After Claude changes code, re-run `python scripts/check.py sonar`
and ask it to re-check.

## What is analysed

- **Python** (`services/analytics/analytics`, tests) with coverage from `coverage.xml`.
  The reported coverage is Python only.
- **Rust** (`services/analytics/src`) — this Community Build does analyse Rust (the first
  analysis counted about 15 000 lines), but it imports no Rust coverage. Clippy
  (`scripts/check.py lint`) remains the Rust gate; Sonar adds extra rules on top.
- **Java** (`services/analytics/bindings/java`).

Findings already reviewed and rejected (false positives, accepted complexity) are listed in
[false-positives.md](false-positives.md): check it before acting on a Sonar finding, and add
to it when you reject one.

Rules and exclusions: `sonar-project.properties`. The gate and rule set can be changed in
the web UI (Quality Gates, Quality Profiles).
