# Local SonarQube (Community Build)

1. Start Docker Desktop, then `docker compose -f sonar/compose.yml up -d`; wait for
   http://localhost:9000 (log in `admin` / `admin`, set a new password).
2. My Account → Security → generate a **user token**; set it for your user, e.g.
   PowerShell: `[Environment]::SetEnvironmentVariable("SONARQUBE_TOKEN", "<token>", "User")`
   (restart the terminal / VS Code afterwards). Never commit the token.
3. Analyse: `python scripts/check.py sonar` (runs the Python tests with coverage, then the
   `sonarsource/sonar-scanner-cli` container against `sonar-project.properties`).
   Run `python scripts/check.py test-java` first if you want Java coverage / bytecode.
4. Claude Code: `.mcp.json` registers the official `mcp/sonarqube` server (Docker image),
   which reads `SONARQUBE_TOKEN` from your environment and talks to the local server, so
   Claude can list and explain issues, quality-gate status and hotspots. Approve the
   project-scoped server when Claude Code prompts for it.

Community Build analyses Python and Java here; whether it analyses Rust depends on the
release, so clippy (`scripts/check.py lint`) remains the Rust gate.
