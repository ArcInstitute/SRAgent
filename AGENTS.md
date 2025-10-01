# Repository Guidelines

## Project Structure & Module Organization
- `SRAgent/`: Python package (CLI in `cli/`, agent logic in `agents/`, tools in `tools/`, workflows in `workflows/`, DB helpers in `db/`).
- `tests/`: Pytest suite mirroring package layout (e.g., `tests/agents/`, `tests/workflows/`).
- `scripts/`: Standalone utilities and evaluation helpers.
- `assets/`: Images used in docs.
- Root files: `pyproject.toml` (build), `Dockerfile`, `README.md`, `.env` (optional), `.github/workflows/` (CI).

## Build, Test, and Development Commands
- Install (user): `pip install .`
- Install (dev): `pip install -e ".[dev]"`
- Run tests: `pytest tests/`
- CLI help: `SRAgent --help`
- Example run: `SRAgent entrez "Convert GSE121737 to SRX accessions"`
- Docker (optional): `docker build -t sragent .` then `docker run --rm -e OPENAI_API_KEY -e ANTHROPIC_API_KEY sragent --help`

## Coding Style & Naming Conventions
- Python ≥ 3.11; follow PEP 8 with 4‑space indents and descriptive names.
- Names: modules/functions `snake_case`, classes `CamelCase`, constants `UPPER_SNAKE_CASE`.
- Prefer type hints and docstrings on public functions; keep functions small and focused.
- No repo‑enforced linter; keep style consistent with nearby code and avoid introducing new tooling/config without discussion.

## Testing Guidelines
- Framework: pytest. Place tests under `tests/` with files named `test_*.py` and test functions `test_*`.
- Mirror CLI and workflow behavior; see `tests/test_cli_help.py` for a pattern to validate subcommand help and exit codes.
- Add tests for new features and edge cases; avoid network‑dependent tests unless mocked.

## Commit & Pull Request Guidelines
- Commits: concise, imperative subject; include context in the body when changing behavior or interfaces (e.g., “Add disease‑ontology agent”).
- PRs: clear description, rationale, and usage notes; link issues when applicable. Include example CLI invocations or sample output for user‑facing changes.
- CI: GitHub Actions runs `pytest` on Python 3.11/3.12 for `main` and `dev`. PRs must be green before merge.

## Security & Configuration Tips
- Secrets via env: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `EMAIL`, `NCBI_API_KEY`, optional `DYNACONF` and `DYNACONF_SETTINGS_PATH`. `.env` is loaded via `python-dotenv`.
- Models and defaults live in `SRAgent/settings.yml`. Do not commit real credentials; use GitHub secrets or local `.env`.
