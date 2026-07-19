# Repository File Policy

The repository separates immutable inputs, maintained code, durable evidence, and local-only state.

| Category | Paths | Policy |
| --- | --- | --- |
| Source code | `src/dutch_sentiment/models/`, `src/dutch_sentiment/experiments/`, serving modules, `tests/`, `scripts/` | Review, lint, test, and version in Git |
| Configuration | `configs/models/`, `configs/training.yaml`, `pyproject.toml`, `Makefile`, `Dockerfile` | One canonical file per model family; version in Git; no secrets |
| Immutable inputs | `Python_Engineer_Challenge_2.csv`, challenge PDF | Never rewrite; verify by hash |
| Production artifacts | `artifacts/model.joblib`, `artifacts/model_metadata.json` | Preserve and verify before serving |
| Durable evidence | selected `artifacts/*.json|csv`, `reports/*.md`, final presentation/PDF | Version only when it supports a documented decision |
| Reproducible outputs | coverage, render PNGs, inspection NDJSON, build output | Ignore and regenerate |
| Local environment | `.venv/`, `.cache/`, `mlruns/`, `mlflow.db` | Ignore; back up MLflow separately when needed |
| Sensitive files | `.secrets/` | Ignore; never inspect, package, or commit |

Presentation source files are content artifacts and are not ignored. Their generated `*.inspect.ndjson`
files and slide-render directories are ignored. Model downloads and embedding matrices remain under
`.cache/`; they are retained locally but excluded from Git and Docker contexts.

`main` is the only long-lived branch. Completed experiment branches receive a remote `archive/*` tag
after their Git SHA is mapped to an MLflow evidence run; the branch can then be removed without losing
recoverability.
