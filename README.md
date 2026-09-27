# RepoPilot

An autonomous coding agent. Point it at a GitHub repository and a task; RepoPilot clones the
repo, analyses it, plans the change, applies the change, runs the tests, diagnoses failures,
retries, and opens a pull request.

RepoPilot streams everything it does as live events, so you watch the agent work in a terminal.

```
GitHub repo -> Create Run -> Execution state -> Live events -> WebSocket -> Terminal UI
```

## Status

**Phase 1 — vertical slice.** The agent loop is not implemented yet. What works today is the
spine: create a run, give it real execution state, drive a real step pipeline that talks to
GitHub and clones the repository, stream every observation as a live event over a WebSocket,
and render them in a terminal UI.

The pipeline really does work, it just does not write code yet:

    validate -> resolve -> clone -> inspect

| Milestone | What it adds |
| --- | --- |
| 1 | Project skeleton, packaging, secrets rules |
| 2 | FastAPI app, config, `/health` |
| 3 | Run state management (`Run`, statuses, in-memory store) |
| 4 | Execution engine: event bus, step pipeline, real GitHub + clone + inspect steps |
| 5 | WebSocket live event streaming |
| 6 | React + Vite terminal UI |

Every seam the agent needs later already exists: repository analysis is `InspectRepositoryStep`,
`StepOutcome.RETRY` plus `Run.attempts` is the fix loop, `RunServices` is where the LLM and
sandbox clients go, and each agent capability is one more entry in `default_pipeline()`.

## Requirements

- Python 3.11+
- Node.js 20+ (for the terminal UI)

## Getting started

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

pip install -e ".[dev]"
```

Run the backend:

```bash
uvicorn repopilot.main:app --reload --port 8000
```

The API is then on <http://localhost:8000>, with docs at <http://localhost:8000/docs>.

Run the tests:

```bash
pytest
```

## Trying a run

```bash
curl -X POST http://127.0.0.1:8000/api/runs \
  -H 'Content-Type: application/json' \
  -d '{"repository": "octocat/Hello-World", "task": "Add a /health endpoint"}'
```

The call returns immediately with a run id. Watch the work:

```bash
curl http://127.0.0.1:8000/api/runs/<run_id>
curl http://127.0.0.1:8000/api/runs/<run_id>/events
```

| Endpoint | Purpose |
| --- | --- |
| `POST /api/runs` | Create a run and start the pipeline |
| `GET /api/runs` | List runs, newest first (`?status=`, `?limit=`) |
| `GET /api/runs/{id}` | Current state of one run |
| `GET /api/runs/{id}/events` | Full event history, with an `?after_sequence=` cursor |
| `DELETE /api/runs/{id}` | Cancel an in-flight run |

## Configuration

All configuration is environment-based and read from the process environment or a `.env` file
in the project root. See `.env.example`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `REPOPILOT_HOST` | `127.0.0.1` | Bind address for the API |
| `REPOPILOT_PORT` | `8000` | Bind port for the API |
| `REPOPILOT_GITHUB_TOKEN` | _unset_ | Token used to clone private repos and open PRs |
| `REPOPILOT_WORKSPACE_DIR` | `./.repopilot-data` | Where repositories are cloned |
| `REPOPILOT_CORS_ORIGINS` | `http://localhost:5173` | Comma-separated allowed frontend origins |

**`.env` is gitignored and must never be committed.** No tokens, keys, or credentials belong
in this repository. If you add a secret-handling path later, read it from the environment only.

## Terminal UI

```bash
cd frontend
npm install
npm run dev
```

The Vite dev server runs on <http://localhost:5173> and proxies API and WebSocket traffic to
the backend on port 8000.

For a production-style check, build the frontend and serve it from the backend:

```bash
cd frontend
npm run build
```

## Architecture

```
backend/repopilot/
  main.py          FastAPI app factory, lifespan, router wiring
  config.py        Environment-based settings
  domain/          Pure data: Run, RunStatus, ExecutionEvent, RunEventType
  execution/       Engine, event bus, step pipeline (the pluggable agent seam)
  github/          GitHub client (repo metadata, clone, PR creation)
  api/             HTTP routes and the WebSocket endpoint
frontend/src/      React terminal UI
tests/             pytest suite
```

The dependency direction is one-way: `api` -> `execution` -> `domain`. `domain` imports
nothing from the project. That is what keeps the agent loop addable later without refactoring.

## License

MIT
