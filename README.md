# RepoPilot

An autonomous coding agent. Point it at a GitHub repository and a task; RepoPilot clones the
repo, analyses it, plans the change, applies the change, runs the tests, diagnoses failures,
retries, and opens a pull request.

RepoPilot streams everything it does as live events, so you watch the agent work in a terminal.

```
GitHub repo -> Create Run -> Execution state -> Live events -> WebSocket -> Terminal UI
```

## Status

**Phase 2A - issue to plan.** A run now reads a real repository and produces a real, grounded
implementation plan for a natural-language task. It still does not edit anything: writing code
(2B) and running the project's tests (2C) are not implemented yet.

    validate -> resolve -> clone -> inspect -> profile -> select_files -> plan

| Milestone | What it adds | Status |
| --- | --- | --- |
| 1 | Project skeleton, packaging, secrets rules | done |
| 2 | FastAPI app, config, `/health` | done |
| 3 | Run state management (`Run`, statuses, in-memory store) | done |
| 4 | Execution engine: event bus, step pipeline, real GitHub + clone + inspect steps | done |
| 5 | WebSocket live event streaming | done |
| 6 | React + Vite terminal UI | done |
| 7 | Isolated per-run workspace | done |
| 8 | Real repository profile and file measurement | done |
| 9 | Task-to-file relevance ranking, no model involved | done |
| 10 | Pluggable LLM client and structured implementation plan | done |
| 11 | Plan and discovered files in the terminal UI | done |
| 12 | Edit files on a branch in the run workspace (2B) | next |
| 13 | Test detection, execution, failure diagnosis, retry (2C) | planned |
| 14 | Structured run summary with VERIFIED / FAILED (2D) | planned |

Every seam the rest of the agent needs already exists: `StepOutcome.RETRY` plus `Run.attempts`
is the fix loop, `RunServices` is where the sandbox and diff writer go, and each agent
capability is one more entry in `default_pipeline()`.

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
repopilot serve
```

The API is then on <http://localhost:8000>, with docs at <http://localhost:8000/docs>.

Use this rather than a bare `uvicorn --reload`. The run workspace defaults to
`./.repopilot-data`, inside the project, and a run clones a whole repository into it. A bare
`--reload` watches the project, sees those cloned `.py` files, and restarts the server in the
middle of the run that is writing them, which cancels the run and loses its events.
`repopilot serve` derives the exclusion from `REPOPILOT_WORKSPACE_DIR` and prunes that directory
from the watch, so reload works and runs survive. `repopilot serve --no-reload` runs a single
process.

Run the tests:

```bash
pytest
```

## Trying a run

Give it a public repository and a task in plain English:

```bash
curl -X POST http://127.0.0.1:8000/api/runs \
  -H 'Content-Type: application/json' \
  -d '{"repository": "pallets/flask", "task": "Add a /health endpoint that reports the service uptime"}'
```

A run with no model configured gets as far as the shortlist and then fails, naming the settings
it needs. Configure [the reasoning backend](#configuring-the-reasoning-backend) first and the
same request produces a plan.

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
| `WS /ws/events` | Live event stream |

## Watching a run live

Polling works, but the stream is the point:

```bash
pip install websockets
```

```python
import asyncio, json, websockets


async def main():
    async with websockets.connect("ws://127.0.0.1:8000/ws/events?run_id=<run_id>") as ws:
        async for raw in ws:
            frame = json.loads(raw)
            if frame["type"] == "event":
                e = frame["event"]
                print(f"{e['sequence']:>3} {e['type']:<15} {e['message']}")
            elif frame["type"] == "bye":
                break


asyncio.run(main())
```

Connecting is **replay-then-stream**: the socket sends stored history first, then live events
follow. That is what lets a client attach to a run that is already half done and still render it
whole.

| Frame | Meaning |
| --- | --- |
| `hello` | Sent first. Protocol version, the run subscribed to, the sequence it resumed after |
| `event` | One `ExecutionEvent`, exactly as stored, with its per-run `sequence` |
| `status` | A run was created or changed status |
| `resync` | The subscriber fell behind; refetch `/events?after_sequence=` from `from_sequence` |
| `error` | The stream could not be established |
| `bye` | The stream is closing, then the socket disconnects |

| Parameter | Default | Purpose |
| --- | --- | --- |
| `run_id` | _none_ | Follow one run. Omit to follow every run |
| `after_sequence` | `0` | Resume after this cursor, so reconnects do not replay what you already have |

A run-scoped socket closes on its own once the run reaches a terminal status. An unscoped socket
stays open. Each subscriber has a bounded queue, so a client that cannot keep up gets a `resync`
frame rather than unbounded memory growth.

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
| `REPOPILOT_LLM_API_KEY` | _unset_ | Key for the reasoning backend. Not needed for local runtimes |
| `REPOPILOT_LLM_BASE_URL` | _empty_ | Provider endpoint. Empty means "no reasoning backend" |
| `REPOPILOT_LLM_MODEL` | _empty_ | Model name. Empty means "no reasoning backend" |
| `REPOPILOT_LLM_DIALECT` | `openai` | `openai` for any OpenAI-compatible API, or `anthropic` |

**`.env` is gitignored and must never be committed.** No tokens, keys, or credentials belong
in this repository. If you add a secret-handling path later, read it from the environment only.

### Configuring the reasoning backend

Phase 2A needs a model to write the implementation plan. `REPOPILOT_LLM_MODEL` and
`REPOPILOT_LLM_BASE_URL` both default to empty, which means *unconfigured*. That is on purpose:
with no default endpoint, a fresh install never builds a client and never makes a network call.
A run that needs reasoning fails with a message naming these settings.

Copy `.env.example` to `.env` and fill in whichever provider you have:

```bash
# OpenAI
REPOPILOT_LLM_API_KEY=sk-...
REPOPILOT_LLM_BASE_URL=https://api.openai.com/v1
REPOPILOT_LLM_MODEL=gpt-4o-mini

# Anthropic
REPOPILOT_LLM_API_KEY=sk-ant-...
REPOPILOT_LLM_BASE_URL=https://api.anthropic.com
REPOPILOT_LLM_MODEL=claude-sonnet-4-5
REPOPILOT_LLM_DIALECT=anthropic

# Ollama, locally, no key required
REPOPILOT_LLM_BASE_URL=http://127.0.0.1:11434/v1
REPOPILOT_LLM_MODEL=llama3.1
```

### The run workspace

Every run gets its own checkout, named after the run:

```
.repopilot-data/<owner>__<repo>/<run_id>/
```

Two runs of the same repository therefore cannot see each other's work, and a run never touches
a checkout you made yourself. Nothing outside `REPOPILOT_WORKSPACE_DIR` is written, and file
reads are checked for containment before use.

## Terminal UI

The Vite dev server runs on <http://localhost:5173> and proxies both `/api` and `/ws` to the
backend, so the browser talks to one origin and there is no CORS preflight.

```bash
cd frontend
npm install
npm run dev
```

Start the backend first, then open the page, enter a repository and a task, and press **Start run**.
Events stream in live and the panel follows the tail until you scroll up, at which point it stops
chasing so you can read.

Below the terminal, two panels fill in as the run discovers things, and both are derived purely
from the event stream, so a reconnecting browser rebuilds them from the replayed events alone:

- **Discovered files** appears as the `select_files` step streams its `file.selected` events, one
  row per file with its score, language, line count, and the reasons it was chosen.
- **Implementation plan** appears on `plan.created`, with the summary, approach, numbered steps,
  the files to change tagged `modify` / `add` / `read`, risks, and token usage.

Neither panel fetches anything the backend did not already report, so what you read here is
exactly what the run did.

```bash
npm run lint      # oxlint
npm run build     # tsc -b && vite build
```

`REPOPILOT_BACKEND` overrides the proxy target if the backend is not on port 8000.

### How the client uses the stream

`frontend/src/useRunStream.ts` holds the three things that make an at-any-time attach work:

- the cursor advances with the highest sequence seen and is sent as `after_sequence` on every
  reconnect, so a flaky connection never re-renders what is already on screen;
- a `resync` frame triggers a REST refetch from the sequence the server names, so a slow client
  ends up with a complete stream rather than a hole;
- events are deduped by sequence on the way in, so a duplicate delivery is a no-op.

Reconnection backs off exponentially to 10s. A `bye` is treated as final, because a run-scoped
socket only says goodbye once the run is terminal.

The panel is a styled event log rather than an `xterm.js` terminal, because Phase 1 renders
structured run events rather than raw process output. When the agent starts shelling out, that is
the point to swap in a real terminal.

## Architecture

```
backend/repopilot/
  main.py          FastAPI app factory, lifespan, router wiring
  config.py        Environment-based settings
  domain/          Pure data: Run, RunStatus, ExecutionEvent, RunEventType
  execution/       Engine, event bus, step pipeline (the pluggable agent seam)
  analysis/        Deterministic repo inspection: profile.py, relevance.py. No model.
  agent/           Model-driven reasoning: llm.py, prompts.py, plan.py, schema.py
  github/          GitHub client (repo metadata, clone, PR creation)
  api/             HTTP routes and the WebSocket endpoint
frontend/
  src/types.ts     Wire types mirroring the backend frames
  src/api.ts       REST client
  src/useRunStream.ts  WebSocket subscription: cursor, resync, reconnect
  src/App.tsx      Form, live terminal, discovered files, plan panel
tests/             pytest suite
```

The dependency direction is one-way: `api` -> `execution` -> `agent`/`analysis` -> `domain`.
`domain` imports nothing from the project. That is what keeps the rest of the agent addable
without refactoring.

### Analysis versus reasoning

The split is deliberate and load-bearing.

`analysis/` measures. It walks the checkout and reports what is really there: languages, line
counts, manifests, test configuration, which files are tests. `analysis/relevance.py` ranks
those files against the task using path, filename, and content signals. No model is involved,
so the expensive half of a run is reproducible, testable, and free.

`agent/` reasons. It is shown that measured inventory and the shortlisted files, with the top
candidates' real contents quoted in, and it is asked for one JSON object. The reply is
validated against `agent/schema.py`, retried once if it will not parse, and every path it
references is checked against files that actually exist. Paths with role `add` are kept, because
a plan is allowed to propose new files; anything else that is missing is dropped and recorded in
the plan's risks. A model that invents `src/app.py` cannot steer the run towards it.

## License

MIT
