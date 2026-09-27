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

| Milestone | What it adds | Status |
| --- | --- | --- |
| 1 | Project skeleton, packaging, secrets rules | done |
| 2 | FastAPI app, config, `/health` | done |
| 3 | Run state management (`Run`, statuses, in-memory store) | done |
| 4 | Execution engine: event bus, step pipeline, real GitHub + clone + inspect steps | done |
| 5 | WebSocket live event streaming | done |
| 6 | React + Vite terminal UI | done |

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

**`.env` is gitignored and must never be committed.** No tokens, keys, or credentials belong
in this repository. If you add a secret-handling path later, read it from the environment only.

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
  github/          GitHub client (repo metadata, clone, PR creation)
  api/             HTTP routes and the WebSocket endpoint
frontend/
  src/types.ts     Wire types mirroring the backend frames
  src/api.ts       REST client
  src/useRunStream.ts  WebSocket subscription: cursor, resync, reconnect
  src/App.tsx      Form and live terminal
tests/             pytest suite
```

The dependency direction is one-way: `api` -> `execution` -> `domain`. `domain` imports
nothing from the project. That is what keeps the agent loop addable later without refactoring.

## License

MIT
