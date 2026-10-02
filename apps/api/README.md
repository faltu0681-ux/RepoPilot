# RepoPilot API

The FastAPI service: run state, the execution engine, and the event stream. The terminal UI is a
separate application in `apps/web`.

```bash
pip install -e ".[dev]"   # from the repository root
repopilot serve           # http://127.0.0.1:8000, docs at /docs
pytest                    # the suite lives here, in apps/api/tests
```

`repopilot serve` rather than a bare `uvicorn --reload`, because the run workspace and the
repositories directory both hold other people's `.py` files. Watching them would restart the server
in the middle of a run. Both are pruned from the watch, derived from the settings that place them.

## Packages

| Package | Responsibility |
| --- | --- |
| `domain` | Data only: `Run`, `RunStatus`, `ExecutionEvent`, `RepositoryRef`. Imports nothing from the project |
| `execution` | Engine, event bus, the step pipeline, and the sandbox the test step runs in |
| `analysis` | Deterministic measurement: profiles, file relevance, test-runner detection |
| `agent` | Model-driven reasoning: plan, implement, diagnose |
| `github` | GitHub metadata, clone, pull requests |
| `api` | HTTP routes and the WebSocket endpoint |

The dependency direction is one-way: `api` -> `execution` -> `agent`/`analysis` -> `domain`.

## Repositories

A `RepositoryRef` is either a GitHub `owner/name` or a `local://<folder>` under
`REPOPILOT_REPOSITORIES_DIR`. The scheme decides, so a GitHub owner literally named `local` is
still a GitHub repository.

A local name is one path segment, validated to be legal as a directory name on Windows as well as
POSIX, and re-checked for containment after resolving. A local run never writes to the source: it
copies the folder into `REPOPILOT_WORKSPACE_DIR/<local__folder>/<run_id>` and the pipeline works
on the copy. `.git` comes along; a folder that is not a checkout gets one initialised, because the
run creates a branch to work on.

`GET /api/runs/repositories` lists what is on disk, so the UI can offer `local://` references
instead of asking anyone to type them.

## Events

Every step emits into the event bus; the store persists the same events with a gap-free per-run
`sequence`. The WebSocket is replay-then-stream, so a client can attach to a run mid-flight and
still render it whole, and can resume from `after_sequence` after a reconnect without re-reading
what it already has. `apps/web/src/useRunStream.ts` is the reference client.

## Adding a step

Write one `Step` and add it to `default_pipeline()`. `StepOutcome.RETRY` plus `Run.attempts` is
the fix loop. Nothing else needs to change: the bus, the store, the socket, and the UI already
render whatever a step emits.