# RepoPilot

An autonomous coding agent. Point it at a repository and a task; RepoPilot clones the repo,
analyses it, plans the change, applies the change, runs the tests, diagnoses failures, retries,
and opens a pull request.

RepoPilot streams everything it does as live events, so you watch the agent work in a terminal.

```
GitHub repo -> Create Run -> Execution state -> Live events -> WebSocket -> Terminal UI
```

A repository can be one on GitHub (`owner/name`) or one already on this machine
(`local://your-folder`). A local run works on its own copy, so your files are untouched afterwards.

## Status

**Phase 2C - proving the change.** A run now finishes its work by running the repository's own
test suite and reading the result. It detects the runner from the files the repo actually
declares, executes it in the run's checkout, reports the verdict live, and when tests fail it
asks the model what broke and repairs up to two times, re-running the suite after each repair.
A failure it cannot explain ends the run with the reason instead of a silent retry loop.

**Phase 2D - reporting on the run.** A run now commits its own change, proposes it, and says
plainly what it proved. The commit is made in the run's own clone after the suite has judged it,
with a message that names the task and records the verdict in the body. A GitHub repository with a
token configured gets a real push and a real pull request. Every run ends with a structured
summary on `Run.summary`, carrying one of three verdicts: `verified` (a suite passed), `failed`
(a suite ran and did not pass), or `unverified` (no suite ran, so nothing is claimed either way).

A run that cannot commit, cannot push, or has no token says which of those it was rather than
failing or pretending. A `local://` folder is the user's own code on this machine, so it is
committed in the run's private copy and no pull request is attempted.

    validate -> resolve -> clone -> inspect -> profile -> select_files -> plan -> apply_plan
    -> test -> diagnose -> commit -> pull_request -> summarise

**V1 - executing untrusted code behind a boundary.** Everything above runs the target repository's
test suite, which is the first thing in this project that executes code nobody reviewed. That used
to happen in a child process on the host, with a working directory and a filtered environment and
nothing else. There is now an `ExecutionEnvironment` interface with two backends behind it: the
existing `local` one, and a `docker` one that runs each command in a throwaway container with only
the run's own checkout mounted, no network, and CPU, memory and process caps. The engine does not
know which is in use, and every run announces which one it used. `REPOPILOT_EXECUTION_BACKEND`
chooses; `local` remains the default so a fresh clone works without Docker. The honest limits of
that boundary, including what Docker does not protect you from, are in
[`docs/architecture.md`](docs/architecture.md).

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
| 12 | Edit files on a branch in the run workspace (2B) | done |
| 13 | Test detection, execution, failure diagnosis, retry (2C) | done |
| 14 | Commit, pull request, structured run summary with VERIFIED / FAILED / UNVERIFIED (2D) | done |

Every seam the rest of the agent needs already exists: `StepOutcome.RETRY` plus `Run.attempts`
is the fix loop, `RunServices` is where the sandbox goes, and each agent capability is one more
entry in `default_pipeline()`.

## Requirements

- Python 3.11+
- Node.js 20+ (for the terminal UI)

## Layout

```
apps/
  api/repopilot/     FastAPI service (Python package)
  api/tests/         pytest suite
  web/               React + Vite terminal UI
packages/
  contracts/         TypeScript wire types mirroring the Python models
docs/
  architecture.md    How V1 works, its threat model, and what V2 defers
scripts/dev.mjs      Runs the API and the web app together
workspaces/          Folders you can point runs at with local://  (gitignored)
```

npm workspaces cover the Node side (`apps/web` and `packages/*`); the Python side stays a normal
pip package with its own virtualenv. There is no task runner. `scripts/dev.mjs` is
dependency-free, and a dependency would earn its place only by doing more than starting two
processes and stopping both together.

## Getting started

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

pip install -e ".[dev]"
npm install
```

Then, to run the API and the UI together in one terminal:

```bash
npm run dev
```

The API is then on <http://localhost:8000>, with docs at <http://localhost:8000/docs>, and the UI
on <http://localhost:5173>. Stopping that one command stops both, including the uvicorn reloader
and the Vite child it spawned.

To run only the backend:

```bash
repopilot serve
```

Prefer this rather than a bare `uvicorn --reload`. Two directories under the project hold other
people's code: the run workspace (`./.repopilot-data`), where a run clones a repository, and the
repositories directory (`./workspaces`), where you drop folders to be worked on. Repositories are
mostly `.py` files, so a bare `--reload` watches the project, sees those files, and restarts the
server in the middle of the run that is writing them, which cancels the run and loses its events.
`repopilot serve` derives both exclusions from `REPOPILOT_WORKSPACE_DIR` and
`REPOPILOT_REPOSITORIES_DIR` and prunes those directories from the watch, so reload works and runs
survive. `repopilot serve --no-reload` runs a single process.

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
| `GET /api/runs/repositories` | Folders available as `local://` references |
| `GET /api/runs/{id}` | Current state of one run |
| `GET /api/runs/{id}/events` | Full event history, with an `?after_sequence=` cursor |
| `DELETE /api/runs/{id}` | Cancel an in-flight run |
| `WS /ws/events` | Live event stream |

## Working on a local repository

Not every repository you want changed is on GitHub. Drop a folder into `workspaces/`:

```
workspaces/
├── project-a/
└── project-c/
```

and start a run against `local://project-a`. Everything downstream is unchanged: the run resolves,
copies, inspects, profiles, plans, edits, and tests exactly as it would for a cloned repository.

Two details make this safe:

- **Your folder is never written to.** The run copies the folder into its own checkout under
  `REPOPILOT_WORKSPACE_DIR` and works on that. Two runs of one folder get two separate copies, and
  the plan, edit, and repair steps can only reach the copy. The copy keeps your `.git`, so your
  branches and history come with it; a folder that is not a checkout gets one initialised, because
  the run creates a branch to work on.
- **The name is a single path segment.** `local://<name>` refers to exactly one direct child of the
  repositories directory, validated to a name that is legal as a directory on Windows as well as
  POSIX and checked for containment after resolving. `local://../anything` is not a repository.

`GET /api/runs/repositories` lists what is actually on disk, which is what the UI offers as
one-click buttons. The UI reads it once on load; it is a convenience, so the repository field
still accepts anything you type.

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
| `REPOPILOT_WORKSPACE_DIR` | `./.repopilot-data` | Where each run's checkout is created |
| `REPOPILOT_REPOSITORIES_DIR` | `./workspaces` | Where the folders `local://` refers to live |
| `REPOPILOT_CORS_ORIGINS` | `http://localhost:5173` | Comma-separated allowed frontend origins |
| `REPOPILOT_LLM_API_KEY` | _unset_ | Key for the reasoning backend. Not needed for local runtimes |
| `REPOPILOT_LLM_BASE_URL` | _empty_ | Provider endpoint. Empty means "no reasoning backend" |
| `REPOPILOT_LLM_MODEL` | _empty_ | Model name. Empty means "no reasoning backend" |
| `REPOPILOT_LLM_DIALECT` | `openai` | `openai` for any OpenAI-compatible API, or `anthropic` |
| `REPOPILOT_TEST_TIMEOUT_SECONDS` | `600` | Wall-clock limit for one test suite run. `0 < value <= 7200` |
| `REPOPILOT_EXECUTION_BACKEND` | `local` | Where untrusted commands run: `local` or `docker`. See below |
| `REPOPILOT_SANDBOX_IMAGE` | `python:3.12-slim` | Image commands run in, for the `docker` backend |
| `REPOPILOT_SANDBOX_NETWORK` | `none` | Container network mode. Keep it `none` unless you mean it |
| `REPOPILOT_SANDBOX_MEMORY` | `2g` | Container memory limit |
| `REPOPILOT_SANDBOX_CPUS` | `2` | Container CPU limit |
| `REPOPILOT_SANDBOX_PIDS_LIMIT` | `512` | Container process limit |
| `REPOPILOT_SANDBOX_USER` | _empty_ | Container user. Empty means the image's own |

**`.env` is gitignored and must never be committed.** No tokens, keys, or credentials belong
in this repository. If you add a secret-handling path later, read it from the environment only.

### Where the agent's code executes

RepoPilot runs code it did not write: a target repository's test suite, and whatever the model
asks for while repairing a failure. `REPOPILOT_EXECUTION_BACKEND` decides where that happens.

|  | `local` (default) | `docker` |
| --- | --- | --- |
| Process | child process on this host | throwaway container per command |
| Filesystem beyond the run's checkout | **accessible** | not mounted |
| Network | **unrestricted** | `none` by default |
| CPU / memory / processes | **unlimited** | capped |
| Privilege escalation | not prevented | refused |
| Needs | nothing | Docker installed |

`local` is the default so a fresh clone can run a test without Docker. It confines a command to the
run's workspace directory and strips `REPOPILOT_*` from its environment, and that is all it does.
**It is not a sandbox.** The UI says which one a run used.

`docker` mounts exactly one path — this run's checkout — and asks the runtime for no network, no
privilege escalation, no Linux capabilities, and hard caps on memory, CPU and process count. No
`REPOPILOT_*` variable is ever forwarded in: the GitHub token stays in RepoPilot's own process, where
the push happens, and the code under test cannot read it.

```bash
REPOPILOT_EXECUTION_BACKEND=docker
REPOPILOT_SANDBOX_IMAGE=python:3.12-slim   # pick one that matches the repository's stack
```

A container is **not** a complete security boundary, and the honest limits are in
[`docs/architecture.md`](docs/architecture.md) — anyone with Docker access effectively has root on
the host, a runtime vulnerability defeats all of it, and the image you choose is trusted input. The
container backend protects a developer running RepoPilot against a repository they do not fully
trust. It is not a multi-tenant hosting boundary.

Both backends implement one interface (`execution/environment.py`), so the agent engine does not know
which is in use. Kubernetes, when it is ever needed, is a third implementation of the same
interface.

### Configuring the reasoning backend

Phase 2A, 2B, and 2C need a model: it writes the implementation plan, the new contents of each
file, and the diagnosis of a failing test. `REPOPILOT_LLM_MODEL` and `REPOPILOT_LLM_BASE_URL`
both default to empty, which means *unconfigured*. That is on purpose: with no default endpoint, a
fresh install never builds a client and never makes a network call. A run that needs reasoning
fails with a message naming these settings.

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
.repopilot-data/local__<folder>/<run_id>/     # for a local:// run
```

Two runs of the same repository therefore cannot see each other's work, and a run never touches
a checkout you made yourself. Nothing outside `REPOPILOT_WORKSPACE_DIR` is written, and file
reads are checked for containment before use.

### The repository workspaces

`REPOPILOT_REPOSITORIES_DIR` defaults to `./workspaces` and holds your own folders, one
repository per directory. It is gitignored, because its contents are your code rather than
RepoPilot's; only its README is tracked, so the directory exists in a fresh clone. Point the
variable anywhere else to keep it outside the project.

Nothing here is written to. A local run reads the folder and copies it into
`REPOPILOT_WORKSPACE_DIR`, and the pipeline works on the copy from there on.

## Terminal UI

The Vite dev server runs on <http://localhost:5173> and proxies both `/api` and `/ws` to the
backend, so the browser talks to one origin and there is no CORS preflight.

```bash
npm run dev
```

`npm run dev` starts the backend too, so this is only needed if the API is already running
elsewhere. Start it, then open the page, enter a repository and a task, and press **Start run**.
Events stream in live and the panel follows the tail until you scroll up, at which point it stops
chasing so you can read.

If `workspaces/` holds any folders, they appear above the terminal as one-click buttons that fill
the repository field with `local://<name>`.

Below the terminal, three panels fill in as the run works, and all of them are derived purely
from the event stream, so a reconnecting browser rebuilds them from the replayed events alone:

- **Discovered files** appears as the `select_files` step streams its `file.selected` events, one
  row per file with its score, language, line count, and the reasons it was chosen.
- **Implementation plan** appears on `plan.created`, with the summary, approach, numbered steps,
  the files to change tagged `modify` / `add` / `read`, risks, and token usage.
- **Changes applied** appears as the `apply_plan` step streams its `file.written` events, one row
  per file with `new` or `edit`, its new line and byte counts, and the branch it was written on.
- **Verification** appears when the run reaches the test step: the runner it chose and the exact
  command, the verdict, the pass/fail counts, the failing tests with their output excerpts, and
  then the diagnosis and which files were repaired to get there. A run whose repository declares
  no test runner, or whose runner is not installed, says so instead of showing an empty verdict.

No panel fetches anything the backend did not already report, so what you read here is exactly
what the run did. The file bodies are deliberately not in the events: events are persisted and
replayed to every client, and a source file has no business in one. The same goes for a repair,
which reports as a `file.written` event flagged `repair`, so the UI can show that a file changed
twice without ever seeing either version.

```bash
npm run lint      # oxlint
npm run build     # tsc -b && vite build
```

`REPOPILOT_BACKEND` overrides the proxy target if the backend is not on port 8000.

### How the client uses the stream

`apps/web/src/useRunStream.ts` holds the three things that make an at-any-time attach work:

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
apps/api/repopilot/
  main.py          FastAPI app factory, lifespan, router wiring
  config.py        Environment-based settings
  cli.py           `repopilot serve`
  domain/          Pure data: Run, RunStatus, ExecutionEvent, RepositoryRef
  execution/       Engine, event bus, step pipeline (the pluggable agent seam)
  analysis/        Deterministic repo inspection: profile.py, relevance.py, testing.py
  agent/           Model-driven reasoning: llm.py, prompts.py, plan.py, implement.py, diagnose.py
  github/          GitHub client (repo metadata, clone, PR creation)
  api/             HTTP routes and the WebSocket endpoint
apps/api/tests/    pytest suite
apps/web/
  src/api.ts       REST client
  src/useRunStream.ts  WebSocket subscription: cursor, resync, reconnect
  src/App.tsx      Form, live terminal, discovered files, plan, changes and verification panels
packages/contracts/
  src/index.ts     Wire types mirroring the backend frames
```

The dependency direction is one-way: `api` -> `execution` -> `agent`/`analysis` -> `domain`.
`domain` imports nothing from the project. That is what keeps the rest of the agent addable
without refactoring.

`packages/contracts` is a hand-maintained mirror of the Python models, not generated from them.
Python stays the single source of truth; the mirror keeps the frontend from importing across the
workspace boundary at build time. When a frame changes, both copies change.

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

### What a write is allowed to do

`agent/implement.py` is where a plan becomes files, and it is the only code in the project that
writes anything. Three rules are enforced there rather than requested in a prompt:

- **One file per call.** The run decides the path and the role; the model is only asked for the
  body. There is no path in the reply to hallucinate, and a file the model gets wrong costs that
  file rather than the whole change.
- **The path is re-checked before every write.** Absolute paths, `..`, anything under `.git`,
  credential files (`.env`, `*.pem`, `id_rsa`, and friends), and anything that resolves outside
  the run's checkout are refused, and a refused path is recorded and skipped rather than
  written. Symlink escapes are caught by resolving before the containment test.
- **The content has to be a whole file.** A reply that opens with prose, carries an unclosed code
  fence, elides the rest of the file, or is byte-identical to what it was asked to change is
  rejected with a reason and asked for once more, not written.

Files are written with the bytes the model produced (`newline=""`, so Windows does not rewrite
line endings), parent directories are created for new files, and nothing is committed: the
changes stay in the working tree on the run's branch for the test step to run against.

### Running the tests

The `test` step answers one question: does the repository have a test suite, and does it pass?

`analysis/testing.py` picks a runner from evidence in the checkout rather than from the model. A
`pyproject.toml` with pytest, a `package.json` with a test script, a `go.mod`, a `Cargo.toml`, a
Maven or Gradle build, each name a framework and a fixed argv; the first one that matches wins
and the reason is streamed in the `test.suite_detected` event, so a wrong guess is visible rather
than silent. A repository that declares no runner emits `test.skipped` and the run finishes
without pretending to have verified anything.

`execution/sandbox.py` runs the argv. It is not a security boundary: it exists to make the test
step *well behaved*, not safe. It runs a fixed argument vector with no shell, in the run's own
checkout and nothing else; it kills the whole process tree if the run is cancelled; it stops at
`REPOPILOT_TEST_TIMEOUT_SECONDS` (600 by default) rather than hanging; and it keeps the tail of
the output rather than buffering all of it. The repository's test suite executes with the
permissions of the RepoPilot process, so run RepoPilot on code you are willing to execute.

The output is then parsed into counts and named failures. Parsing is the weakest part of any
such system, so the step is built to be honest when it is wrong: a runner that prints a tally
gets one, a runner that does not reports `null` rather than a fabricated zero, and a nonzero exit
with no recognisable failure list is reported as a failure of the suite itself instead of being
passed off as a clean run.

### Diagnosing a failure

A failing suite is not an error yet. The `diagnose` step exists to answer "what broke, and was
it something this run did?"

The model is shown the failing test names, the output excerpts, and the contents of the files
**this run wrote** - nothing else. It returns a `FailureDiagnosis` per test with a cause and
either a whole-file replacement or an honest "pre-existing". The reply is validated against
`agent/schema.py` and retried once if it will not parse, and the same whole-file check the
implementer uses is applied to any proposed content.

The rules that keep this from becoming a loop that rewrites the repository are structural:

- **A repair may only touch a file the run already wrote.** The prompt lists them, and the paths
  in the reply are re-checked against that list before anything is opened for writing. A model
  that decides the bug is somewhere else gets to say so, not edit there.
- **Bounded rounds.** At most two repairs per run, and the suite is re-run after each one, so
  the verdict reported at the end is the verdict of the code that is actually on disk.
- **A refused repair is not a repair.** Content that is not a whole file, or that leaves a file
  byte-identical, is rejected with a reason and the round is spent. The model cannot pass off a
  no-op as progress.
- **Pre-existing failures are reported, not fixed.** If the model attributes the failure to code
  the run never touched, that is the end of the diagnosis, stated as such.
- **Exhaustion is an honest outcome.** If the suite still fails after the rounds, the run asks
  for another attempt; if this was the last attempt, it fails with the reason, the round count,
  and the output already in the event stream.

One thing to be clear about: diagnosing a failure needs the model, so a run that fails its tests
with no reasoning backend configured ends at the verdict rather than in a repair loop.

## License

MIT
