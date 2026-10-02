# RepoPilot V1 architecture

This document describes how V1 works, where its security boundary actually is, and what has been
deliberately left out. It is written for someone deciding whether to trust a run, and for whoever
picks this up next.

The short version: V1 is one Python process, one agent, and a boundary around the code that agent
runs. Everything else is either a well-understood dependency or deliberately absent.

---

## 1. High-level architecture

```
                 browser (React + Vite)
                            |  HTTP + WebSocket
                            v
        +------------------------------------+
        |  FastAPI (single process)           |
        |                                    |
        |  api/      runs, events, websocket  |
        |  execution/ engine + step pipeline  |
        |  agent/    LLM client, planning     |
        |  analysis/ profile, diff, tests     |
        |  github/   REST client              |
        |  domain/   Run model + event store  |
        +------------------------------------+
             |                    |
             v                    v
   ExecutionEnvironment      GitHub REST API
   (local | docker)
```

Two boundaries matter and they are different things:

* **The control plane** is this process. It holds the GitHub token and the LLM key, decides what to
  do, and drives git. It is trusted.
* **The execution boundary** is where code RepoPilot did not write runs — a target repository's test
  suite, and anything the model asks for. It is untrusted, and it is the part that is sandboxed.

A run never moves a credential across that boundary.

### Module map

| Module | Responsibility |
| --- | --- |
| `api/` | HTTP and WebSocket surface. Holds no state of its own. |
| `execution/engine.py` | Run lifecycle, retries, status transitions, event emission. |
| `execution/steps.py` | The step pipeline: one class per unit of work. |
| `execution/environment.py` | The `ExecutionEnvironment` interface and its factory. |
| `execution/sandbox.py` | The `local` backend: a child process. |
| `execution/docker.py` | The `docker` backend: an ephemeral container. |
| `execution/workspace.py` | `local://` resolution and the per-run copy. |
| `agent/` | LLM client, prompts, structured plan schema. |
| `analysis/` | Repository profile, file ranking, diffs, test parsing. |
| `github/` | Repository reads and pull-request creation. |
| `domain/` | The `Run` model and the in-memory store. |

---

## 2. Workspace flow

```
user's folder                 RepoPilot's private copy
workspaces/<name>     --->    .repopilot-data/<owner>__<repo>/<run_id>
(read-only input)             (disposable; the only thing the agent sees)
```

`local://<name>` names one folder in the configured repositories directory. It is resolved as a
name, not a path: the name must be a single segment, the result is re-checked for containment, and
anything else is refused rather than searched for. `apps/api/repopilot/execution/workspace.py:60`

Every run works on a **copy**. The copy is what gets branched, edited, tested, and committed, and
the original is never opened for writing. This is asserted by tests rather than assumed, and it is
what makes it safe to point RepoPilot at a repository you care about.

Build artefacts (`node_modules`, `.venv`, `dist`, …) are excluded from the copy. The checkout's own
`.git` *is* copied, because a run needs a history to branch from.

---

## 3. Sandbox flow

The `ExecutionEnvironment` protocol has one method that matters:

```python
async def run(argv, *, cwd, timeout_seconds=None, is_cancelled=None, env=None) -> CommandResult
```

Both backends implement it. The engine and every step call it without knowing which is in use.

**The workspace is durable; the sandbox is not.** RepoPilot does not keep a container alive across a
run. Each command gets a fresh container with `--rm`, and what carries state between commands is the
mounted workspace on the host. A suite that fails, gets repaired, and runs again sees its own edits
because they are the same files, not because anything survived.

Consequences, both deliberate:

* There is no long-lived thing to leak, and therefore no cleanup path that can itself fail.
* A per-command container costs about a second of startup. A test suite does not care.

### Backends

| | `local` (default) | `docker` |
| --- | --- | --- |
| Process | child process on the host | throwaway container |
| Working directory | enforced | enforced |
| Environment | inherited, `REPOPILOT_*` stripped | built from an allowlist |
| Filesystem beyond the workspace | **accessible** | not mounted |
| Network | **unrestricted** | `none` by default |
| CPU / memory / processes | **unlimited** | capped |
| Privilege escalation | not prevented | refused |

`local` is the default so a fresh clone can run a test without Docker installed. It is not a sandbox
and the UI says so in words, not just in colour.

### What the container actually enforces

One mount, and only one: this run's checkout, at `/workspace`. Alongside it:

* `--network none`
* `--memory` / `--memory-swap` (equal, so overcommit is off)
* `--cpus`, `--pids-limit`
* `--cap-drop ALL`, `--security-opt no-new-privileges`, `--read-only`, `--user` when set
* a `tmpfs /tmp`, which is also where `HOME` points, because the image's root filesystem is read-only

---

## 4. Agent flow

```
task -> validate -> resolve -> clone -> inspect -> profile -> select files -> plan
                                                                      |
                                                                      v
                                      apply plan -> test -> diagnose -> commit
                                                                              |
                                                                    pull request -> summarise
```

* **Profile** measures the checkout: languages, sizes, entry points, test framework.
* **Select** ranks files by relevance with no model involved, so the expensive step sees less.
* **Plan** is the only step that calls the LLM for reasoning, and it returns a schema-validated
  `ImplementationPlan`. An unconfigured model raises rather than inventing output.
* **Apply** writes files through a path-safe writer that refuses to escape the checkout.
* **Diagnose** may repair, bounded on both sides: at most N rounds within an attempt, and at most 3
  attempts across the run. A failure in a file this run did not write is reported, not repaired.

One agent, one run, one attempt loop. The boundaries that a future agent pool would need —
`Agent`, `Task`, `Run`, `ExecutionEnvironment` — already exist as types.

---

## 5. Test and verification flow

The verdict is three-valued and the distinction is the point:

| Verdict | Meaning |
| --- | --- |
| `verified` | The suite ran and passed. |
| `failed` | The suite ran and did not pass. |
| `unverified` | Nothing proved or disproved the change: no runner found, runner missing, or timeout. |

`unverified` exists so that "I could not check" is never reported as either "it works" or "it is
broken". A repository that declares no test runner was never verified either way.

A container backend that cannot run the command at all — missing image, dead daemon — reports
**unavailability**, not a failing suite. This distinction is why `_error_for` in
`apps/api/repopilot/execution/docker.py` exists: without it, "no test ever ran" would arrive
wearing the costume of "the tests failed".

---

## 6. Git flow

Git runs **on the host, in the control plane**, never in the sandbox:

1. `_measure_change` — what the run actually changed, read from git rather than guessed.
2. `CommitChangesStep` — stages and commits on the run's branch.
3. `OpenPullRequestStep` — pushes, then opens the PR.

The token lives here because a push needs it. The agent's code never sees it.

---

## 7. GitHub PR flow

```
repository -> isolated workspace -> change -> test -> commit -> push branch -> open PR -> PR URL
```

`OpenPullRequestStep` skips rather than fails, and says why:

* local repository, no remote, no token, nothing to propose, push refused, API refused.

A run that cannot open a PR has still made and tested a change, and says so.

The PR body carries the verdict and the diff. Neither the commit subject nor the PR title contains
the word "verified" — a title that says "VERIFIED" invites a reviewer to trust the word instead of
the suite.

**Mocked vs real.** Tests use `httpx.MockTransport` for the API and a real local bare remote for
`git push`. Both halves are therefore genuinely exercised: the ordering (push, then propose) and
the request body are asserted, not assumed. What is *not* covered is a live PR against GitHub, which
needs a token and a throwaway repository.

---

## 8. Security boundaries

### Threat model

**What this is for.** A developer running RepoPilot locally against a repository they do not fully
trust — a public project, a colleague's code, an unfamiliar codebase. The agent writes code from a
model's reading of that repository, and that code is executed.

**What RepoPilot assumes about its operator.** The operator is trusted with the RepoPilot process.
Whoever can start the process, read its memory, or read its environment already holds the GitHub and
LLM credentials. This is a single-user tool and does not pretend otherwise.

**Adversaries, in rough order of plausibility:**

1. **The model, going wrong.** It writes something that deletes files, loops forever, forks
   thousands of processes, or runs a command it was never asked to run. *Mitigations:* the agent
   only ever sees the run's copy; the path-safe writer refuses escapes; timeouts kill the process
   tree; `docker` caps memory, CPU and process count; the verdict reports what actually happened.
2. **The target repository, running its own code.** A test suite that reaches outside the workspace,
   reads the environment, or phones home. *Mitigations:* in `docker`, one mount and no network; in
   `local`, a filtered environment and a working directory that is a convention rather than a
   boundary.
3. **A malicious pull request or task prompt.** Prompt injection through repository content steering
   the agent. *Mitigations:* the schema-constrained plan, the write allowlist, and — most
   importantly — the sandbox and the honest verdict, which mean a hijacked agent produces a failing
   run in a container rather than a compromised workstation.

**Explicitly not defended against:**

* **Anyone with Docker access.** Anyone who can talk to the Docker daemon, or start containers,
  effectively has root on the host. This is Docker's own documented position and RepoPilot does not
  improve on it. On a shared or multi-tenant machine, the container backend is not a security
  boundary.
* **A kernel or container-runtime vulnerability.** All container isolation reduces to this.
* **A malicious image.** `REPOPILOT_SANDBOX_IMAGE` is trusted input. Use an image you would run
  anyway.
* **Resource exhaustion of the host.** Capped inside the container, uncapped outside it. A run that
  fills the disk fails on the host's terms.
* **Secret exfiltration via the model.** The model receives repository content and sends a task.
  Anything sensitive in the repository leaves with it. That is inherent to sending code to a model.
* **The local backend as a security boundary.** It is not one. `local` is a convenience default for
  trusted code, not protection.

### The credential boundary

```
control plane                         sandbox
-------------                         -------
REPOPILOT_GITHUB_TOKEN  ──✗──>   never forwarded
REPOPILOT_LLM_API_KEY   ──✗──>   never forwarded
git push / PR creation  ──✗──>   never needed inside
```

* `docker` builds its environment from an **allowlist**, not an inherited one minus known secrets. A
  host variable nobody considered cannot ride along.
* Host tool paths (`PATH`, `JAVA_HOME`, `GOPATH`) are **excluded by default**: they name host
  locations that do not exist in a container, and forwarding them produces a confusing "command not
  found" instead of an honest missing runner. The image supplies its own toolchain.
* `local` filters `REPOPILOT_*` out of an inherited environment. Weaker by construction, because
  filtering an inheritance is a denylist and a denylist is only as good as its completeness.
* Tokens are `SecretStr`, so they do not appear in logs or `repr()`.

### Repository safety

The original folder is never written to. A run clones or copies, and every subsequent write targets
the copy. Git and the network are the only ways out of the workspace, and git is driven by the
control plane.

---

## 9. What V1 is

* A local, single-user agent that takes a repository and a task and returns a change with an honest
  verdict.
* One agent, one attempt loop, bounded.
* Two execution backends behind one interface, so the choice of boundary is configuration.
* Commit, push and PR toward GitHub, with every failure mode reported rather than hidden.
* A workspace UI that shows the plan, the diff, the test result, the verdict, and the PR.
* In-memory run state. Nothing survives a restart, on purpose: V1 is a demonstrator, not a service.

## 10. What is deferred to V2

| Deferred | Why it is safe to defer | Where it would attach |
| --- | --- | --- |
| Kubernetes | Same interface, different backend. | A third `ExecutionEnvironment`. |
| Agent pool | One agent per run; `Agent`/`Task`/`Run` already exist as types. | A scheduler choosing an agent per run. |
| Redis / job queue | The engine already drives runs as background tasks and owns the lifecycle. | `engine.start` becomes an enqueue. |
| Microservices | Modules are already separable; nothing crosses a process boundary. | Each `api/` module is already a router. |
| Postgres | `RunStore` is one class with one interface. | Reimplement it; nothing above it knows runs are a dict. |
| Autoscaling / multi-user / billing | V1 is single-user and local by definition. | Entirely new product surface. |

The test for whether this list is honest: **each item should attach to an existing seam rather than
requiring a rewrite.** That is the whole reason the seams exist.

---

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `REPOPILOT_EXECUTION_BACKEND` | `local` | `local` or `docker`. |
| `REPOPILOT_SANDBOX_IMAGE` | `python:3.12-slim` | Image commands run in. Per stack. |
| `REPOPILOT_SANDBOX_NETWORK` | `none` | Container network mode. |
| `REPOPILOT_SANDBOX_MEMORY` | `2g` | Container memory limit. |
| `REPOPILOT_SANDBOX_CPUS` | `2` | Container CPU limit. |
| `REPOPILOT_SANDBOX_PIDS_LIMIT` | `512` | Container process limit. |
| `REPOPILOT_SANDBOX_USER` | *(empty)* | Container user. Empty means the image's own. |
| `REPOPILOT_SANDBOX_STARTUP_TIMEOUT_SECONDS` | `300` | Ceiling for starting a container. |
| `REPOPILOT_SANDBOX_DOCKER_BINARY` | `docker` | Container CLI. Point at a shim to test. |
| `REPOPILOT_TEST_TIMEOUT_SECONDS` | `600` | Per-command ceiling, in either backend. |

### Choosing an image

One image cannot contain every language runtime. Pick one that matches the repository's stack; the
runner probe will report `unavailable` and the run will end `unverified` if it does not. That is the
honest outcome, and it is why the verdict system has three states rather than two.