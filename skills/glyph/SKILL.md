---
name: glyph
description: Use when debugging, refactoring, tracing callers, or asking where something is used, how a feature works, what a push changed, or what a change would break — in a repository connected to Glyph. Consults the indexed codebase over MCP instead of manual grep. Also for onboarding to an unfamiliar repo.
---

# Glyph

This repository is indexed by Glyph: semantic code search over the
full codebase, available as MCP tools. Prefer these tools over manual
`grep`/`Glob`/file-reading when you don't already know exact file paths.

## Workflow

0. **Send your dirty paths first (nothing to install, nothing to set
   up).** The index tracks `main`; your checkout usually doesn't. Before
   the first `describe_diff` on a feature branch or dirty tree — and again
   after every edit, commit, pull, or push — call `sync_working_tree`:
   branch name from `git rev-parse --abbrev-ref HEAD`, merge-base from
   `git merge-base HEAD origin/main` (local `main` if origin is stale),
   plus `porcelain` with raw `git status --porcelain` output (a few
   hundred bytes, never file contents). Skip `.env` files, keys, and
   secrets. Omit sessionId and use the returned one, then pass it
   explicitly on every tool call. Pushed branch with nothing to sync?
   Pass `branch` (and `headSha`) straight to `describe_diff` — the
   server fetches it. Need symbol deltas for uncommitted work? Run the
   piped-upload command from the `describe_diff` stamp instead.
   If the branch was pushed from elsewhere, `git fetch origin`
   and fast-forward first; if that fails, say the checkout is diverged
   and stop. No overlay (or an overlay older than your last edit — every
   response stamps when the diff was computed) means re-sync, not work
   around it. Re-syncing an unchanged tree replies `unchanged: true` —
   cheap, no penalty, and proof your overlay is current. Sessions slide:
   every call renews the 10-minute window, so long tasks never lose it.
1. **Understand before changing.** For unfamiliar code, debugging, or any
   task spanning more than one file, call `get_context` FIRST — one call
   replaces dozens of file reads with implementations, callers, and
   related code in a token budget.
2. **Check dependents before editing shared code.** Call `find_references`
   for direct callers, `analyze_impact` for transitive dependents (callers,
   type references, heritage — two hops). Never claim something breaks
   unless the tools show evidence.
3. **Know what a change breaks — git diff can't tell you.** When asked what changed
   recently, what a push did, or to review a branch, use `git diff` for
   the file list (free, instant, complete), then call `describe_diff`
   for the part git can't answer: per-symbol added/removed deltas and
   who breaks, with file:line cites. After step 0 the brief has two sections:
   indexed `main` history plus your branch-vs-main (committed and
   uncommitted), stamped with the indexed SHA and when your diff was
   computed. Without a fresh sync, scope is pushed/indexed history only:
   for uncommitted working-tree edits, use `git diff`.
4. **Orient cheaply.** `get_symbol` for definitions, `get_file_outline`
   to map a file without reading it all, `search_code` when you need
   meaning-ranked results instead of exact strings.
5. **Verify after.** Inspect the actual source files before editing.
   Do not re-read files already shown in MCP output — cite the shown
   lines. Verify with the single most relevant test file plus `-q`;
   paste only failures.
   If answers look stale after a push, call `index_status` — the index
   may still be catching up.
6. **Skip for trivial tasks.** Single-file edits you already understand
   don't need context queries.

## Tool guide

| Task                                        | Tool                |
| ------------------------------------------- | ------------------- |
| Finding relevant code for a task            | `search_code`       |
| Understanding a feature before modifying it | `get_context`       |
| Checking direct callers and dependencies    | `find_references`   |
| Understanding what a change could break     | `analyze_impact`    |
| Reviewing what a change breaks              | `describe_diff`     |
| Sending your branch diff                    | `sync_working_tree` |
| Reading a definition                        | `get_symbol`        |
| Mapping a file                              | `get_file_outline`  |
| Checking index freshness                    | `index_status`      |

## Always-on project rule

Paste this into `AGENTS.md` (or your agent's equivalent rules file) so
agents consult the index without needing this skill file loaded:

```md
This repo is indexed by Glyph over MCP. For any multi-file task —
debugging, refactoring, tracing callers, or asking how a feature works —
call `get_context` before reading files. Check `analyze_impact` before
modifying shared code. For change questions: `git diff` for what changed,
`describe_diff` for what it breaks. Nothing to install and nothing to
set up: at session start and after every edit/commit/pull, call
`sync_working_tree` with the branch diff and pass the same session id
explicitly on every tool call.
```

## Install

Copy this file from the console (Agent skill tab) and save it as
`SKILL.md` in your agent's skills directory, then connect a repo.
Check your agent's docs for the exact skills path — it is usually a
`skills/glyph/SKILL.md` file under the agent's personal or project
config directory (personal applies everywhere, project-shared applies
to the team).

Then add the MCP server (remote HTTP with your per-repo key) and
confirm with `index_status` before the first real question.
