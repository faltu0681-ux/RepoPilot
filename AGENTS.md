## Glyph (AI code search)

This repo is indexed by Glyph over MCP. For any multi-file task — debugging, refactoring, tracing callers, or asking how a feature works — call `get_context` before reading files. Check `analyze_impact` before modifying shared code. For change questions: `git diff` for what changed, `describe_diff` for what it breaks. Nothing to install and nothing to set up: at session start and after every edit/commit/pull, call `sync_working_tree` with the branch diff and pass the same session id explicitly on every tool call.

