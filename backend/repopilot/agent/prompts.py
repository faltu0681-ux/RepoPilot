"""Prompts for the reasoning steps.

Kept in one module so the agent's instructions are auditable in one place and can be diffed
when quality needs tuning. Each prompt states the repository's ground rules explicitly, because
the single most common failure mode for a coding model is inventing files that do not exist.
"""

from __future__ import annotations

from pathlib import Path

from repopilot.agent.schema import ImplementationPlan, PlanStep
from repopilot.analysis.profile import RepositoryProfile
from repopilot.analysis.relevance import CandidateFile, snippet

PLANNER_SYSTEM = """\
You are RepoPilot, a senior engineer planning a change to an existing codebase.

You are given a repository that has already been cloned, a measured inventory of its files, \
and a shortlist of files that look relevant. Produce an implementation plan.

Hard rules:
- Only reference file paths that appear in the inventory you were given. Never invent a path, \
module, class, or function. If the change seems to need a file you were not shown, say so in \
risks instead of naming it as if it exists.
- Ground every step in the supplied file contents. Quote the actual symbol names you rely on.
- Prefer changing existing files over creating new ones. Create a file only when the task \
genuinely calls for one, and use role "add" for it.
- Mark a file "read" when it is context you consulted but do not intend to modify.
- Steps must be ordered, concrete, and independently checkable. No "and then clean things up".
- If the task is ambiguous, pick the most reasonable reading, state the assumption in \
approach, and carry on. Do not stall.

Respond with a single JSON object and nothing else. Shape:
{
  "summary": "one paragraph: what the change does",
  "approach": "the chosen reading of the task, the key assumptions, and why this shape of change",
  "relevant_files": [
    {"path": "<path from the inventory>", "reason": "<why it matters>", "role": "modify|add|read"}
  ],
  "steps": [
    {"order": 1, "title": "<short imperative>", "detail": "<what to change and how>",
     "files": ["<path>"], "rationale": "<why this step comes here>"}
  ],
  "risks": ["<risk, or an empty list>"],
  "test_strategy": "how this change should be verified"
}
"""

#: Cap on the number of candidates whose contents are quoted into the prompt.
SNIPPET_BUDGET_CHARS = 60_000
MAX_SNIPPET_FILES = 12

IMPLEMENTER_SYSTEM = """\
You are RepoPilot, a senior engineer applying an approved implementation plan to a real \
checkout. You are given one file and you return that file's new contents.

Hard rules:
- Reply with the complete new contents of the file you were asked about, and nothing else. No \
preamble, no explanation, no markdown code fence, no comments about being an AI.
- Output the whole file. Never a fragment, never a diff, never an elision: writing "..." or \
"# rest of the file unchanged" is a failure, because what you return is written to disk verbatim.
- Preserve the file's existing style exactly: indentation, naming, comment density, import order, \
and line length. Make the smallest change that satisfies the task and keep everything the task \
did not ask you to change.
- Keep the file working on its own terms: do not rename or delete public symbols unless the task \
says to, and do not leave a placeholder, a TODO, or a stub where working code was.
- Never write a secret, credential, token, or key into a file, and never target a .env file.
- The plan is a plan, not gospel. If it names a symbol that is not in the file you were given, \
implement the intent against what is actually there.
"""


def _inventory_lines(profile: RepositoryProfile, candidates: list[CandidateFile]) -> list[str]:
    lines: list[str] = []
    for candidate in candidates:
        flag = " [test]" if candidate.is_test else ""
        lines.append(
            f"- {candidate.path} ({candidate.language or 'unknown'}, {candidate.lines} lines){flag}"
        )
    return lines


def _profile_block(profile: RepositoryProfile) -> str:
    languages = ", ".join(f"{name} ({count})" for name, count in profile.languages[:8]) or "unknown"
    return (
        f"Primary language: {profile.primary_language or 'unknown'}\n"
        f"Languages by file count: {languages}\n"
        f"Source files measured: {profile.total_files}\n"
        f"Top-level entries: {', '.join(profile.top_level[:30]) or '(none)'}\n"
        f"Build/dependency manifests: {', '.join(profile.manifests) or '(none)'}\n"
        f"Test configuration: {', '.join(profile.test_configs) or '(none)'}\n"
        f"Docs: {', '.join(profile.docs) or '(none)'}"
    )


def build_planner_prompt(
    *,
    task: str,
    repository: str,
    default_branch: str | None,
    profile: RepositoryProfile,
    candidates: list[CandidateFile],
    root: Path | None,
) -> list[tuple[str, str]]:
    """Build the chat turns for one planning request."""
    if not candidates:
        raise ValueError("cannot plan without any candidate files")

    inventory = "\n".join(_inventory_lines(profile, candidates))
    sections = [
        f"Repository: {repository}"
        + (f" (default branch {default_branch})" if default_branch else ""),
        "",
        "Measured profile:",
        _profile_block(profile),
        "",
        f"Shortlisted files ({len(candidates)} of {profile.total_files} measured), highest "
        "confidence first:",
        inventory,
    ]

    # Spend the context budget on the files most likely to matter, and stop cleanly.
    budget = SNIPPET_BUDGET_CHARS
    quoted: list[str] = []
    for candidate in candidates:
        if len(quoted) >= MAX_SNIPPET_FILES or budget <= 0 or root is None:
            break
        text = snippet(root, candidate.path, max_chars=min(budget, 8_000))
        if not text:
            continue
        quoted.append(f"--- {candidate.path} ---\n{text}")
        budget -= len(text)

    if quoted:
        sections += ["", "Contents of the highest-ranked files:", *quoted]

    sections += [
        "",
        "The shortlist is a heuristic, not a verdict. If a file you need is not shortlisted but \
is visible in the profile above, you may still reference it by its real path. If nothing in \
the profile plausibly relates to the task, set relevant_files to [] and explain why in risks.",
        "",
        f"Task from the user:\n{task}",
    ]

    return [
        ("system", PLANNER_SYSTEM),
        ("user", "\n".join(sections)),
    ]


def _plan_steps_for(plan: ImplementationPlan, path: str) -> list[PlanStep]:
    return [step for step in plan.steps if path in step.files]


def build_implementer_prompt(
    *,
    task: str,
    repository: str,
    plan: ImplementationPlan,
    path: str,
    role: str,
    current: str,
    siblings: list[str] | None = None,
) -> list[tuple[str, str]]:
    """Build the chat turns that ask for one file's new contents.

    The plan is included in condensed form rather than whole: the step that touches this file,
    why the plan picked it, and the other files this run is changing, so the model writes
    against the same names the rest of the change uses.
    """
    reason = next(
        (f.reason for f in plan.relevant_files if f.path == path),
        "",
    )
    steps = _plan_steps_for(plan, path)

    sections = [
        f"Repository: {repository}",
        "",
        f"Task from the user:\n{task}",
        "",
        "Implementation plan you are applying:",
        f"- summary: {plan.summary}",
    ]
    if plan.approach:
        sections.append(f"- approach: {plan.approach}")
    if reason:
        sections.append(f"- {path} ({role}): {reason}")
    for step in steps:
        detail = f" - {step.detail}" if step.detail else ""
        sections.append(f"- step {step.order}: {step.title}{detail}")

    if siblings:
        sections += [
            "",
            "Other files this same run is changing. Do not write them here, but keep names, "
            "imports, and signatures consistent with them: " + ", ".join(siblings),
        ]

    if current:
        sections += [
            "",
            f"Current contents of {path}, exactly as they are on disk:",
            "-----",
            current,
            "-----",
            "",
            f"Return the new contents of {path}. Output only the file.",
        ]
    else:
        sections += [
            "",
            f"{path} does not exist yet. Create it, complete and runnable, with whatever "
            "imports, types, and docstring the surrounding code implies.",
            "",
            f"Return the contents of the new {path}. Output only the file.",
        ]

    return [
        ("system", IMPLEMENTER_SYSTEM),
        ("user", "\n".join(sections)),
    ]
