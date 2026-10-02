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
from repopilot.analysis.testing import TestOutcome

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


DIAGNOSER_SYSTEM = """\
You are RepoPilot, a senior engineer looking at a test suite that just failed after a change was \
applied to a real checkout. You are given the failure output and the files that were changed, and \
you return a diagnosis.

Hard rules:
- The failure output is your only evidence. Read it before deciding anything, and do not claim a \
cause the output does not support.
- Only the files listed as changed by this run are yours to fix. If the honest fix is in another \
file, name that file in "path" anyway and explain in "cause" that it is outside this run; the run \
will refuse the change and say so rather than write it.
- Return the complete new contents of the file you are fixing, verbatim, as the "content" field. \
Never a diff, never a fragment, never "..." or "rest of the file unchanged": what you return is \
written to disk as the whole file.
- If the failure cannot be caused by these changes, say so: set "pre_existing" to true and leave \
"content" empty. A repository that was already broken is reported, not rewritten.
- Never make a failing test pass by weakening it. Do not delete a test, lower an assertion, skip a \
case, or catch the exception the test is there to catch. If the test is genuinely wrong, say so in \
"cause" and return the fixed source, not the fixed test.
- Fix the cause, not the symptom, and keep the change as small as the cause allows. Do not \
refactor, reformat, or touch code the failure does not point at.
- Never write a secret, credential, token, or key, and never target a .env file.

Reply with one JSON object and nothing else:

{
  "summary": "one paragraph: what is failing and why",
  "findings": [
    {
      "test": "the failing test, as the runner named it",
      "path": "a file to rewrite, or the file the fix belongs in even if you may not write it",
      "cause": "what in the code causes this failure, and the specific change that fixes it",
      "pre_existing": false,
      "content": "the complete new contents of that file, or an empty string"
    }
  ]
}
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


#: A repair round quotes the failing output and the files it may rewrite, and both are capped:
#: the output is trimmed to its end by the caller, and a file too large to quote is named without
#: its contents rather than truncated into something the model would copy.
DIAGNOSIS_SNIPPET_CHARS = 12_000


def build_diagnoser_prompt(
    *,
    task: str,
    repository: str,
    outcome: TestOutcome,
    files: list[tuple[str, str | None]],
    round_number: int,
    previous: str = "",
) -> list[tuple[str, str]]:
    """Build the chat turns that ask why the suite failed and what would fix it.

    ``files`` is the run's own changes as ``(path, contents)``, with ``None`` for a file too
    large to quote. Only those paths may be written back, so the inventory is both the evidence
    and the boundary: the model is told the limit rather than discovering it.
    """
    sections = [
        f"Repository: {repository}",
        "",
        f"Task from the user:\n{task}",
        "",
        f"The {outcome.framework} suite failed. Command: {outcome.as_command_text()}",
        f"Exit status: {outcome.exit_code}, after {outcome.duration_seconds:.1f}s",
    ]
    if outcome.tally.failures:
        sections.append("")
        sections.append("Failing tests:")
        for failure in outcome.tally.failures:
            detail = f" - {failure.message}" if failure.message else ""
            sections.append(f"- {failure.test}{detail}")
    if outcome.output:
        sections += [
            "",
            "Output (the end of the run):",
            "-----",
            outcome.output,
            "-----",
        ]
    if previous:
        sections += [
            "",
            "A previous repair round did not make the suite pass. It is still failing as shown "
            f"above. That earlier attempt said: {previous}",
        ]

    sections += ["", "Files this run changed, and the only files you may return contents for:"]
    for path, contents in files:
        if contents is None:
            sections.append(f"- {path} (too large to quote here; you may still return it)")
        else:
            sections += ["", f"Current contents of {path}:", "-----", contents, "-----"]
        sections.append(f"^^ {path}")

    sections += [
        "",
        f"Diagnose the failure (repair round {round_number}) and reply with the JSON object "
        "described in your instructions.",
    ]
    return [
        ("system", DIAGNOSER_SYSTEM),
        ("user", "\n".join(sections)),
    ]
