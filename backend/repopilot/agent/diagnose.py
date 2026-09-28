"""Turning a failing test suite into a decision about what to change.

The diagnoser is asked one question, in one JSON object: given this output, what is broken and
which file fixes it. The interesting part is what happens to the answer, and that is not in this
module: the step decides whether the named file is one the run is allowed to rewrite, refuses
anything else, and writes through the same path-safe helpers the planner's writer used. A model
naming a file outside the run's own changes cannot get it written, so the diagnosis can talk about
a fix in a test file without the run reaching outside the change it made.

Two rules carry over from planning deliberately: a reply that will not parse is re-asked once with
the reason attached, and a transport failure is raised rather than retried, because asking a
broken endpoint again is how a run burns its time.
"""

from __future__ import annotations

import logging

from pydantic import ValidationError

from repopilot.agent.llm import ChatClient, ChatMessage, LLMError, extract_json_object
from repopilot.agent.prompts import build_diagnoser_prompt
from repopilot.agent.schema import TestFailureReport
from repopilot.analysis.testing import TestOutcome

logger = logging.getLogger("repopilot.agent.diagnose")

#: How many times to re-ask after a reply that is not a usable diagnosis.
MAX_DIAGNOSIS_ATTEMPTS = 2


class DiagnosisRejectedError(RuntimeError):
    """The model could not produce a usable diagnosis."""


async def generate_diagnosis(
    client: ChatClient,
    *,
    task: str,
    repository: str,
    outcome: TestOutcome,
    files: list[tuple[str, str | None]],
    round_number: int,
    previous: str = "",
    temperature: float = 0.0,
) -> TestFailureReport:
    """Ask the model why the suite failed and what would fix it.

    ``files`` is the run's own changes with their current contents, and it is the boundary: the
    prompt tells the model these are the only paths whose contents it may return.
    """
    turns = build_diagnoser_prompt(
        task=task,
        repository=repository,
        outcome=outcome,
        files=files,
        round_number=round_number,
        previous=previous,
    )
    messages = [ChatMessage(role=role, content=content) for role, content in turns]

    last_error = ""
    for attempt in range(1, MAX_DIAGNOSIS_ATTEMPTS + 1):
        try:
            completion = await client.complete(messages, temperature=temperature, json_mode=True)
        except LLMError:
            raise  # transport and auth problems are not fixed by asking again

        try:
            raw = extract_json_object(completion.text)
        except LLMError as exc:
            last_error = str(exc)
            messages = _correct(
                messages, completion.text, attempt, f"it was not valid JSON ({exc})"
            )
            continue

        try:
            report = TestFailureReport.model_validate(raw)
        except ValidationError as exc:
            last_error = _summarise_validation_error(exc)
            messages = _correct(
                messages,
                completion.text,
                attempt,
                f"it did not match the required shape ({last_error})",
            )
            continue

        report.model = completion.model or client.model
        report.prompt_tokens = completion.prompt_tokens
        report.completion_tokens = completion.completion_tokens
        logger.info(
            "diagnosis round %d from %s: %d findings, %d actionable",
            round_number,
            report.model,
            len(report.findings),
            len(report.actionable()),
        )
        return report

    raise DiagnosisRejectedError(
        f"the model did not return a usable diagnosis after {MAX_DIAGNOSIS_ATTEMPTS} attempts "
        f"({last_error})"
    )


def _correct(
    messages: list[ChatMessage], reply: str, attempt: int, reason: str
) -> list[ChatMessage]:
    logger.warning("diagnosis attempt %d returned unusable JSON: %s", attempt, reason)
    return [
        *messages,
        ChatMessage(role="assistant", content=reply[:2_000]),
        ChatMessage(
            role="user",
            content=(
                f"That reply was not usable because {reason}. Reply with only the JSON object "
                "described in your instructions, no prose and no code fences."
            ),
        ),
    ]


def _summarise_validation_error(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(part) for part in error['loc']) or 'report'}: {error['msg']}"
        for error in exc.errors()[:4]
    )
