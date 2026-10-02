"""RepoPilot's reasoning layer.

The boundary between deterministic analysis and model-driven reasoning:

* :mod:`repopilot.analysis` measures a checkout and ranks files without a model.
* :mod:`repopilot.agent` asks a configured model to turn that evidence into a plan.

:mod:`repopilot.agent.llm` is the only module that knows which provider is in use.
"""

from repopilot.agent.llm import (
    ChatClient,
    ChatMessage,
    Completion,
    LLMError,
    LLMNotConfiguredError,
    LLMRequestError,
    LLMResponseError,
    build_chat_client,
)
from repopilot.agent.plan import PlanRejectedError, generate_plan
from repopilot.agent.schema import FileSelection, ImplementationPlan, PlanStep

__all__ = [
    "ChatClient",
    "ChatMessage",
    "Completion",
    "FileSelection",
    "ImplementationPlan",
    "LLMError",
    "LLMNotConfiguredError",
    "LLMRequestError",
    "LLMResponseError",
    "PlanRejectedError",
    "PlanStep",
    "build_chat_client",
    "generate_plan",
]
