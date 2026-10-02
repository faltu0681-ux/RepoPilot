"""The reasoning backend RepoPilot talks to.

Phase 2 needs a real language model to turn a natural-language task into an implementation
plan (2A), to write code (2B), and to diagnose failing tests (2C). This module is the single
seam between the agent and whichever model the operator has configured.

Two dialects cover the realistic choices:

* ``openai`` -- any OpenAI-compatible ``/chat/completions`` endpoint. This is OpenAI itself,
  Ollama, OpenRouter, vLLM, LM Studio, and most gateways.
* ``anthropic`` -- Anthropic's Messages API.

Both are implemented over ``httpx``, which the project already depends on, so adding a
provider costs a config value rather than a dependency.

Nothing here fakes a response. If no model is configured, or the provider returns an error,
the run fails loudly: a fabricated plan would be worse than no plan, because every later phase
trusts it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import httpx

from repopilot.config import Settings

logger = logging.getLogger("repopilot.agent.llm")


class LLMError(RuntimeError):
    """Base class for every reasoning-backend failure."""


class LLMNotConfiguredError(LLMError):
    """No usable model is configured, so reasoning cannot be attempted at all."""


class LLMRequestError(LLMError):
    """The provider rejected the request, or returned something unusable."""


class LLMResponseError(LLMRequestError):
    """The provider replied, but the body did not contain usable text."""


# --------------------------------------------------------------------------- types


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One message in a chat exchange."""

    role: str  # "system" | "user" | "assistant"
    content: str

    def as_openai(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}

    def as_anthropic(self) -> dict[str, str]:
        # Anthropic takes the system prompt out of band; as_anthropic_system() pulls it out.
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True, slots=True)
class Completion:
    """A model's reply, plus whatever usage the provider reported."""

    text: str
    model: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def total_tokens(self) -> int | None:
        if self.prompt_tokens is None and self.completion_tokens is None:
            return None
        return (self.prompt_tokens or 0) + (self.completion_tokens or 0)


@runtime_checkable
class ChatClient(Protocol):
    """What the agent needs from a reasoning backend."""

    model: str

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> Completion:
        """Return the model's reply. Raises ``LLMError`` on any failure."""
        ...

    async def aclose(self) -> None:
        """Release the underlying HTTP connections."""
        ...


# ------------------------------------------------------------------- JSON handling


def extract_json_object(text: str) -> dict[str, Any]:
    """Pull a JSON object out of a model reply.

    Models wrap JSON in prose or ```json fences often enough that strict parsing fails on
    otherwise good answers, so this tries the whole string first and then the widest balanced
    brace span. Raises ``LLMResponseError`` when there is no object to be found.
    """
    text = text.strip()
    if not text:
        raise LLMResponseError("model returned an empty response")

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, list):
        raise LLMResponseError("model returned a JSON array where an object was required")

    start = text.find("{")
    while start != -1:
        depth = 0
        for index in range(start, len(text)):
            char = text[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : index + 1]
                    try:
                        result = json.loads(candidate)
                    except json.JSONDecodeError:
                        break  # unbalanced prose; try the next '{'
                    if isinstance(result, dict):
                        return result
                    break
        start = text.find("{", start + 1)

    raise LLMResponseError(
        f"could not find a JSON object in the model response (first 200 chars: {text[:200]!r})"
    )


# ------------------------------------------------------------------- HTTP clients


class _HttpChatClient:
    """Shared httpx plumbing for the concrete dialects."""

    dialect = ""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str | None,
        timeout: float = 120.0,
        max_tokens: int = 4096,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if not model.strip():
            raise LLMNotConfiguredError("no model configured (set REPOPILOT_LLM_MODEL)")
        if not base_url.strip():
            raise LLMNotConfiguredError("no base url configured (set REPOPILOT_LLM_BASE_URL)")
        self.model = model.strip()
        self.base_url = base_url.strip().rstrip("/")
        self.max_tokens = max_tokens
        self._api_key = api_key or None
        self._owns_client = http_client is None
        self._http = http_client or httpx.AsyncClient(timeout=timeout)

    def _auth_headers(self) -> dict[str, str]:
        if not self._api_key:
            # Local runtimes (Ollama, LM Studio) need no credentials.
            return {}
        if self.dialect == "anthropic":
            return {
                "x-api-key": self._api_key,
                "anthropic-version": "2023-06-01",
            }
        return {"Authorization": f"Bearer {self._api_key}"}

    def _headers(self, json_mode: bool) -> dict[str, str]:
        headers = {"content-type": "application/json", **self._auth_headers()}
        if json_mode and self.dialect == "openai":
            headers["response_format"] = json.dumps({"type": "json_object"})
        return headers

    async def _post(self, url: str, payload: dict[str, Any], *, json_mode: bool) -> dict[str, Any]:
        headers = self._headers(json_mode)
        if json_mode and self.dialect == "anthropic":
            # Anthropic has no response_format; prefill the assistant turn to force an object.
            payload = {**payload, "messages": _prefill_json_assistant(payload["messages"])}

        logger.info("llm request dialect=%s model=%s url=%s", self.dialect, self.model, url)
        try:
            response = await self._http.post(url, json=payload, headers=headers)
        except httpx.HTTPError as exc:
            raise LLMRequestError(f"could not reach the model at {url}: {exc}") from exc

        if response.status_code >= 400:
            detail = _truncate(response.text, 500)
            raise LLMRequestError(
                f"{self.dialect} endpoint returned HTTP {response.status_code}: {detail}"
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise LLMResponseError(
                f"model response was not JSON: {_truncate(response.text, 200)}"
            ) from exc
        if not isinstance(body, dict):
            raise LLMResponseError(f"expected a JSON object, got {type(body).__name__}")
        return body

    async def aclose(self) -> None:
        if self._owns_client:
            await self._http.aclose()


class OpenAIChatClient(_HttpChatClient):
    """Client for any OpenAI-compatible ``/chat/completions`` endpoint."""

    dialect = "openai"

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> Completion:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.as_openai() for m in messages],
            "max_tokens": max_tokens or self.max_tokens,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        body = await self._post(f"{self.base_url}/chat/completions", payload, json_mode=json_mode)

        try:
            text = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMResponseError(
                f"response had no choices[0].message.content: {_truncate(json.dumps(body), 300)}"
            ) from exc
        if not isinstance(text, str) or not text.strip():
            raise LLMResponseError("model returned empty content")

        usage = body.get("usage") or {}
        return Completion(
            text=text,
            model=str(body.get("model") or self.model),
            prompt_tokens=_as_int(usage.get("prompt_tokens")),
            completion_tokens=_as_int(usage.get("completion_tokens")),
            raw=body,
        )


class AnthropicChatClient(_HttpChatClient):
    """Client for Anthropic's Messages API."""

    dialect = "anthropic"

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> Completion:
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        turns = [m.as_anthropic() for m in messages if m.role != "system"]
        if not turns:
            raise LLMRequestError("at least one user or assistant message is required")

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": turns,
            "max_tokens": max_tokens or self.max_tokens,
        }
        if system:
            payload["system"] = system
        if temperature is not None:
            payload["temperature"] = temperature

        body = await self._post(f"{self.base_url}/messages", payload, json_mode=json_mode)

        blocks = body.get("content")
        if not isinstance(blocks, list) or not blocks:
            raise LLMResponseError("response had no content blocks")
        text = "".join(
            b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text"
        )
        if not text.strip():
            raise LLMResponseError("model returned no text content")

        usage = body.get("usage") or {}
        return Completion(
            text=text,
            model=str(body.get("model") or self.model),
            prompt_tokens=_as_int(usage.get("input_tokens")),
            completion_tokens=_as_int(usage.get("output_tokens")),
            raw=body,
        )


# ---------------------------------------------------------------------- factory


def build_chat_client(settings: Settings) -> ChatClient | None:
    """Create the configured client, or ``None`` when no model is set up.

    Returning ``None`` rather than raising lets the API boot without a key; the step that
    actually needs reasoning then raises ``LLMNotConfiguredError`` and the run fails with a
    message an operator can act on.
    """
    if not settings.has_llm_api_key():
        return None
    key = settings.llm_api_key.get_secret_value().strip() if settings.llm_api_key else None
    factory = AnthropicChatClient if settings.llm_dialect == "anthropic" else OpenAIChatClient
    logger.info(
        "reasoning backend: dialect=%s model=%s base_url=%s key=%s",
        settings.llm_dialect,
        settings.llm_model,
        settings.llm_base_url,
        "set" if key else "none",
    )
    return factory(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=key,
        timeout=settings.llm_timeout,
        max_tokens=settings.llm_max_tokens,
    )


# ---------------------------------------------------------------------- helpers


def _prefill_json_assistant(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Force a JSON object for dialects with no ``response_format``.

    Anthropic allows a trailing assistant turn that the model continues, so an opening brace
    is all it takes to constrain the reply. The turn must come last, and the conversation must
    still start with a user turn.
    """
    if not messages or messages[0].get("role") != "user":
        return messages
    return [*messages, {"role": "assistant", "content": "{"}]


def _truncate(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else f"{text[:limit]}..."


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
