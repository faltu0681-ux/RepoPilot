"""Tests for the reasoning layer.

The HTTP clients are driven through ``httpx.MockTransport`` so the real request and response
shapes are asserted, and ``generate_plan`` runs against a real client over that transport, so
the JSON extraction, validation, retry, and hallucination-filtering paths are all genuinely
exercised. Nothing here fabricates a "plan" directly: every plan in this file arrives as bytes
over an HTTP response body.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from repopilot.agent.llm import (
    AnthropicChatClient,
    ChatMessage,
    Completion,
    LLMRequestError,
    LLMResponseError,
    OpenAIChatClient,
    build_chat_client,
    extract_json_object,
)
from repopilot.agent.plan import PlanRejectedError, generate_plan
from repopilot.agent.schema import ImplementationPlan
from repopilot.analysis.profile import build_profile
from repopilot.analysis.relevance import rank_candidates
from repopilot.config import Settings

# --------------------------------------------------------------- JSON extraction


class TestExtractJsonObject:
    def test_parses_a_bare_object(self) -> None:
        assert extract_json_object('{"a": 1}') == {"a": 1}

    def test_strips_a_code_fence(self) -> None:
        reply = '```json\n{"summary": "ok", "steps": []}\n```'

        assert extract_json_object(reply) == {"summary": "ok", "steps": []}

    def test_recovers_an_object_wrapped_in_prose(self) -> None:
        reply = 'Here is the plan you asked for:\n{"summary": "ok"}\nHope that helps!'

        assert extract_json_object(reply) == {"summary": "ok"}

    def test_takes_the_first_balanced_object(self) -> None:
        reply = 'prose {"a": {"b": 1}} trailing'

        assert extract_json_object(reply) == {"a": {"b": 1}}

    def test_rejects_an_array(self) -> None:
        with pytest.raises(LLMResponseError, match="array"):
            extract_json_object("[1, 2, 3]")

    def test_rejects_an_empty_reply(self) -> None:
        with pytest.raises(LLMResponseError, match="empty"):
            extract_json_object("   ")

    def test_rejects_prose_with_no_object(self) -> None:
        with pytest.raises(LLMResponseError, match="could not find"):
            extract_json_object("I cannot help with that.")


# ------------------------------------------------------------------ HTTP clients


def openai_body(content: str, **extra: object) -> dict:
    return {
        "model": "test-model",
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7},
        **extra,
    }


def openai_client(handler, **kwargs) -> OpenAIChatClient:
    return OpenAIChatClient(
        model="test-model",
        base_url="https://model.invalid/v1",
        api_key="secret-key",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        **kwargs,
    )


class TestOpenAIChatClient:
    async def test_posts_a_real_chat_completions_request(self) -> None:
        seen: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["auth"] = request.headers.get("authorization")
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=openai_body("hello"))

        client = openai_client(handler)
        completion = await client.complete(
            [ChatMessage(role="system", content="sys"), ChatMessage(role="user", content="hi")]
        )

        assert seen["url"] == "https://model.invalid/v1/chat/completions"
        assert seen["auth"] == "Bearer secret-key"
        assert seen["body"]["model"] == "test-model"
        assert seen["body"]["messages"] == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ]
        assert completion.text == "hello"
        assert completion.total_tokens == 18
        await client.aclose()

    async def test_asks_for_json_when_json_mode_is_set(self) -> None:
        seen: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["format"] = request.headers.get("response_format")
            return httpx.Response(200, json=openai_body("{}"))

        client = openai_client(handler)
        await client.complete([ChatMessage(role="user", content="hi")], json_mode=True)

        assert json.loads(seen["format"]) == {"type": "json_object"}
        await client.aclose()

    async def test_sends_no_authorization_header_without_a_key(self) -> None:
        """Local runtimes such as Ollama need no credentials."""
        seen: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["auth"] = request.headers.get("authorization")
            return httpx.Response(200, json=openai_body("ok"))

        client = OpenAIChatClient(
            model="llama3",
            base_url="http://127.0.0.1:11434/v1",
            api_key=None,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        await client.complete([ChatMessage(role="user", content="hi")])

        assert seen["auth"] is None
        await client.aclose()

    async def test_reports_a_provider_error_with_its_status(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, text="invalid api key")

        client = openai_client(handler)
        with pytest.raises(LLMRequestError, match="HTTP 401"):
            await client.complete([ChatMessage(role="user", content="hi")])
        await client.aclose()

    async def test_rejects_a_response_with_no_choices(self) -> None:
        client = openai_client(lambda r: httpx.Response(200, json={"error": "nope"}))

        with pytest.raises(LLMResponseError, match="no choices"):
            await client.complete([ChatMessage(role="user", content="hi")])
        await client.aclose()

    async def test_rejects_empty_content(self) -> None:
        client = openai_client(lambda r: httpx.Response(200, json=openai_body("   ")))

        with pytest.raises(LLMResponseError, match="empty content"):
            await client.complete([ChatMessage(role="user", content="hi")])
        await client.aclose()

    async def test_rejects_a_non_json_body(self) -> None:
        client = openai_client(lambda r: httpx.Response(200, text="<html>hi</html>"))

        with pytest.raises(LLMResponseError, match="not JSON"):
            await client.complete([ChatMessage(role="user", content="hi")])
        await client.aclose()

    async def test_wraps_transport_errors(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        client = openai_client(handler)
        with pytest.raises(LLMRequestError, match="could not reach"):
            await client.complete([ChatMessage(role="user", content="hi")])
        await client.aclose()

    def test_refuses_to_build_without_a_model(self) -> None:
        with pytest.raises(Exception, match="no model configured"):
            OpenAIChatClient(model="  ", base_url="https://x/v1", api_key=None)


def anthropic_body(text: str) -> dict:
    return {
        "model": "claude-x",
        "content": [{"type": "text", "text": text}],
        "usage": {"input_tokens": 5, "output_tokens": 9},
    }


class TestAnthropicChatClient:
    async def test_lifts_the_system_prompt_out_of_the_message_list(self) -> None:
        seen: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["body"] = json.loads(request.content)
            seen["version"] = request.headers.get("anthropic-version")
            seen["key"] = request.headers.get("x-api-key")
            return httpx.Response(200, json=anthropic_body("ok"))

        client = AnthropicChatClient(
            model="claude-x",
            base_url="https://api.anthropic.invalid",
            api_key="sk-ant",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        completion = await client.complete(
            [ChatMessage(role="system", content="sys"), ChatMessage(role="user", content="hi")]
        )

        assert seen["url"] == "https://api.anthropic.invalid/messages"
        assert seen["body"]["system"] == "sys"
        assert seen["body"]["messages"] == [{"role": "user", "content": "hi"}]
        assert seen["version"] == "2023-06-01"
        assert seen["key"] == "sk-ant"
        assert completion.prompt_tokens == 5
        assert completion.completion_tokens == 9
        await client.aclose()

    async def test_prefills_a_trailing_brace_for_json_mode(self) -> None:
        seen: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["messages"] = json.loads(request.content)["messages"]
            return httpx.Response(200, json=anthropic_body('"summary": "ok"}'))

        client = AnthropicChatClient(
            model="claude-x",
            base_url="https://api.anthropic.invalid",
            api_key="sk-ant",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        await client.complete([ChatMessage(role="user", content="hi")], json_mode=True)

        messages = seen["messages"]
        assert messages[0]["role"] == "user"
        assert messages[-1] == {"role": "assistant", "content": "{"}
        await client.aclose()

    async def test_requires_a_user_turn(self) -> None:
        client = AnthropicChatClient(
            model="claude-x",
            base_url="https://api.anthropic.invalid",
            api_key="sk-ant",
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(lambda r: httpx.Response(200))
            ),
        )

        with pytest.raises(LLMRequestError, match="at least one"):
            await client.complete([ChatMessage(role="system", content="only system")])
        await client.aclose()

    async def test_joins_multiple_text_blocks(self) -> None:
        body = {"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}
        client = AnthropicChatClient(
            model="claude-x",
            base_url="https://api.anthropic.invalid",
            api_key="sk-ant",
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(lambda r: httpx.Response(200, json=body))
            ),
        )

        completion = await client.complete([ChatMessage(role="user", content="hi")])

        assert completion.text == "ab"
        await client.aclose()


class TestBuildChatClient:
    def test_returns_none_when_no_model_is_configured(self) -> None:
        """The out-of-the-box state: no model, no base URL, therefore no client and no call."""
        assert build_chat_client(Settings(_env_file=None)) is None

    def test_returns_none_when_only_a_model_is_set(self) -> None:
        assert build_chat_client(Settings(_env_file=None, llm_model="gpt-4o-mini")) is None

    def test_builds_an_openai_client_by_default(self) -> None:
        client = build_chat_client(
            Settings(
                _env_file=None,
                llm_model="gpt-4o-mini",
                llm_base_url="https://api.openai.com/v1",
            )
        )

        assert isinstance(client, OpenAIChatClient)
        assert client.model == "gpt-4o-mini"

    def test_builds_an_anthropic_client_for_that_dialect(self) -> None:
        client = build_chat_client(
            Settings(
                _env_file=None,
                llm_dialect="anthropic",
                llm_model="claude-x",
                llm_base_url="https://api.anthropic.com",
            )
        )

        assert isinstance(client, AnthropicChatClient)

    def test_rejects_an_unknown_dialect(self) -> None:
        with pytest.raises(Exception, match="unsupported llm_dialect"):
            Settings(_env_file=None, llm_dialect="gemini")


# ------------------------------------------------------------- plan generation


PLAN_PAYLOAD = {
    "summary": "Add a health endpoint.",
    "approach": "Add a route next to the existing version route.",
    "relevant_files": [
        {"path": "src/app/main.py", "reason": "defines the router", "role": "modify"},
        {"path": "tests/test_health.py", "reason": "where the test belongs", "role": "add"},
    ],
    "steps": [
        {
            "order": 1,
            "title": "Add the route",
            "detail": "Define GET /api/health returning status ok.",
            "files": ["src/app/main.py"],
            "rationale": "The router lives here.",
        }
    ],
    "risks": [],
    "test_strategy": "Add a test asserting 200 and the status field.",
}


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    (root / "src" / "app").mkdir(parents=True)
    (root / "src" / "app" / "main.py").write_text(
        "from fastapi import FastAPI\n\napp = FastAPI()\n", encoding="utf-8"
    )
    return root


def plan_client(replies: list[str]) -> tuple[OpenAIChatClient, list[dict]]:
    """A real client over a transport that replays canned HTTP bodies in order."""
    bodies: list[dict] = []
    queue = list(replies)

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        reply = queue.pop(0) if queue else "{}"
        return httpx.Response(200, json=openai_body(reply))

    client = OpenAIChatClient(
        model="test-model",
        base_url="https://model.invalid/v1",
        api_key="k",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return client, bodies


async def make_plan(
    client: OpenAIChatClient,
    repo: Path,
    task: str = "add a health route to the FastAPI app in main.py",
):
    profile = build_profile(repo, default_branch="main")
    candidates = rank_candidates(profile, task, root=repo)
    return await generate_plan(
        client,
        task=task,
        repository="acme/app",
        profile=profile,
        candidates=candidates,
        default_branch="main",
        root=repo,
    )


class TestGeneratePlan:
    async def test_returns_a_validated_plan(self, repo: Path) -> None:
        client, _ = plan_client([json.dumps(PLAN_PAYLOAD)])

        plan, dropped = await make_plan(client, repo)

        assert dropped == []
        assert plan.summary == "Add a health endpoint."
        assert [s.title for s in plan.steps] == ["Add the route"]
        assert plan.files_to_change == ["src/app/main.py", "tests/test_health.py"]
        assert plan.model == "test-model"
        assert plan.prompt_tokens == 11
        await client.aclose()

    async def test_prompt_grounds_the_model_in_real_paths(self, repo: Path) -> None:
        client, bodies = plan_client([json.dumps(PLAN_PAYLOAD)])

        await make_plan(client, repo)

        user_message = bodies[0]["messages"][-1]["content"]
        assert "src/app/main.py" in user_message
        assert "from fastapi import FastAPI" in user_message
        assert "Primary language: Python" in user_message
        await client.aclose()

    async def test_keeps_a_file_the_plan_intends_to_create(self, repo: Path) -> None:
        """A path with role "add" is meant to be new, so it must survive validation."""
        client, _ = plan_client([json.dumps(PLAN_PAYLOAD)])

        plan, dropped = await make_plan(client, repo)

        assert dropped == []
        assert "tests/test_health.py" in plan.files_to_change
        assert not (repo / "tests" / "test_health.py").exists()
        await client.aclose()

    async def test_drops_paths_the_model_invented(self, repo: Path) -> None:
        """A "modify" path that does not exist is a hallucination and is removed."""
        payload = {
            **PLAN_PAYLOAD,
            "relevant_files": [
                *PLAN_PAYLOAD["relevant_files"],
                {"path": "src/app/does_not_exist.py", "reason": "invented", "role": "modify"},
            ],
            "steps": [
                {
                    "order": 1,
                    "title": "Add the route",
                    "detail": "x",
                    "files": ["src/app/main.py", "src/app/does_not_exist.py"],
                    "rationale": "y",
                }
            ],
        }
        client, _ = plan_client([json.dumps(payload)])

        plan, dropped = await make_plan(client, repo)

        assert dropped == ["src/app/does_not_exist.py"]
        assert "src/app/does_not_exist.py" not in plan.files_to_change
        assert plan.steps[0].files == ["src/app/main.py"]
        assert any("does_not_exist.py" in risk for risk in plan.risks)
        await client.aclose()

    async def test_retries_once_when_the_reply_is_not_json(self, repo: Path) -> None:
        client, bodies = plan_client(["I refuse", json.dumps(PLAN_PAYLOAD)])

        plan, _ = await make_plan(client, repo)

        assert plan.summary == "Add a health endpoint."
        assert len(bodies) == 2
        # The second request shows the model its own bad reply and asks for a correction.
        assert "not valid JSON" in bodies[1]["messages"][-1]["content"]
        await client.aclose()

    async def test_retries_once_when_the_shape_is_wrong(self, repo: Path) -> None:
        client, bodies = plan_client([json.dumps({"summary": 123}), json.dumps(PLAN_PAYLOAD)])

        plan, _ = await make_plan(client, repo)

        assert plan.summary == "Add a health endpoint."
        assert len(bodies) == 2
        assert "did not match the required shape" in bodies[1]["messages"][-1]["content"]
        await client.aclose()

    async def test_gives_up_after_the_retry_limit(self, repo: Path) -> None:
        client, bodies = plan_client(["nope", "still nope"])

        with pytest.raises(PlanRejectedError, match="did not return a valid implementation plan"):
            await make_plan(client, repo)

        assert len(bodies) == 2
        await client.aclose()

    async def test_does_not_retry_a_transport_failure(self, repo: Path) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        client = OpenAIChatClient(
            model="m",
            base_url="https://model.invalid/v1",
            api_key="k",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )

        # A dead endpoint will not fix itself; asking again would only double the wait.
        with pytest.raises(LLMRequestError):
            await make_plan(client, repo)
        await client.aclose()

    async def test_refuses_to_plan_with_no_candidates(self, repo: Path) -> None:
        client, _ = plan_client([json.dumps(PLAN_PAYLOAD)])
        profile = build_profile(repo)

        with pytest.raises(PlanRejectedError, match="no files in this repository look related"):
            await generate_plan(
                client,
                task="a",
                repository="acme/app",
                profile=profile,
                candidates=[],
                root=repo,
            )
        await client.aclose()


class TestImplementationPlan:
    def test_renumbers_steps_densely(self) -> None:
        plan = ImplementationPlan.model_validate(
            {
                "summary": "s",
                "steps": [
                    {"order": 7, "title": "first"},
                    {"order": 99, "title": "second"},
                ],
            }
        )

        assert [s.order for s in plan.steps] == [1, 2]

    def test_files_to_change_excludes_read_only_files(self) -> None:
        plan = ImplementationPlan.model_validate(
            {
                "summary": "s",
                "relevant_files": [
                    {"path": "a.py", "role": "modify"},
                    {"path": "b.py", "role": "read"},
                    {"path": "c.py", "role": "add"},
                ],
            }
        )

        assert plan.files_to_change == ["a.py", "c.py"]

    def test_rejects_unknown_fields_so_a_typo_is_not_silently_dropped(self) -> None:
        with pytest.raises(Exception, match="summary|messag"):
            ImplementationPlan.model_validate({"summary": "s", "messag": "typo"})

    def test_serialises_for_the_event_stream(self) -> None:
        plan = ImplementationPlan.model_validate(PLAN_PAYLOAD)

        payload = plan.as_dict()

        assert payload["steps"][0]["order"] == 1
        assert payload["relevant_files"][0]["path"] == "src/app/main.py"
        assert payload["files_to_change"] == ["src/app/main.py", "tests/test_health.py"]


class TestCompletion:
    def test_total_tokens_is_none_when_the_provider_reports_nothing(self) -> None:
        assert Completion(text="x").total_tokens is None
