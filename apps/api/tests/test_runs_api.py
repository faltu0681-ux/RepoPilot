"""Tests for the run HTTP surface.

The engine is replaced with a stub so these stay fast and deterministic; the engine's own
behaviour is covered in ``test_engine.py``.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from repopilot.domain.models import EventLevel, EventType, RunStatus
from repopilot.execution.steps import Step, StepContext, StepOutcome


class EchoStep(Step):
    """Emits a couple of events and succeeds."""

    name = "echo"

    async def execute(self, ctx: StepContext) -> StepOutcome:
        await ctx.emit(EventType.STEP_STARTED, "echoing")
        await ctx.emit(EventType.LOG, "a quiet detail", level=EventLevel.DEBUG)
        await ctx.emit(EventType.STEP_COMPLETED, "echoed")
        return StepOutcome.OK


@pytest.fixture
def stub_engine(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Swap the engine for one that records its calls and runs a trivial pipeline."""
    calls: dict[str, Any] = {"started": [], "cancelled": []}

    from repopilot.execution.engine import RunEngine

    async def fake_start(self, *, repository, task, branch=None, metadata=None):
        calls["started"].append(
            {"repository": repository, "task": task, "branch": branch, "metadata": metadata}
        )
        run = await self._store.create_run(
            repository=repository, task=task, branch=branch, metadata=metadata
        )
        await self._store.update_run(run.id, total_steps=1)
        await self._store.append_event(run.id, type=EventType.RUN_CREATED, message="run created")
        for line in (metadata or {}).get("seed_events", []):
            await self._store.append_event(run.id, type=EventType.LOG, message=line)
        await self._store.transition(run.id, RunStatus.RUNNING)
        return await self._store.require_run(run.id)

    async def fake_cancel(self, run_id):
        calls["cancelled"].append(run_id)
        return await self._store.transition(run_id, RunStatus.CANCELLED, error="cancelled")

    monkeypatch.setattr(RunEngine, "start", fake_start)
    monkeypatch.setattr(RunEngine, "cancel", fake_cancel)
    return calls


class TestCreateRun:
    def test_creates_a_run(self, client: TestClient, stub_engine: dict) -> None:
        response = client.post(
            "/api/runs",
            json={"repository": "pallets/flask", "task": "Add a health check"},
        )

        assert response.status_code == 201
        body = response.json()
        assert body["repository"] == "pallets/flask"
        assert body["task"] == "Add a health check"
        assert body["status"] in {"pending", "running"}
        assert body["id"].startswith("run_")
        assert body["branch"].startswith("repopilot/")

    def test_accepts_a_github_url(self, client: TestClient, stub_engine: dict) -> None:
        response = client.post(
            "/api/runs",
            json={
                "repository": "https://github.com/pallets/flask",
                "task": "Add a health check",
            },
        )

        assert response.status_code == 201
        assert response.json()["repository"] == "pallets/flask"

    def test_accepts_an_explicit_branch(self, client: TestClient, stub_engine: dict) -> None:
        response = client.post(
            "/api/runs",
            json={
                "repository": "pallets/flask",
                "task": "t",
                "branch": "feat/my-work",
            },
        )

        assert response.status_code == 201
        assert response.json()["branch"] == "feat/my-work"
        assert stub_engine["started"][0]["branch"] == "feat/my-work"

    def test_passes_metadata_through(self, client: TestClient, stub_engine: dict) -> None:
        response = client.post(
            "/api/runs",
            json={"repository": "a/b", "task": "t", "metadata": {"source": "cli"}},
        )

        assert response.status_code == 201
        assert response.json()["metadata"] == {"source": "cli"}

    @pytest.mark.parametrize(
        "payload",
        [
            {"task": "t"},
            {"repository": "a/b"},
            {"repository": "", "task": "t"},
            {"repository": "not-a-repo", "task": "t"},
            {"repository": "a/b", "task": "   "},
        ],
    )
    def test_rejects_bad_payloads(
        self, client: TestClient, stub_engine: dict, payload: dict
    ) -> None:
        assert client.post("/api/runs", json=payload).status_code == 422

    def test_requires_a_body(self, client: TestClient, stub_engine: dict) -> None:
        assert client.post("/api/runs").status_code == 422

    def test_does_not_leak_internal_fields(self, client: TestClient, stub_engine: dict) -> None:
        body = client.post("/api/runs", json={"repository": "a/b", "task": "t"}).json()

        assert "model_config" not in body
        assert "model_fields" not in body
        assert set(body) >= {"id", "repository", "task", "branch", "status", "created_at"}


class TestListRuns:
    def test_lists_newest_first(self, client: TestClient, stub_engine: dict) -> None:
        first = client.post("/api/runs", json={"repository": "a/first", "task": "t"}).json()
        second = client.post("/api/runs", json={"repository": "a/second", "task": "t"}).json()

        body = client.get("/api/runs").json()

        assert [r["id"] for r in body["runs"]] == [second["id"], first["id"]]
        assert body["total"] == 2

    def test_filters_by_status(self, client: TestClient, stub_engine: dict) -> None:
        client.post("/api/runs", json={"repository": "a/b", "task": "t"})

        assert client.get("/api/runs?status=running").json()["total"] == 1
        assert client.get("/api/runs?status=pending").json()["total"] == 0
        assert client.get("/api/runs?status=nonsense").status_code == 422

    def test_honours_limit(self, client: TestClient, stub_engine: dict) -> None:
        for i in range(5):
            client.post("/api/runs", json={"repository": "a/b", "task": f"task {i}"})

        assert len(client.get("/api/runs?limit=2").json()["runs"]) == 2
        assert client.get("/api/runs?limit=0").status_code == 422
        assert client.get("/api/runs?limit=9999").status_code == 422

    def test_empty_by_default(self, client: TestClient) -> None:
        body = client.get("/api/runs").json()
        assert body == {"runs": [], "total": 0}


class TestGetRun:
    def test_fetches_one_run(self, client: TestClient, stub_engine: dict) -> None:
        created = client.post("/api/runs", json={"repository": "a/b", "task": "t"}).json()

        body = client.get(f"/api/runs/{created['id']}").json()

        assert body["id"] == created["id"]

    def test_unknown_run_is_404(self, client: TestClient) -> None:
        response = client.get("/api/runs/run_missing")

        assert response.status_code == 404
        assert "run_missing" in response.json()["detail"]


class TestEvents:
    def test_lists_a_runs_events(self, client: TestClient, stub_engine: dict) -> None:
        created = client.post(
            "/api/runs",
            json={
                "repository": "a/b",
                "task": "t",
                "metadata": {"seed_events": ["line 0", "line 1", "line 2"]},
            },
        ).json()

        body = client.get(f"/api/runs/{created['id']}/events").json()

        assert [e["message"] for e in body["events"]] == [
            "run created",
            "line 0",
            "line 1",
            "line 2",
        ]
        assert [e["sequence"] for e in body["events"]] == [1, 2, 3, 4]
        assert body["last_sequence"] == 4

    def test_after_sequence_acts_as_a_cursor(self, client: TestClient, stub_engine: dict) -> None:
        created = client.post(
            "/api/runs",
            json={
                "repository": "a/b",
                "task": "t",
                "metadata": {"seed_events": ["line 0", "line 1", "line 2"]},
            },
        ).json()

        body = client.get(f"/api/runs/{created['id']}/events?after_sequence=2").json()

        assert [e["sequence"] for e in body["events"]] == [3, 4]
        assert body["last_sequence"] == 4

    def test_an_empty_stream_reports_the_cursor(
        self, client: TestClient, stub_engine: dict
    ) -> None:
        created = client.post("/api/runs", json={"repository": "a/b", "task": "t"}).json()

        body = client.get(f"/api/runs/{created['id']}/events?after_sequence=99").json()

        assert body["events"] == []
        assert body["last_sequence"] == 99

    def test_unknown_run_is_404(self, client: TestClient) -> None:
        assert client.get("/api/runs/run_missing/events").status_code == 404


class TestCancelRun:
    def test_cancels_a_run(self, client: TestClient, stub_engine: dict) -> None:
        created = client.post("/api/runs", json={"repository": "a/b", "task": "t"}).json()

        body = client.delete(f"/api/runs/{created['id']}").json()

        assert body["status"] == "cancelled"
        assert body["error"] == "cancelled"
        assert stub_engine["cancelled"] == [created["id"]]

    def test_unknown_run_is_404(self, client: TestClient) -> None:
        assert client.delete("/api/runs/run_missing").status_code == 404


class TestOpenApi:
    def test_run_routes_are_documented(self, client: TestClient) -> None:
        paths = client.get("/openapi.json").json()["paths"]

        assert "/api/runs" in paths
        assert "/api/runs/{run_id}" in paths
        assert "/api/runs/{run_id}/events" in paths
        assert "post" in paths["/api/runs"]
        assert "delete" in paths["/api/runs/{run_id}"]


class TestRunSerialisation:
    def test_progress_is_serialised(self) -> None:
        from repopilot.domain.models import Run

        run = Run(repository="a/b", task="t", total_steps=4, completed_steps=2)
        assert run.model_dump()["progress"] == 0.5

    def test_duration_is_serialised_when_finished(self) -> None:
        from repopilot.core.time import utcnow
        from repopilot.domain.models import Run

        run = Run(repository="a/b", task="t")
        run.started_at = utcnow()
        run.finished_at = utcnow()
        assert run.model_dump()["duration_seconds"] >= 0

    def test_run_carries_the_phase_one_pipeline(self, client: TestClient) -> None:
        created = client.post("/api/runs", json={"repository": "a/b", "task": "t"}).json()
        events = client.get(f"/api/runs/{created['id']}/events").json()["events"]
        assert events[0]["type"] == "run.created"
