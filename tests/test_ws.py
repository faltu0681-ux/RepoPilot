"""Tests for the live event WebSocket.

These drive a real socket through Starlette's TestClient against a real app with a stubbed
engine. Replay, live delivery, filtering, resync, and teardown are all covered.

Anything that needs to happen on the app's event loop is scheduled with ``client.portal.call``,
which is how the test thread reaches into the running app.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from repopilot.domain.models import EventType, ExecutionEvent, RunStatus
from repopilot.execution.bus import EventBus
from repopilot.execution.steps import Step, StepContext, StepOutcome

WS = "/ws/events"


class FakeStep(Step):
    name = "fake"

    async def execute(self, ctx: StepContext) -> StepOutcome:
        await ctx.emit(EventType.STEP_STARTED, "working")
        await ctx.emit(EventType.STEP_COMPLETED, "done")
        return StepOutcome.OK


@pytest.fixture
def stub_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the engine's run loop with a synchronous two-event one."""
    from repopilot.execution.engine import RunEngine
    from repopilot.execution.steps import RunServices

    async def fake_start(self, *, repository, task, branch=None, metadata=None):
        run = await self._store.create_run(
            repository=repository, task=task, branch=branch, metadata=metadata
        )
        await self._store.update_run(run.id, total_steps=1)
        await self._emit(run.id, EventType.RUN_CREATED, "run created")
        return await self._store.require_run(run.id)

    async def fake_execute(self, run_id: str) -> None:
        store = self._store
        await store.transition(run_id, RunStatus.RUNNING)
        await self._emit(run_id, EventType.RUN_STARTED, "starting")
        run = await store.update_run(run_id, current_step="fake", current_step_index=0)
        ctx = StepContext(
            run=run,
            store=store,
            services=RunServices(github=self.github, workspace=self._settings.workspace_dir),
            emit=self._make_emitter(run_id, "fake", 0, 1),
        )
        await FakeStep().execute(ctx)
        await store.update_run(run_id, completed_steps=1)
        await self._emit(run_id, EventType.RUN_FINISHED, "run finished")
        await store.transition(run_id, RunStatus.SUCCEEDED)
        await self._publish_status(run_id, RunStatus.SUCCEEDED)

    monkeypatch.setattr(RunEngine, "start", fake_start)
    monkeypatch.setattr(RunEngine, "_execute", fake_execute)


@pytest.fixture
def tiny_queue_client(monkeypatch: pytest.MonkeyPatch, settings) -> TestClient:
    """A client whose event bus drops as soon as a subscriber falls one behind."""
    import repopilot.main as main_module

    monkeypatch.setattr(main_module, "EventBus", lambda: EventBus(queue_size=1))
    with TestClient(main_module.create_app(settings)) as test_client:
        yield test_client


# ---------------------------------------------------------------------- helpers


def drain(ws: Any, count: int) -> list[dict[str, Any]]:
    return [json.loads(ws.receive_text()) for _ in range(count)]


def read_until(ws: Any, matches: Any, limit: int = 30) -> list[dict[str, Any]]:
    """Read frames until one satisfies ``matches``, returning everything read.

    Stream tests must not assume how much replay precedes a live frame, so they wait for the
    frame they actually care about instead of counting.
    """
    out: list[dict[str, Any]] = []
    for _ in range(limit):
        frame = json.loads(ws.receive_text())
        out.append(frame)
        if matches(frame):
            return out
    raise AssertionError(f"no matching frame within {limit} reads; saw {out}")


def of_kind(kind: str) -> Any:
    return lambda frame: frame["type"] == kind


def saying(text: str) -> Any:
    return lambda frame: frame["type"] == "event" and frame["event"]["message"] == text


def only_events(payload: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [f["event"] for f in payload if f["type"] == "event"]


def make_run(client: TestClient, repository: str = "a/b") -> str:
    return client.post("/api/runs", json={"repository": repository, "task": "t"}).json()["id"]


async def publish(
    store: Any,
    bus: EventBus,
    run_id: str,
    message: str,
    event_type: EventType = EventType.LOG,
) -> ExecutionEvent:
    event = await store.append_event(run_id, type=event_type, message=message)
    await bus.publish_event(event)
    return event


async def publish_status(bus: EventBus, run_id: str, status: RunStatus) -> None:
    await bus.publish_status(run_id, status)


async def finish_and_publish(store: Any, bus: EventBus, run_id: str, status: RunStatus) -> None:
    """Move a run to a terminal status, then emit its final event."""
    await terminate(store, run_id, status)
    event = await store.append_event(run_id, type=EventType.RUN_FINISHED, message="run finished")
    await bus.publish_event(event)


async def terminate(store: Any, run_id: str, status: RunStatus) -> None:
    await store.transition(run_id, RunStatus.RUNNING)
    await store.transition(run_id, status)


# ------------------------------------------------------------------------ tests


class TestConnect:
    def test_hello_is_sent_first(self, client: TestClient, stub_engine: None) -> None:
        with client.websocket_connect(WS) as ws:
            hello = json.loads(ws.receive_text())

        assert hello == {
            "type": "hello",
            "version": 1,
            "run_id": None,
            "after_sequence": 0,
        }

    def test_unknown_run_is_closed_with_4404(self, client: TestClient, stub_engine: None) -> None:
        with (
            pytest.raises(WebSocketDisconnect) as excinfo,
            client.websocket_connect(f"{WS}?run_id=run_missing") as ws,
        ):
            ws.receive_text()

        assert excinfo.value.code == 4404

    def test_a_negative_cursor_is_rejected(self, client: TestClient, stub_engine: None) -> None:
        with (
            pytest.raises(WebSocketDisconnect),
            client.websocket_connect(f"{WS}?after_sequence=-1") as ws,
        ):
            ws.receive_text()


class TestReplay:
    def test_history_is_replayed_before_live_events(
        self, client: TestClient, stub_engine: None
    ) -> None:
        run_id = make_run(client)
        client.get(f"/api/runs/{run_id}/events")

        with client.websocket_connect(f"{WS}?run_id={run_id}") as ws:
            payload = drain(ws, 2)

        assert [f["type"] for f in payload] == ["hello", "event"]
        assert payload[1]["event"]["message"] == "run created"
        assert payload[1]["event"]["sequence"] == 1

    def test_after_sequence_skips_what_the_client_already_has(
        self, client: TestClient, stub_engine: None
    ) -> None:
        run_id = make_run(client)
        client.get(f"/api/runs/{run_id}/events")

        with client.websocket_connect(f"{WS}?run_id={run_id}&after_sequence=1") as ws:
            payload = drain(ws, 1)

        assert payload[0] == {
            "type": "hello",
            "version": 1,
            "run_id": run_id,
            "after_sequence": 1,
        }

    def test_subscribing_to_everything_replays_every_run(
        self, client: TestClient, stub_engine: None
    ) -> None:
        first = make_run(client, "a/first")
        second = make_run(client, "a/second")

        with client.websocket_connect(WS) as ws:
            payload = drain(ws, 3)

        assert {e["run_id"] for e in only_events(payload)} == {first, second}

    def test_an_event_published_during_replay_is_not_sent_twice(
        self, client: TestClient, stub_engine: None
    ) -> None:
        """The subscription opens before the replay, so the window can double-deliver."""
        run_id = make_run(client)
        store, bus = client.app.state.store, client.app.state.bus
        client.portal.call(publish, store, bus, run_id, "during replay")
        late = client.portal.call(publish, store, bus, run_id, "after replay")

        with client.websocket_connect(f"{WS}?run_id={run_id}") as ws:
            client.portal.call(publish, store, bus, run_id, "live")
            payload = read_until(ws, saying("live"))

        messages = [e["message"] for e in only_events(payload)]
        assert messages.count("during replay") == 1
        assert messages.count("after replay") == 1
        assert only_events(payload)[-1]["sequence"] == late.sequence + 1
        assert "bye" not in {f["type"] for f in payload}


class TestLiveDelivery:
    def test_events_arrive_as_they_happen(self, client: TestClient, stub_engine: None) -> None:
        run_id = make_run(client)

        with client.websocket_connect(f"{WS}?run_id={run_id}") as ws:
            client.portal.call(
                publish,
                client.app.state.store,
                client.app.state.bus,
                run_id,
                "streaming now",
            )
            payload = read_until(ws, saying("streaming now"))

        assert payload[0]["type"] == "hello"
        assert only_events(payload)[-1]["sequence"] == 2

    def test_a_run_scoped_socket_ignores_other_runs(
        self, client: TestClient, stub_engine: None
    ) -> None:
        run_id = make_run(client)
        other_id = make_run(client, "a/other")
        store, bus = client.app.state.store, client.app.state.bus

        with client.websocket_connect(f"{WS}?run_id={run_id}") as ws:
            client.portal.call(publish, store, bus, other_id, "not for you")
            client.portal.call(publish, store, bus, run_id, "for my run only")
            payload = read_until(ws, saying("for my run only"))

        run_ids = {e["run_id"] for e in only_events(payload)}
        assert run_ids == {run_id}

    def test_status_frames_are_pushed(self, client: TestClient, stub_engine: None) -> None:
        with client.websocket_connect(WS) as ws:
            client.portal.call(publish_status, client.app.state.bus, "run_x", RunStatus.FAILED)
            payload = read_until(ws, of_kind("status"))

        assert payload[-1] == {
            "type": "status",
            "run_id": "run_x",
            "status": "failed",
        }

    def test_the_stream_closes_when_the_run_finishes(
        self, client: TestClient, stub_engine: None
    ) -> None:
        run_id = make_run(client)

        with client.websocket_connect(f"{WS}?run_id={run_id}") as ws:
            client.portal.call(
                finish_and_publish,
                client.app.state.store,
                client.app.state.bus,
                run_id,
                RunStatus.SUCCEEDED,
            )
            payload = read_until(ws, of_kind("bye"))

        assert payload[-2]["type"] == "event"
        assert payload[-2]["event"]["type"] == "run.finished"
        assert payload[-1] == {"type": "bye", "reason": "run finished"}

    def test_a_terminal_status_also_closes_a_scoped_stream(
        self, client: TestClient, stub_engine: None
    ) -> None:
        run_id = make_run(client)
        store, bus = client.app.state.store, client.app.state.bus

        with client.websocket_connect(f"{WS}?run_id={run_id}") as ws:
            client.portal.call(terminate, store, run_id, RunStatus.FAILED)
            client.portal.call(publish_status, bus, run_id, RunStatus.FAILED)
            payload = read_until(ws, of_kind("bye"))

        assert payload[-2] == {
            "type": "status",
            "run_id": run_id,
            "status": "failed",
        }
        assert payload[-1] == {"type": "bye", "reason": "run failed"}

    def test_an_unscoped_socket_stays_open_after_a_run_finishes(
        self, client: TestClient, stub_engine: None
    ) -> None:
        run_id = make_run(client)

        with client.websocket_connect(WS) as ws:
            client.portal.call(
                finish_and_publish,
                client.app.state.store,
                client.app.state.bus,
                run_id,
                RunStatus.SUCCEEDED,
            )
            client.portal.call(publish_status, client.app.state.bus, run_id, RunStatus.SUCCEEDED)
            payload = read_until(ws, of_kind("status"))

        # A bye would have arrived before the status if the socket had closed.
        assert [f["type"] for f in payload] == ["hello", "event", "event", "status"]
        assert payload[-1]["status"] == "succeeded"

    def test_the_whole_run_story_arrives_in_order(
        self, client: TestClient, stub_engine: None
    ) -> None:
        run_id = make_run(client)

        with client.websocket_connect(f"{WS}?run_id={run_id}") as ws:
            client.portal.call(
                _run_to_completion,
                client.app.state.store,
                client.app.state.bus,
                client.app.state,
                run_id,
            )
            payload = read_until(ws, saying("run finished"))

        events = only_events(payload)
        assert [e["type"] for e in events] == [
            "run.created",
            "run.started",
            "step.started",
            "step.completed",
            "run.finished",
        ]
        assert [e["sequence"] for e in events] == [1, 2, 3, 4, 5]


async def _run_to_completion(store: Any, bus: EventBus, app_state: Any, run_id: str) -> None:
    """Replay the stub engine's pipeline through the store and the bus."""
    from repopilot.execution.steps import RunServices

    engine = app_state.engine
    await store.transition(run_id, RunStatus.RUNNING)
    await engine._emit(run_id, EventType.RUN_STARTED, "starting")
    run = await store.update_run(run_id, current_step="fake", current_step_index=0)
    ctx = StepContext(
        run=run,
        store=store,
        services=RunServices(github=engine.github, workspace=engine._settings.workspace_dir),
        emit=engine._make_emitter(run_id, "fake", 0, 1),
    )
    await FakeStep().execute(ctx)
    await store.update_run(run_id, completed_steps=1)
    await store.transition(run_id, RunStatus.SUCCEEDED)
    await engine._emit(run_id, EventType.RUN_FINISHED, "run finished")


class TestResync:
    def test_dropped_events_trigger_a_resync_frame(
        self, tiny_queue_client: TestClient, stub_engine: None
    ) -> None:
        run_id = make_run(tiny_queue_client)

        with tiny_queue_client.websocket_connect(f"{WS}?run_id={run_id}") as ws:
            tiny_queue_client.portal.call(
                _flood,
                tiny_queue_client.app.state.store,
                tiny_queue_client.app.state.bus,
                run_id,
                5,
            )
            payload = read_until(ws, of_kind("resync"))

        resync = payload[-1]
        assert resync["dropped"] > 0
        assert resync["from_sequence"] >= 1
        assert resync["run_id"] == run_id
        assert resync["reason"] == "subscriber fell behind"


async def _flood(store: Any, bus: EventBus, run_id: str, count: int) -> None:
    for i in range(count):
        event = await store.append_event(run_id, type=EventType.LOG, message=f"line {i}")
        await bus.publish_event(event)


class TestTeardown:
    def test_bus_shutdown_says_bye_then_disconnects(
        self, client: TestClient, stub_engine: None
    ) -> None:
        with pytest.raises(WebSocketDisconnect), client.websocket_connect(WS) as ws:
            client.portal.call(client.app.state.bus.shutdown)
            payload = read_until(ws, of_kind("bye"))
            assert payload[-1] == {
                "type": "bye",
                "reason": "server shutting down",
            }
            ws.receive_text()

    def test_a_finished_run_closes_the_socket(self, client: TestClient, stub_engine: None) -> None:
        run_id = make_run(client)

        with (
            pytest.raises(WebSocketDisconnect),
            client.websocket_connect(f"{WS}?run_id={run_id}") as ws,
        ):
            client.portal.call(
                finish_and_publish,
                client.app.state.store,
                client.app.state.bus,
                run_id,
                RunStatus.SUCCEEDED,
            )
            read_until(ws, of_kind("bye"))
            ws.receive_text()
