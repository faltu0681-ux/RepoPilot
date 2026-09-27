"""WebSocket endpoint for live run events.

The server speaks JSON text frames. The client sends nothing; everything flows server to client,
which keeps the terminal's job to a single ``onmessage``.

Frame types, discriminated by ``type``:

* ``hello``   - sent once, first, so the client knows the stream is open before anything
  arrives: protocol version, the run subscribed to, and the sequence it is resuming after.
* ``event``   - one :class:`ExecutionEvent`, exactly as stored. Each carries its ``sequence``,
  so the client always knows its own cursor.
* ``status``  - a run changed status, or a run was created.
* ``resync``  - the subscriber fell behind and events were dropped. ``from_sequence`` says where
  the last delivered event ended, so the client refetches
  ``GET /api/runs/{id}/events?after_sequence=`` and continues.
* ``error``   - the stream could not be established.
* ``bye``     - the stream is closing, because the run finished or the server is shutting down.

Connecting is replay-then-stream: ``hello``, then stored history, then live events. That
ordering is what lets the terminal attach to a run mid-flight and still render it whole, and it
is why every event carries a per-run, gap-free ``sequence``.

Keepalive is handled at the transport layer by uvicorn's WebSocket pings, so this module has no
heartbeat frame of its own.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any, Literal

from fastapi import APIRouter, Query, WebSocket
from pydantic import BaseModel

from repopilot.domain.models import TERMINAL_RUN_STATUSES, EventType, ExecutionEvent
from repopilot.domain.store import RunStore
from repopilot.execution.bus import EventBus, RunStatusUpdate, Subscription

logger = logging.getLogger("repopilot.ws")

router = APIRouter()

PROTOCOL_VERSION = 1

#: Replay bound, so attaching to a very long run cannot produce an unbounded first burst.
MAX_REPLAY_EVENTS = 2_000

#: How many runs to replay when subscribed to everything rather than to one run.
MAX_REPLAY_RUNS = 200


class HelloFrame(BaseModel):
    type: Literal["hello"] = "hello"
    version: int = PROTOCOL_VERSION
    run_id: str | None = None
    #: The cursor this stream resumed from. Anything at or below it was not replayed.
    after_sequence: int = 0


class EventFrame(BaseModel):
    type: Literal["event"] = "event"
    event: ExecutionEvent


class StatusFrame(BaseModel):
    type: Literal["status"] = "status"
    run_id: str
    status: str


class ResyncFrame(BaseModel):
    type: Literal["resync"] = "resync"
    run_id: str | None = None
    from_sequence: int = 0
    dropped: int = 0
    reason: str = "subscriber fell behind"


class ErrorFrame(BaseModel):
    type: Literal["error"] = "error"
    code: str = "error"
    detail: str


class ByeFrame(BaseModel):
    type: Literal["bye"] = "bye"
    reason: str


@router.websocket("/ws/events")
async def stream_events(
    websocket: WebSocket,
    run_id: str | None = Query(default=None),
    after_sequence: int = Query(default=0, ge=0),
) -> None:
    """Stream execution events live, replaying stored history first.

    Without ``run_id`` the client sees every run. With it, only that run, and the stream closes
    by itself once that run reaches a terminal status.
    """
    store: RunStore = websocket.app.state.store
    bus: EventBus = websocket.app.state.bus

    if run_id is not None and await store.get_run(run_id) is None:
        await websocket.close(code=4404, reason=f"run {run_id} does not exist")
        return

    await websocket.accept()
    logger.info("websocket open (run_id=%s, after_sequence=%d)", run_id or "*", after_sequence)

    try:
        async with bus.subscribe(run_id=run_id) as subscription:
            await _send(websocket, HelloFrame(run_id=run_id, after_sequence=after_sequence))
            replayed = await _replay(websocket, store, run_id, after_sequence)
            await _pump(websocket, subscription, store, run_id, replayed)
        await websocket.close(code=1000)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - one bad socket must not take down the server
        logger.exception("websocket failed (run_id=%s)", run_id or "*")
        with contextlib.suppress(Exception):
            await _send(websocket, ErrorFrame(code="stream_failed", detail=str(exc)))
            await websocket.close(code=1011, reason="stream failed")
    finally:
        logger.info("websocket closed (run_id=%s)", run_id or "*")


async def _replay(
    websocket: WebSocket, store: RunStore, run_id: str | None, after_sequence: int
) -> int:
    """Send stored history oldest-first. Returns the highest sequence sent."""
    last = after_sequence

    if run_id is not None:
        for event in await store.get_events(
            run_id, after_sequence=after_sequence, limit=MAX_REPLAY_EVENTS
        ):
            await _send(websocket, EventFrame(event=event))
            last = event.sequence
        return last

    for run in await store.list_runs(limit=MAX_REPLAY_RUNS):
        for event in await store.get_events(
            run.id, after_sequence=after_sequence, limit=MAX_REPLAY_EVENTS
        ):
            await _send(websocket, EventFrame(event=event))
            last = max(last, event.sequence)
    return last


async def _pump(
    websocket: WebSocket,
    subscription: Subscription,
    store: RunStore,
    run_id: str | None,
    replayed: int = 0,
) -> None:
    """Forward bus items to the client, closing once the run is over.

    ``replayed`` is the last sequence the replay already sent. The subscription is opened
    *before* the replay runs, so an event published in that window arrives on the queue having
    also been read from history. Skipping it keeps the sequence gap-free, which is what lets a
    client trust that it has seen every event exactly once.
    """
    while True:
        item = await subscription.queue.get()

        if item is None:
            await _send(websocket, ByeFrame(reason="server shutting down"))
            return

        if isinstance(item, RunStatusUpdate):
            await _send(
                websocket,
                StatusFrame(run_id=item.run_id, status=item.status.value),
            )
            if run_id is not None and item.status in TERMINAL_RUN_STATUSES:
                await _send(websocket, ByeFrame(reason=f"run {item.status.value}"))
                return
            continue

        if run_id is not None and item.sequence <= replayed:
            continue

        await _send(websocket, EventFrame(event=item))

        if subscription.dropped:
            dropped, subscription.dropped = subscription.dropped, 0
            logger.warning(
                "dropped %d events for run %s; telling the client to resync",
                dropped,
                run_id or "*",
            )
            await _send(
                websocket,
                ResyncFrame(run_id=run_id, from_sequence=item.sequence, dropped=dropped),
            )

        if run_id is not None and await _run_has_finished(store, run_id, item):
            await _send(websocket, ByeFrame(reason="run finished"))
            return


async def _run_has_finished(store: RunStore, run_id: str, event: ExecutionEvent) -> bool:
    """True when this event was the last one a finished run will ever emit."""
    if event.type is not EventType.RUN_FINISHED:
        return False
    run = await store.get_run(run_id)
    return run is not None and run.is_terminal


async def _send(websocket: WebSocket, frame: Any) -> None:
    await websocket.send_json(frame.model_dump(mode="json"))


__all__ = [
    "PROTOCOL_VERSION",
    "ByeFrame",
    "ErrorFrame",
    "EventFrame",
    "HelloFrame",
    "ResyncFrame",
    "StatusFrame",
    "router",
]
