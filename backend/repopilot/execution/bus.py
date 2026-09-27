"""The event bus.

The engine publishes to this; the WebSocket endpoint and anything else that wants to observe a
run subscribes to it. Subscribers get their own bounded queue and are never allowed to block or
fail the engine, because a slow consumer must not stall a run.

Delivery contract: at-most-once, best effort. Nothing is lost that matters, because every event
is also written to the :class:`~repopilot.domain.store.RunStore`, which is the durable record.
The bus is only the fast path that makes the terminal feel live.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from repopilot.domain.models import ExecutionEvent, RunStatus

logger = logging.getLogger("repopilot.bus")

#: Per-subscriber backlog. When a consumer falls this far behind further events are dropped
#: rather than allowed to consume unbounded memory; it recovers by replaying from the store.
DEFAULT_QUEUE_SIZE = 1_000


class Subscription:
    """One consumer's view of the bus."""

    __slots__ = ("run_id", "queue", "dropped")

    def __init__(self, *, run_id: str | None = None, maxsize: int = DEFAULT_QUEUE_SIZE) -> None:
        self.run_id = run_id
        self.queue: asyncio.Queue[ExecutionEvent | RunStatus | None] = asyncio.Queue(
            maxsize=maxsize
        )
        self.dropped = 0

    def __repr__(self) -> str:
        return (
            f"<Subscription run_id={self.run_id!r} "
            f"queued={self.queue.qsize()} dropped={self.dropped}>"
        )

    def __aiter__(self) -> AsyncIterator[ExecutionEvent | RunStatus]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[ExecutionEvent | RunStatus]:
        while True:
            item = await self.queue.get()
            if item is None:
                return
            yield item

    async def offer(self, item: ExecutionEvent | RunStatus) -> bool:
        """Enqueue without blocking. Returns False if the subscriber is too far behind."""
        try:
            self.queue.put_nowait(item)
        except asyncio.QueueFull:
            self.dropped += 1
            logger.warning(
                "subscriber for run %s is %d events behind; dropping",
                self.run_id,
                self.dropped,
            )
            return False
        return True


class EventBus:
    """Fan-out of execution events and run status changes to live subscribers."""

    def __init__(self, queue_size: int = DEFAULT_QUEUE_SIZE) -> None:
        self._queue_size = queue_size
        self._subscriptions: set[Subscription] = set()
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def subscribe(self, *, run_id: str | None = None) -> AsyncIterator[Subscription]:
        """Subscribe to every event, or only those for one run.

        The ``None`` sentinel is pushed on exit so a blocked consumer wakes up and stops.
        """
        subscription = Subscription(run_id=run_id, maxsize=self._queue_size)
        async with self._lock:
            self._subscriptions.add(subscription)
        try:
            yield subscription
        finally:
            async with self._lock:
                self._subscriptions.discard(subscription)
            with contextlib.suppress(asyncio.QueueFull):
                subscription.queue.put_nowait(None)

    async def publish_event(self, event: ExecutionEvent) -> None:
        await self._publish(event)

    async def publish_status(self, run_id: str, status: RunStatus) -> None:
        await self._publish(status, run_id=run_id)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscriptions)

    async def _publish(
        self, item: ExecutionEvent | RunStatus, *, run_id: str | None = None
    ) -> None:
        event_run_id = run_id or getattr(item, "run_id", None)
        async with self._lock:
            targets = [
                s for s in self._subscriptions if s.run_id is None or s.run_id == event_run_id
            ]
        for subscription in targets:
            await subscription.offer(item)

    async def shutdown(self) -> None:
        """Close every subscription. Called on application shutdown."""
        async with self._lock:
            subscriptions = list(self._subscriptions)
            self._subscriptions.clear()
        for subscription in subscriptions:
            with contextlib.suppress(asyncio.QueueFull):
                subscription.queue.put_nowait(None)

    def stats(self) -> dict[str, Any]:
        return {
            "subscribers": self.subscriber_count,
            "dropped": sum(s.dropped for s in self._subscriptions),
        }
