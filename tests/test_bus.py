"""Tests for the event bus."""

from __future__ import annotations

import asyncio

from repopilot.domain.models import EventType, ExecutionEvent, RunStatus
from repopilot.execution.bus import EventBus, RunStatusUpdate


def make_event(run_id: str, message: str = "hello") -> ExecutionEvent:
    return ExecutionEvent(run_id=run_id, sequence=1, type=EventType.LOG, message=message)


class TestDelivery:
    async def test_subscriber_receives_events(self) -> None:
        bus = EventBus()

        async with bus.subscribe() as subscription:
            await bus.publish_event(make_event("run_a"))
            received = await asyncio.wait_for(subscription.queue.get(), 1)

        assert received.message == "hello"

    async def test_events_from_other_runs_are_filtered(self) -> None:
        bus = EventBus()

        async with bus.subscribe(run_id="run_a") as subscription:
            await bus.publish_event(make_event("run_b"))
            await bus.publish_event(make_event("run_a", "mine"))

            received = await asyncio.wait_for(subscription.queue.get(), 1)

        assert received.run_id == "run_a"
        assert received.message == "mine"

    async def test_unfiltered_subscriber_sees_every_run(self) -> None:
        bus = EventBus()

        async with bus.subscribe() as subscription:
            await bus.publish_event(make_event("run_a"))
            await bus.publish_event(make_event("run_b"))

            first = await asyncio.wait_for(subscription.queue.get(), 1)
            second = await asyncio.wait_for(subscription.queue.get(), 1)

        assert {first.run_id, second.run_id} == {"run_a", "run_b"}

    async def test_fan_out_to_multiple_subscribers(self) -> None:
        bus = EventBus()

        async with bus.subscribe() as first, bus.subscribe() as second:
            await bus.publish_event(make_event("run_a"))

            a = await asyncio.wait_for(first.queue.get(), 1)
            b = await asyncio.wait_for(second.queue.get(), 1)

        assert a.id == b.id

    async def test_status_updates_are_published(self) -> None:
        bus = EventBus()

        async with bus.subscribe(run_id="run_a") as subscription:
            await bus.publish_status("run_a", RunStatus.SUCCEEDED)
            received = await asyncio.wait_for(subscription.queue.get(), 1)

        assert received == RunStatusUpdate(run_id="run_a", status=RunStatus.SUCCEEDED)

    async def test_status_updates_are_filtered_by_run(self) -> None:
        bus = EventBus()

        async with bus.subscribe(run_id="run_a") as subscription:
            await bus.publish_status("run_b", RunStatus.FAILED)
            await bus.publish_status("run_a", RunStatus.FAILED)

            received = await asyncio.wait_for(subscription.queue.get(), 1)

        assert received.run_id == "run_a"

    async def test_events_published_before_subscribing_are_not_delivered(self) -> None:
        bus = EventBus()
        await bus.publish_event(make_event("run_a"))

        async with bus.subscribe() as subscription:
            assert subscription.queue.empty()

    async def test_ordering_is_preserved(self) -> None:
        bus = EventBus()

        async with bus.subscribe() as subscription:
            for i in range(20):
                await bus.publish_event(make_event("run_a", f"line {i}"))

            messages = [
                (await asyncio.wait_for(subscription.queue.get(), 1)).message for _ in range(20)
            ]

        assert messages == [f"line {i}" for i in range(20)]


class TestLifecycle:
    async def test_unsubscribe_closes_the_iterator(self) -> None:
        bus = EventBus()
        received: list[str] = []

        async def consume() -> None:
            async with bus.subscribe() as subscription:
                received.append((await subscription.queue.get()).message)
            received.append("closed")

        task = asyncio.create_task(consume())
        await asyncio.sleep(0)
        assert bus.subscriber_count == 1

        await bus.publish_event(make_event("run_a", "only one"))
        await asyncio.wait_for(task, 2)

        assert received == ["only one", "closed"]
        assert bus.subscriber_count == 0

    async def test_shutdown_ends_an_active_iterator(self) -> None:
        bus = EventBus()
        received: list[str] = []

        async def consume() -> None:
            async with bus.subscribe() as subscription:
                async for item in subscription:
                    received.append(item.message if hasattr(item, "message") else str(item))
            received.append("closed")

        task = asyncio.create_task(consume())
        await asyncio.sleep(0)
        await bus.publish_event(make_event("run_a", "before shutdown"))
        await asyncio.sleep(0)
        await bus.shutdown()

        await asyncio.wait_for(task, 2)
        assert received == ["before shutdown", "closed"]

    async def test_slow_subscriber_does_not_block_the_publisher(self) -> None:
        bus = EventBus(queue_size=4)

        async with bus.subscribe() as slow:
            for i in range(4):
                await bus.publish_event(make_event("run_a", f"line {i}"))
            assert slow.dropped == 0
            assert slow.queue.qsize() == 4

            # The next publish is dropped, not awaited: the publisher never blocks.
            await bus.publish_event(make_event("run_a", "one too many"))
            assert slow.dropped == 1

            # The queue still holds the first four, in order.
            assert slow.queue.qsize() == 4

    async def test_dropped_events_are_counted_not_raised(self) -> None:
        bus = EventBus(queue_size=5)

        async with bus.subscribe() as subscription:
            for i in range(8):
                await bus.publish_event(make_event("run_a", str(i)))

            assert subscription.dropped == 3
            assert bus.stats()["dropped"] == 3

    async def test_shutdown_closes_every_subscription(self) -> None:
        bus = EventBus()

        async with bus.subscribe() as subscription:
            await bus.shutdown()
            assert subscription.queue.get_nowait() is None

    async def test_publishing_with_no_subscribers_is_a_noop(self) -> None:
        bus = EventBus()
        await bus.publish_event(make_event("run_a"))
        await bus.publish_status("run_a", RunStatus.RUNNING)
        assert bus.subscriber_count == 0
