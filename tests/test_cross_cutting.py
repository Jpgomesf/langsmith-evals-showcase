"""Unit tests for the online simulator and annotation-queue wiring (no API)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from evals_showcase.annotation import enqueue_from_project
from evals_showcase.online import SAMPLE_TRAFFIC, take_traffic


def test_take_traffic_cycles_through_pool():
    assert len(take_traffic(20)) == 20
    assert take_traffic(3) == SAMPLE_TRAFFIC[:3]
    # wraps around when n exceeds the pool size
    assert take_traffic(len(SAMPLE_TRAFFIC) + 1)[-1] == SAMPLE_TRAFFIC[0]


class _StubClient:
    """Records the annotation-queue calls made against it."""

    def __init__(self, runs: list[Any]):
        self._runs = runs
        self.added: Any = None
        self.created: Any = None

    def list_runs(self, *, project_name: str, limit: int) -> list[Any]:
        return self._runs[:limit]

    def create_annotation_queue(
        self, *, name: str, description: str = "", rubric_instructions: str = ""
    ) -> Any:
        self.created = (name, rubric_instructions)
        return SimpleNamespace(id="queue-1")

    def add_runs_to_annotation_queue(self, queue_id: str, *, run_ids: list[str]) -> None:
        self.added = (queue_id, run_ids)


def test_enqueue_from_project_creates_and_enqueues():
    client = _StubClient([SimpleNamespace(id=f"r{i}") for i in range(5)])
    queue = enqueue_from_project(client, project="p", queue_name="review", limit=3)
    assert queue.id == "queue-1"
    assert client.created[0] == "review"
    assert client.added == ("queue-1", ["r0", "r1", "r2"])


def test_enqueue_from_project_handles_no_runs():
    client = _StubClient([])
    enqueue_from_project(client, project="p", queue_name="review", limit=3)
    assert client.added is None  # nothing to enqueue
