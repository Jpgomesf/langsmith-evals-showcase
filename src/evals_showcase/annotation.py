"""Annotation queues — route runs to humans for review.

Automated metrics only go so far; an annotation queue collects a sample of runs
for human grading against a rubric. The graded results become feedback in
LangSmith and can be used to calibrate the automated judges (see
``docs/EVALUATION_STRATEGY.md`` on validating judges against human labels).
"""

from __future__ import annotations

from typing import Any

from langsmith import Client

DEFAULT_RUBRIC = (
    "Rate the response's overall quality from 1 (poor) to 5 (excellent), "
    "considering correctness, helpfulness, and tone."
)


def create_queue(client: Client, name: str, *, rubric_instructions: str = DEFAULT_RUBRIC) -> Any:
    """Create (or address) an annotation queue with a human-review rubric."""
    return client.create_annotation_queue(
        name=name,
        description="Human review sample for the evals showcase.",
        rubric_instructions=rubric_instructions,
    )


def enqueue_from_project(
    client: Client,
    *,
    project: str,
    queue_name: str,
    limit: int = 10,
) -> Any:
    """Create a queue and enqueue the most recent ``limit`` runs from ``project``."""
    runs = list(client.list_runs(project_name=project, limit=limit))
    queue = create_queue(client, queue_name)
    if runs:
        client.add_runs_to_annotation_queue(queue.id, run_ids=[run.id for run in runs])
    return queue
