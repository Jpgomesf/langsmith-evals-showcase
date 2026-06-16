"""Idempotent dataset seeding.

Datasets live as committed JSONL files next to each scenario; this module pushes
them to LangSmith so re-running ``evals seed`` always reflects the data on disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langsmith import Client


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of example dicts."""
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def upsert_dataset(
    client: Client,
    name: str,
    examples: list[dict[str, Any]],
    *,
    description: str = "",
) -> str:
    """Create the dataset if absent and replace its examples to match the input.

    Each example is a dict shaped like ``{"inputs": {...}, "outputs": {...},
    "metadata": {...}}`` (``outputs``/``metadata`` optional). Returns the dataset id.
    """
    if client.has_dataset(dataset_name=name):
        dataset = client.read_dataset(dataset_name=name)
    else:
        dataset = client.create_dataset(dataset_name=name, description=description)

    # Replace existing examples so the dataset always mirrors the committed file.
    existing = list(client.list_examples(dataset_id=dataset.id))
    if existing:
        client.delete_examples(example_ids=[example.id for example in existing])

    client.create_examples(dataset_id=dataset.id, examples=examples)
    return str(dataset.id)
