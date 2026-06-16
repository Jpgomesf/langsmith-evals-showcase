"""Scenario 2 experiment wiring: seed the dataset and run the extraction evaluation.

Showcases a single custom code evaluator that emits multiple feedback keys
(per-field matches + ``field_accuracy`` + ``all_fields_present``), each compared
with the right rule per field type (text / numeric tolerance / exact).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langsmith import Client, evaluate

from ...config import get_settings
from ...datasets import load_jsonl, upsert_dataset
from ...evaluators.code import FieldSpec, make_field_extraction_evaluator
from .app import run_extractor

DATA_PATH = Path(__file__).parent / "data.jsonl"
SCENARIO = "extract"

# How each extracted field is compared to its reference.
FIELDS = [
    FieldSpec("merchant", "text"),
    FieldSpec("amount", "numeric", tolerance=0.01),
    FieldSpec("date", "exact"),
    FieldSpec("category", "exact"),
]


def seed(client: Client) -> str:
    """Push the committed dataset to LangSmith (idempotent)."""
    name = get_settings().dataset_name(SCENARIO)
    return upsert_dataset(
        client,
        name,
        load_jsonl(DATA_PATH),
        description="Transaction field extraction (merchant, amount, date, category).",
    )


def run_experiment(*, repetitions: int = 1, max_concurrency: int = 4) -> Any:
    """Run the extraction experiment against the seeded dataset."""
    settings = get_settings()
    return evaluate(
        run_extractor,
        data=settings.dataset_name(SCENARIO),
        evaluators=[make_field_extraction_evaluator(FIELDS)],
        experiment_prefix=SCENARIO,
        num_repetitions=repetitions,
        max_concurrency=max_concurrency,
        metadata={"app_model": settings.app_model, "scenario": SCENARIO},
    )
