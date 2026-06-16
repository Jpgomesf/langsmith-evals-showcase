"""Scenario 1 experiment wiring: seed the dataset and run the offline evaluation.

Showcases: offline ``evaluate()``, a heuristic row-level evaluator, dataset-level
summary metrics, and ``num_repetitions`` for measuring run-to-run variance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langsmith import Client, evaluate

from ...config import get_settings
from ...datasets import load_jsonl, upsert_dataset
from ...evaluators.heuristic import make_exact_match
from ...evaluators.summary import make_classification_summary
from .app import run_classifier

DATA_PATH = Path(__file__).parent / "data.jsonl"
SCENARIO = "classify"


def seed(client: Client) -> str:
    """Push the committed dataset to LangSmith (idempotent)."""
    name = get_settings().dataset_name(SCENARIO)
    return upsert_dataset(
        client,
        name,
        load_jsonl(DATA_PATH),
        description="Support-ticket intent classification (6 intents).",
    )


def run_experiment(*, repetitions: int = 1, max_concurrency: int = 4) -> Any:
    """Run the classification experiment against the seeded dataset."""
    settings = get_settings()
    return evaluate(
        run_classifier,
        data=settings.dataset_name(SCENARIO),
        evaluators=[make_exact_match("label")],
        # The summary evaluator returns the documented {"results": [...]} dict, which
        # LangSmith coerces at runtime; its stub types expect EvaluationResults.
        summary_evaluators=[make_classification_summary("label")],  # type: ignore[list-item]
        experiment_prefix=SCENARIO,
        num_repetitions=repetitions,
        max_concurrency=max_concurrency,
        metadata={"app_model": settings.app_model, "scenario": SCENARIO},
    )
