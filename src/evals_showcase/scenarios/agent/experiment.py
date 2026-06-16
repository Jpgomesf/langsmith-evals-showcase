"""Scenario 4 experiment wiring: seed the dataset and run the agent evaluation.

Showcases async ``aevaluate()`` over a LangGraph agent, with trajectory
evaluators (ordered tool sequence + order-agnostic tool selection) alongside a
final-answer LLM-judge. Async results are materialized into a row list so the
shared gate helper works the same as the sync scenarios.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from langsmith import Client, aevaluate

from ...config import get_settings
from ...datasets import load_jsonl, upsert_dataset
from ...evaluators.judges import make_llm_judge
from ...evaluators.trajectory import make_trajectory_match
from .app import arun_agent

DATA_PATH = Path(__file__).parent / "data.jsonl"
SCENARIO = "agent"

trajectory_match = make_trajectory_match(mode="in_order", feedback_key="trajectory_match")
tool_selection = make_trajectory_match(mode="set", feedback_key="tool_selection")

final_answer_judge = make_llm_judge(
    "correctness",
    "You grade an agent's final answer against a reference answer. Score 1 if the "
    "answer contains the same numeric/factual result as the reference (wording and "
    "formatting may differ), otherwise 0.",
    lambda inputs, outputs, reference: (
        f"Question: {inputs['question']}\n"
        f"Reference answer: {reference.get('answer', '')}\n"
        f"Agent answer: {outputs.get('answer', '')}"
    ),
)


def seed(client: Client) -> str:
    """Push the committed dataset to LangSmith (idempotent)."""
    name = get_settings().dataset_name(SCENARIO)
    return upsert_dataset(
        client,
        name,
        load_jsonl(DATA_PATH),
        description="Tool-using ReAct agent tasks (calculator + word_count).",
    )


def run_experiment(*, repetitions: int = 1, max_concurrency: int = 4) -> list[Any]:
    """Run the async agent experiment and return the materialized result rows."""
    settings = get_settings()

    async def _run() -> list[Any]:
        results = await aevaluate(
            arun_agent,
            data=settings.dataset_name(SCENARIO),
            evaluators=[trajectory_match, tool_selection, final_answer_judge],
            experiment_prefix=SCENARIO,
            num_repetitions=repetitions,
            max_concurrency=max_concurrency,
            metadata={
                "app_model": settings.app_model,
                "judge_model": settings.judge_model,
                "scenario": SCENARIO,
            },
        )
        return [row async for row in results]

    return asyncio.run(_run())
