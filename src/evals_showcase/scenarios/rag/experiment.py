"""Scenario 3 experiment wiring: seed the dataset and run the RAG evaluation.

Showcases a hybrid eval suite: a heuristic retrieval metric (recall@k over the
intermediate retrieved ids) alongside three custom LLM-as-judge evaluators —
reference-based answer-correctness, reference-free faithfulness/groundedness, and
context-relevance. See ``docs/EVALUATION_STRATEGY.md`` on why RAG is a hybrid.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langsmith import Client, evaluate

from ...config import get_settings
from ...datasets import load_jsonl, upsert_dataset
from ...evaluators.heuristic import make_recall_at_k
from ...evaluators.judges import make_llm_judge
from .app import run_rag

DATA_PATH = Path(__file__).parent / "data.jsonl"
SCENARIO = "rag"


def _join_contexts(outputs: dict[str, Any]) -> str:
    return "\n\n".join(outputs.get("contexts") or [])


correctness_judge = make_llm_judge(
    "correctness",
    "You grade a candidate answer against a reference answer. Score 1 if the candidate "
    "conveys the same key facts as the reference (semantically equivalent, minor wording "
    "differences are fine), otherwise 0.",
    lambda inputs, outputs, reference: (
        f"Question: {inputs['question']}\n"
        f"Reference answer: {reference.get('answer', '')}\n"
        f"Candidate answer: {outputs.get('answer', '')}"
    ),
)

faithfulness_judge = make_llm_judge(
    "faithfulness",
    "You check whether an answer is grounded in the provided context. Score 1 if every "
    "factual claim in the answer is supported by the context; score 0 if the answer adds "
    "any claim not present in the context.",
    lambda inputs, outputs, reference: (
        f"Context:\n{_join_contexts(outputs)}\n\nAnswer: {outputs.get('answer', '')}"
    ),
)

context_relevance_judge = make_llm_judge(
    "context_relevance",
    "You judge retrieval quality. Score 1 if the retrieved context contains information "
    "relevant to answering the question; score 0 if it is off-topic or unhelpful.",
    lambda inputs, outputs, reference: (
        f"Question: {inputs['question']}\n\nRetrieved context:\n{_join_contexts(outputs)}"
    ),
)


def seed(client: Client) -> str:
    """Push the committed dataset to LangSmith (idempotent)."""
    name = get_settings().dataset_name(SCENARIO)
    return upsert_dataset(
        client,
        name,
        load_jsonl(DATA_PATH),
        description="RAG QA over the fictional Nimbus product corpus.",
    )


def run_experiment(*, repetitions: int = 1, max_concurrency: int = 4) -> Any:
    """Run the RAG experiment: retrieval recall@k + three LLM-judges."""
    settings = get_settings()
    return evaluate(
        run_rag,
        data=settings.dataset_name(SCENARIO),
        evaluators=[
            make_recall_at_k(),
            correctness_judge,
            faithfulness_judge,
            context_relevance_judge,
        ],
        experiment_prefix=SCENARIO,
        num_repetitions=repetitions,
        max_concurrency=max_concurrency,
        metadata={
            "app_model": settings.app_model,
            "judge_model": settings.judge_model,
            "scenario": SCENARIO,
        },
    )
