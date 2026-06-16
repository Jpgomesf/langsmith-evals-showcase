"""Scenario 5 experiment wiring: Prompt Hub versioning + pairwise comparison.

Runs the summarizer twice (prompt ``v1`` and ``v2``), each scored by a
reference-free multi-criteria rubric judge, then compares the two experiments
head-to-head with ``evaluate_comparative`` and a pairwise judge.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langsmith import Client, evaluate
from langsmith.evaluation import evaluate_comparative

from ...config import get_settings
from ...datasets import load_jsonl, upsert_dataset
from ...evaluators.judges import make_pairwise_judge, make_rubric_judge
from .app import make_summarizer
from .prompts import push_prompts

DATA_PATH = Path(__file__).parent / "data.jsonl"
SCENARIO = "generate"

rubric_judge = make_rubric_judge(
    ["conciseness", "faithfulness", "helpfulness"],
    "You grade a summary of a document. Score each criterion 1-5: conciseness (no "
    "fluff or redundancy), faithfulness (no claims absent from or contradicting the "
    "document), and helpfulness (captures what matters). Judge the summary on its own "
    "merits — there is no reference summary.",
    lambda inputs, outputs, reference: (
        f"Document:\n{inputs['document']}\n\nSummary:\n{outputs.get('summary', '')}"
    ),
)

pairwise_judge = make_pairwise_judge(
    "preference",
    "You compare two summaries of the same document and choose the better one overall, "
    "weighing faithfulness first, then helpfulness, then conciseness.",
    lambda inputs, first, second: (
        f"Document:\n{inputs['document']}\n\n"
        f"Summary 1:\n{first.get('summary', '')}\n\n"
        f"Summary 2:\n{second.get('summary', '')}"
    ),
)


def seed(client: Client) -> str:
    """Push the dataset and both prompt versions to LangSmith (idempotent-ish)."""
    settings = get_settings()
    dataset_id = upsert_dataset(
        client,
        settings.dataset_name(SCENARIO),
        load_jsonl(DATA_PATH),
        description="Documents to summarize (reference-free).",
    )
    push_prompts(client)
    return dataset_id


def run_experiment(*, repetitions: int = 1, max_concurrency: int = 4) -> dict[str, Any]:
    """Run both prompt-version experiments and the pairwise comparison between them."""
    settings = get_settings()
    data = settings.dataset_name(SCENARIO)

    def _run(version: str) -> Any:
        return evaluate(
            make_summarizer(version),
            data=data,
            evaluators=[rubric_judge],
            experiment_prefix=f"{SCENARIO}-{version}",
            num_repetitions=repetitions,
            max_concurrency=max_concurrency,
            metadata={"scenario": SCENARIO, "prompt_version": version},
        )

    exp_v1 = _run("v1")
    exp_v2 = _run("v2")
    comparison = evaluate_comparative(
        (exp_v1.experiment_name, exp_v2.experiment_name),
        evaluators=[pairwise_judge],
        experiment_prefix=f"{SCENARIO}-pairwise",
    )
    return {"v1": exp_v1, "v2": exp_v2, "comparison": comparison}
