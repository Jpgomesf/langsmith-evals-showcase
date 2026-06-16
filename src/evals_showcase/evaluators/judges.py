"""LLM-as-judge evaluators.

Custom judges built on the configured ``JUDGE_MODEL`` with structured output.
The grade schema puts ``reasoning`` before ``score`` so the model reasons before
committing to a verdict (chain-of-thought grading); judges default to
``temperature=0``. The model factory is injectable, which keeps the wiring
unit-testable without any API call. See ``docs/EVALUATION_STRATEGY.md`` for when
an LLM-judge is the right tool and how to control its biases.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any

from langsmith.evaluation.evaluator import ComparisonEvaluationResult
from pydantic import BaseModel, Field, create_model

from ..models import get_judge_model

# Renders the (inputs, outputs, reference_outputs) triple into the judge's user prompt.
Renderer = Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], str]
# Renders (inputs, first_outputs, second_outputs) into a pairwise judge's prompt.
PairRenderer = Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], str]
RowEvaluator = Callable[..., dict[str, Any]]
ComparativeEvaluator = Callable[..., ComparisonEvaluationResult]


class Grade(BaseModel):
    """A binary judge verdict with its reasoning."""

    reasoning: str = Field(description="Concise justification for the verdict.")
    score: int = Field(description="1 if the criterion is satisfied, else 0.", ge=0, le=1)


def make_llm_judge(
    key: str,
    instructions: str,
    render: Renderer,
    *,
    model_factory: Callable[..., Any] = get_judge_model,
) -> RowEvaluator:
    """Build a binary LLM-judge evaluator emitting feedback ``key`` with a comment.

    ``instructions`` is the judge's system prompt (the rubric); ``render`` builds
    the user message from the example. ``model_factory`` is injectable for tests.
    """

    def judge(
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        reference_outputs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        model = model_factory().with_structured_output(Grade)
        grade = model.invoke(
            [
                ("system", instructions),
                ("human", render(inputs, outputs, reference_outputs or {})),
            ]
        )
        assert isinstance(grade, Grade)  # narrows for type-checkers
        return {"key": key, "score": grade.score, "comment": grade.reasoning}

    judge.__name__ = f"judge_{key}"
    return judge


def make_rubric_judge(
    criteria: list[str],
    instructions: str,
    render: Renderer,
    *,
    scale: int = 5,
    model_factory: Callable[..., Any] = get_judge_model,
) -> RowEvaluator:
    """Build a reference-free, multi-criteria rubric judge.

    Each criterion is scored ``1..scale`` (continuous feedback); the judge also
    emits an ``overall`` mean and a categorical ``verdict`` (pass/fail), showing
    both continuous and categorical feedback in one evaluator.
    """
    fields: dict[str, Any] = {
        "reasoning": (str, Field(description="Justification for the scores."))
    }
    for name in criteria:
        fields[name] = (int, Field(ge=1, le=scale, description=f"{name} score, 1-{scale}."))
    schema = create_model("RubricGrade", **fields)
    pass_mark = (scale + 1) / 2

    def rubric(
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        reference_outputs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        model = model_factory().with_structured_output(schema)
        graded = model.invoke(
            [("system", instructions), ("human", render(inputs, outputs, reference_outputs or {}))]
        )
        scores = [getattr(graded, name) for name in criteria]
        overall = sum(scores) / len(scores)
        results: list[dict[str, Any]] = [
            {"key": name, "score": getattr(graded, name)} for name in criteria
        ]
        results.append({"key": "overall", "score": overall})
        results.append({"key": "verdict", "value": "pass" if overall >= pass_mark else "fail"})
        return {"results": results}

    return rubric


class PairwiseChoice(BaseModel):
    """A pairwise judge verdict."""

    reasoning: str = Field(description="Why the chosen candidate is better.")
    winner: int = Field(
        description="1 if the first candidate is better, 2 if the second is.", ge=1, le=2
    )


def make_pairwise_judge(
    key: str,
    instructions: str,
    render_pair: PairRenderer,
    *,
    model_factory: Callable[..., Any] = get_judge_model,
) -> ComparativeEvaluator:
    """Build a comparative evaluator that picks the better of two experiment runs.

    Mitigates position bias by deterministically varying which candidate is shown
    first (keyed on the example id), so order effects cancel across the dataset.
    """

    def comparative(runs: list[Any], example: Any) -> ComparisonEvaluationResult:
        run_a, run_b = runs[0], runs[1]
        swap = int(hashlib.md5(str(example.id).encode()).hexdigest(), 16) % 2 == 1
        first, second = (run_b, run_a) if swap else (run_a, run_b)
        model = model_factory().with_structured_output(PairwiseChoice)
        choice = model.invoke(
            [
                ("system", instructions),
                (
                    "human",
                    render_pair(
                        dict(example.inputs or {}),
                        dict(first.outputs or {}),
                        dict(second.outputs or {}),
                    ),
                ),
            ]
        )
        assert isinstance(choice, PairwiseChoice)
        winner, loser = (first, second) if choice.winner == 1 else (second, first)
        return ComparisonEvaluationResult(
            key=key, scores={winner.id: 1, loser.id: 0}, comment=choice.reasoning
        )

    comparative.__name__ = f"pairwise_{key}"
    return comparative
