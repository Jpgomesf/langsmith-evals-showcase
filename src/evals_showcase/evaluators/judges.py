"""LLM-as-judge evaluators.

Custom judges built on the configured ``JUDGE_MODEL`` with structured output.
The grade schema puts ``reasoning`` before ``score`` so the model reasons before
committing to a verdict (chain-of-thought grading); judges default to
``temperature=0``. The model factory is injectable, which keeps the wiring
unit-testable without any API call. See ``docs/EVALUATION_STRATEGY.md`` for when
an LLM-judge is the right tool and how to control its biases.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from ..models import get_judge_model

# Renders the (inputs, outputs, reference_outputs) triple into the judge's user prompt.
Renderer = Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], str]
RowEvaluator = Callable[..., dict[str, Any]]


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
