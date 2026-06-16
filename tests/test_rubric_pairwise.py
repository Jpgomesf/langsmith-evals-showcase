"""Unit tests for the rubric and pairwise judges, using injected stub models."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from langsmith.evaluation.evaluator import ComparisonEvaluationResult

from evals_showcase.evaluators.judges import (
    PairwiseChoice,
    make_pairwise_judge,
    make_rubric_judge,
)


class _StubModel:
    def __init__(self, result: Any):
        self._result = result

    def with_structured_output(self, schema: Any) -> _StubModel:
        return self

    def invoke(self, messages: Any) -> Any:
        return self._result


def test_rubric_judge_emits_criteria_overall_and_verdict():
    graded = SimpleNamespace(reasoning="ok", conciseness=5, faithfulness=4, helpfulness=3)
    judge = make_rubric_judge(
        ["conciseness", "faithfulness", "helpfulness"],
        "INSTRUCTIONS",
        lambda inputs, outputs, reference: "prompt",
        model_factory=lambda: _StubModel(graded),
    )
    by_key = {r["key"]: r for r in judge({"document": "d"}, {"summary": "s"})["results"]}
    assert by_key["conciseness"]["score"] == 5
    assert by_key["overall"]["score"] == 4.0
    assert by_key["verdict"]["value"] == "pass"  # 4.0 >= pass mark (3.0)


def test_rubric_verdict_fails_below_pass_mark():
    graded = SimpleNamespace(reasoning="weak", conciseness=2, faithfulness=1, helpfulness=2)
    judge = make_rubric_judge(
        ["conciseness", "faithfulness", "helpfulness"],
        "INSTRUCTIONS",
        lambda inputs, outputs, reference: "prompt",
        model_factory=lambda: _StubModel(graded),
    )
    by_key = {r["key"]: r for r in judge({"document": "d"}, {"summary": "s"})["results"]}
    assert by_key["verdict"]["value"] == "fail"


def test_pairwise_judge_returns_one_winner():
    judge = make_pairwise_judge(
        "preference",
        "INSTRUCTIONS",
        lambda inputs, first, second: "prompt",
        model_factory=lambda: _StubModel(PairwiseChoice(reasoning="cleaner", winner=1)),
    )
    runs = [
        SimpleNamespace(id="run-a", outputs={"summary": "A"}),
        SimpleNamespace(id="run-b", outputs={"summary": "B"}),
    ]
    example = SimpleNamespace(id="ex-1", inputs={"document": "d"})
    result = judge(runs, example)
    assert isinstance(result, ComparisonEvaluationResult)
    assert result.key == "preference"
    assert sorted(result.scores.values()) == [0, 1]
    assert set(result.scores.keys()) == {"run-a", "run-b"}
