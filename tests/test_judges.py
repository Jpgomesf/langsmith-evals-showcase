"""Unit tests for the LLM-judge wiring, using an injected stub model (no API)."""

from __future__ import annotations

from typing import Any

from evals_showcase.evaluators.judges import Grade, make_llm_judge


class _StubModel:
    """Stands in for a chat model; records the messages it was invoked with."""

    def __init__(self, grade: Grade):
        self._grade = grade
        self.last_messages: Any = None

    def with_structured_output(self, schema: Any) -> _StubModel:
        return self

    def invoke(self, messages: Any) -> Grade:
        self.last_messages = messages
        return self._grade


def test_llm_judge_shapes_feedback_and_renders_prompt():
    stub = _StubModel(Grade(reasoning="conveys the same fact", score=1))
    judge = make_llm_judge(
        "correctness",
        "INSTRUCTIONS",
        lambda inputs, outputs, reference: (
            f"q={inputs['q']} a={outputs['a']} ref={reference.get('ref')}"
        ),
        model_factory=lambda: stub,
    )
    out = judge({"q": "x"}, {"a": "y"}, {"ref": "z"})
    assert out == {"key": "correctness", "score": 1, "comment": "conveys the same fact"}
    assert stub.last_messages[0] == ("system", "INSTRUCTIONS")
    assert stub.last_messages[1] == ("human", "q=x a=y ref=z")


def test_llm_judge_handles_missing_reference():
    stub = _StubModel(Grade(reasoning="unsupported claim", score=0))
    judge = make_llm_judge(
        "faithfulness",
        "INSTRUCTIONS",
        lambda inputs, outputs, reference: "prompt",
        model_factory=lambda: stub,
    )
    out = judge({"q": "x"}, {"a": "y"})  # no reference_outputs passed
    assert out["key"] == "faithfulness"
    assert out["score"] == 0
