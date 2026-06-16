"""Deterministic unit tests for the evaluator library — no API calls."""

from __future__ import annotations

from types import SimpleNamespace

from evals_showcase.evaluators.heuristic import make_exact_match
from evals_showcase.evaluators.summary import (
    accuracy,
    macro_precision_recall_f1,
    make_classification_summary,
)


def test_exact_match_is_case_insensitive():
    evaluator = make_exact_match("label")
    assert evaluator({"label": "Billing"}, {"label": "billing"}) == {"key": "correct", "score": 1}


def test_exact_match_mismatch_scores_zero():
    evaluator = make_exact_match("label")
    assert evaluator({"label": "billing"}, {"label": "complaint"})["score"] == 0


def test_accuracy():
    assert accuracy([("a", "a"), ("a", "b"), ("b", "b"), ("b", "a")]) == 0.5
    assert accuracy([]) == 0.0


def test_macro_prf_perfect():
    assert macro_precision_recall_f1([("x", "x"), ("y", "y")]) == (1.0, 1.0, 1.0)


def test_macro_prf_mixed():
    # Two classes, each 1 correct / 1 wrong both ways -> 0.5 across the board.
    pairs = [("a", "a"), ("a", "b"), ("b", "b"), ("b", "a")]
    assert macro_precision_recall_f1(pairs) == (0.5, 0.5, 0.5)


def test_macro_prf_empty():
    assert macro_precision_recall_f1([]) == (0.0, 0.0, 0.0)


def test_classification_summary_emits_four_metrics():
    runs = [SimpleNamespace(outputs={"label": "a"}), SimpleNamespace(outputs={"label": "b"})]
    examples = [SimpleNamespace(outputs={"label": "a"}), SimpleNamespace(outputs={"label": "b"})]
    out = make_classification_summary("label")(runs, examples)
    keys = {r["key"] for r in out["results"]}
    assert keys == {"accuracy", "macro_precision", "macro_recall", "macro_f1"}
    assert all(r["score"] == 1.0 for r in out["results"])
