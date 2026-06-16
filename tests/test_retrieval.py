"""Deterministic unit tests for retrieval math and the recall@k evaluator."""

from __future__ import annotations

from evals_showcase.evaluators.heuristic import make_recall_at_k
from evals_showcase.scenarios.rag.retriever import cosine, rank_by_cosine


def test_cosine_identical_and_orthogonal():
    assert cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_zero_vector_is_safe():
    assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_rank_by_cosine_orders_by_similarity():
    query = [1.0, 0.0]
    docs = [[0.0, 1.0], [1.0, 0.0], [0.9, 0.1]]
    assert rank_by_cosine(query, docs, k=2) == [1, 2]


def test_recall_at_k_partial():
    evaluator = make_recall_at_k()
    out = evaluator({"doc_ids": ["a", "b", "c"]}, {"doc_ids": ["a", "x"]})
    assert out == {"key": "recall_at_k", "score": 0.5}


def test_recall_at_k_full_and_empty_reference():
    evaluator = make_recall_at_k()
    assert evaluator({"doc_ids": ["a"]}, {"doc_ids": ["a"]})["score"] == 1.0
    assert evaluator({"doc_ids": []}, {"doc_ids": []})["score"] == 1.0
