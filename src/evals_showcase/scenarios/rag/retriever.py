"""A tiny in-memory vector retriever for the RAG scenario.

Embeds a small committed corpus with local ``fastembed`` and ranks by cosine
similarity. The ranking math is pure (no embeddings) so it is unit-testable; the
embedding call is isolated to :meth:`Retriever.__init__` / :meth:`retrieve`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from ...embeddings import embed

CORPUS_PATH = Path(__file__).parent / "corpus.jsonl"


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def rank_by_cosine(query_vec: list[float], doc_vecs: list[list[float]], k: int) -> list[int]:
    """Return the indices of the ``k`` doc vectors most similar to the query."""
    scored = sorted(
        range(len(doc_vecs)), key=lambda i: cosine(query_vec, doc_vecs[i]), reverse=True
    )
    return scored[:k]


def load_corpus(path: Path = CORPUS_PATH) -> list[dict[str, str]]:
    """Load the corpus as a list of ``{"id", "text"}`` documents."""
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


class Retriever:
    """Embeds a corpus once, then retrieves top-k documents per query."""

    def __init__(self, docs: list[dict[str, str]]):
        self.docs = docs
        self._vecs = embed([doc["text"] for doc in docs])

    def retrieve(self, query: str, k: int = 3) -> list[dict[str, Any]]:
        query_vec = embed([query])[0]
        return [self.docs[i] for i in rank_by_cosine(query_vec, self._vecs, k)]
