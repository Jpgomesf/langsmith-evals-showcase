"""Scenario 3 — retrieval-augmented QA (the app-under-test).

Retrieves top-k docs from the local corpus and answers strictly from them. The
target returns the answer *and* the retrieved context/ids, so evaluators can
score both the final answer (LLM-judges) and the intermediate retrieval (recall@k).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain_core.messages import BaseMessage

from ...models import get_chat_model
from .retriever import Retriever, load_corpus

_SYSTEM_PROMPT = (
    "Answer the question using ONLY the provided context. "
    "If the answer is not contained in the context, say you don't know. "
    "Be concise and do not invent facts."
)


@lru_cache
def _retriever() -> Retriever:
    return Retriever(load_corpus())


def _as_text(message: BaseMessage) -> str:
    """Flatten message content to text (handles str or block-list content)."""
    content = message.content
    if isinstance(content, str):
        return content
    parts = [b.get("text", "") if isinstance(b, dict) else str(b) for b in content]
    return "".join(parts)


def answer(question: str, k: int = 3) -> dict[str, Any]:
    """Run retrieval + generation, returning the answer and retrieved context."""
    docs = _retriever().retrieve(question, k=k)
    context = "\n\n".join(f"[{doc['id']}] {doc['text']}" for doc in docs)
    response = get_chat_model().invoke(
        [("system", _SYSTEM_PROMPT), ("human", f"Context:\n{context}\n\nQuestion: {question}")]
    )
    return {
        "answer": _as_text(response),
        "contexts": [doc["text"] for doc in docs],
        "doc_ids": [doc["id"] for doc in docs],
    }


def run_rag(inputs: dict[str, str]) -> dict[str, Any]:
    """LangSmith experiment target: ``{"question": ...}`` -> answer + retrieved context."""
    return answer(inputs["question"])
