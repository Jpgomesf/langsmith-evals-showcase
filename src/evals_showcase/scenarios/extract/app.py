"""Scenario 2 — structured field extraction (the app-under-test).

Parses a free-text transaction line into a typed ``Transaction``. The interesting
part is the evaluation: one custom code evaluator grading every field plus an
aggregate (see ``evaluators/code.py``).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from ...models import get_chat_model


class Category(StrEnum):
    """Closed set of spending categories."""

    GROCERIES = "groceries"
    DINING = "dining"
    TRANSPORT = "transport"
    UTILITIES = "utilities"
    ENTERTAINMENT = "entertainment"
    SHOPPING = "shopping"
    HEALTH = "health"
    OTHER = "other"


class Transaction(BaseModel):
    """The structured output schema enforced on the model."""

    merchant: str = Field(description="Merchant or payee name, as written.")
    amount: float = Field(description="Numeric amount, no currency symbol or thousands separators.")
    date: str = Field(description="Transaction date in ISO format, YYYY-MM-DD.")
    category: Category = Field(description="Best-matching spending category.")


_SYSTEM_PROMPT = (
    "Extract the transaction fields from the user's text. "
    "The date must be ISO format YYYY-MM-DD. The amount must be a plain number "
    "without currency symbols. Pick the single best-matching category."
)


def extract(text: str) -> dict[str, Any]:
    """Extract a transaction, returned as a JSON-friendly flat dict."""
    model = get_chat_model().with_structured_output(Transaction)
    result = model.invoke([("system", _SYSTEM_PROMPT), ("human", text)])
    assert isinstance(result, Transaction)  # narrows the union for type-checkers
    return result.model_dump(mode="json")


def run_extractor(inputs: dict[str, str]) -> dict[str, Any]:
    """LangSmith experiment target: ``{"text": ...}`` -> ``{merchant, amount, date, category}``."""
    return extract(inputs["text"])
