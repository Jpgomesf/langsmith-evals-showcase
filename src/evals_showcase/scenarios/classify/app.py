"""Scenario 1 — intent classification (the app-under-test).

A support-ticket router: one structured-output LLM call maps a message to exactly
one intent. Deliberately simple, because the interesting part is the *evaluation*
(heuristic exact-match + dataset-level summary metrics), not the model.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from ...models import get_chat_model


class Intent(StrEnum):
    """The closed set of support intents the router chooses from."""

    BILLING = "billing"
    TECHNICAL_SUPPORT = "technical_support"
    ACCOUNT_ACCESS = "account_access"
    FEATURE_REQUEST = "feature_request"
    COMPLAINT = "complaint"
    OTHER = "other"


class Classification(BaseModel):
    """Structured output schema enforced on the model."""

    label: Intent = Field(description="The single best-matching intent for the message.")


_SYSTEM_PROMPT = (
    "You are a support-ticket router. Read the customer's message and classify it "
    "into exactly one intent:\n"
    "- billing: invoices, charges, refunds, payment methods, pricing.\n"
    "- technical_support: errors, bugs, crashes, something not working.\n"
    "- account_access: login problems, passwords, lockouts, 2FA.\n"
    "- feature_request: asking for new capabilities or improvements.\n"
    "- complaint: expressing dissatisfaction with service, staff, or experience.\n"
    "- other: greetings, thanks, or anything that fits none of the above.\n"
    "Choose the dominant intent if several apply."
)


def classify(text: str) -> str:
    """Classify a single message, returning the intent label as a string."""
    model = get_chat_model().with_structured_output(Classification)
    result = model.invoke([("system", _SYSTEM_PROMPT), ("human", text)])
    assert isinstance(result, Classification)  # narrows the union for type-checkers
    return result.label.value


def run_classifier(inputs: dict[str, str]) -> dict[str, str]:
    """LangSmith experiment target: ``{"text": ...}`` -> ``{"label": ...}``."""
    return {"label": classify(inputs["text"])}
