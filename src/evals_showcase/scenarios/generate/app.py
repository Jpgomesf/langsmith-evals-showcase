"""Scenario 5 — summarization (the app-under-test).

Pulls a versioned prompt from the LangSmith Prompt Hub and runs it through the
configured model. Each prompt version becomes its own experiment; the two are
compared pairwise. Requires ``push_prompts`` to have run (via ``evals seed``).
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import Any

from langsmith import Client

from ...models import get_chat_model, message_text
from .prompts import prompt_name


@lru_cache
def _prompt(version: str) -> Any:
    """Pull a prompt version from the Hub by tag (cached per version)."""
    return Client().pull_prompt(f"{prompt_name()}:{version}")


def summarize(document: str, version: str = "v1") -> str:
    """Summarize a document with the given Prompt Hub version."""
    chain = _prompt(version) | get_chat_model()
    return message_text(chain.invoke({"document": document}))


def make_summarizer(version: str) -> Callable[[dict[str, str]], dict[str, str]]:
    """Build a LangSmith target that summarizes with a fixed prompt version."""

    def run(inputs: dict[str, str]) -> dict[str, str]:
        return {"summary": summarize(inputs["document"], version)}

    run.__name__ = f"summarize_{version}"
    return run
