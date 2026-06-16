"""Prompt Hub integration for the generate scenario.

Two summarization prompt versions are pushed to the LangSmith Prompt Hub under
one name, tagged ``v1`` and ``v2``; the app pulls a specific version by tag.
This is how prompts are versioned and pinned independently of the code.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langsmith import Client

from ...config import get_settings


def prompt_name() -> str:
    """Hub identifier for the summarization prompt (workspace-prefixed)."""
    return f"{get_settings().dataset_prefix}-summarize"


# v1: terse. v2: more thorough. The pairwise judge decides which the dataset prefers.
PROMPT_V1 = ChatPromptTemplate.from_messages(
    [
        ("system", "Summarize the document in a single concise sentence."),
        ("human", "{document}"),
    ]
)
PROMPT_V2 = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Summarize the document in 2-3 sentences, capturing the key points and "
            "preserving any important numbers or names.",
        ),
        ("human", "{document}"),
    ]
)


def push_prompts(client: Client) -> None:
    """Push both prompt versions to the Hub, tagged ``v1`` and ``v2`` (idempotent-ish)."""
    name = prompt_name()
    client.push_prompt(name, object=PROMPT_V1, tags=["v1"])
    client.push_prompt(name, object=PROMPT_V2, tags=["v2"])
