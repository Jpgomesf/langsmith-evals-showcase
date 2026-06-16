"""Provider-agnostic model factory.

Both the apps-under-test and the LLM-as-judge evaluators are built here, so the
entire showcase switches providers (Anthropic, OpenAI, ...) by changing the
``APP_MODEL`` / ``JUDGE_MODEL`` env vars — no code changes.
"""

from __future__ import annotations

from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from .config import get_settings


def get_chat_model(**kwargs: Any) -> BaseChatModel:
    """The application-under-test model (``APP_MODEL``)."""
    return init_chat_model(get_settings().app_model, **kwargs)


def get_judge_model(**kwargs: Any) -> BaseChatModel:
    """The LLM-as-judge model (``JUDGE_MODEL``).

    Judges default to ``temperature=0`` for the most stable grading possible;
    callers can override. Determinism is a mitigation, not a guarantee — see
    ``docs/EVALUATION_STRATEGY.md`` on judge variance.
    """
    kwargs.setdefault("temperature", 0)
    return init_chat_model(get_settings().judge_model, **kwargs)
