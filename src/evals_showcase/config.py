"""Runtime configuration, sourced from the environment / ``.env``."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed settings for the showcase.

    Models are ``"<provider>:<model>"`` strings consumed by langchain's
    ``init_chat_model`` (e.g. ``"anthropic:claude-haiku-4-5"``), which keeps the
    whole project provider-agnostic: swap a provider by changing one env var.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # The application-under-test model and the LLM-as-judge model.
    app_model: str = "anthropic:claude-haiku-4-5"
    judge_model: str = "anthropic:claude-sonnet-4-6"

    # LangSmith project that traces and experiments are written to.
    langsmith_project: str = "evals-showcase"

    # Prefix applied to every dataset name this repo creates.
    dataset_prefix: str = "evals-showcase"

    def dataset_name(self, scenario: str) -> str:
        """Fully-qualified dataset name for a scenario, e.g. ``evals-showcase-classify``."""
        return f"{self.dataset_prefix}-{scenario}"


@lru_cache
def get_settings() -> Settings:
    """Return cached settings so every module sees the same configuration."""
    return Settings()
