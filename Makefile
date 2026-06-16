.DEFAULT_GOAL := help
.PHONY: help install lint format typecheck test eval seed gate clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Create the venv and install all dependencies (incl. dev)
	uv sync

lint:  ## Lint with ruff
	uv run ruff check src tests

format:  ## Auto-format with ruff
	uv run ruff format src tests
	uv run ruff check --fix src tests

typecheck:  ## Type-check with mypy
	uv run mypy

test:  ## Run the deterministic unit tests (no API calls)
	uv run pytest

gate:  ## Run the LangSmith-backed regression gates (requires credentials)
	uv run pytest -m live

seed:  ## Push all scenario datasets to LangSmith (idempotent)
	uv run evals seed

eval:  ## Run a scenario experiment, e.g. `make eval SCENARIO=classify`
	uv run evals run $(SCENARIO)

clean:  ## Remove caches
	rm -rf .pytest_cache .ruff_cache .mypy_cache **/__pycache__
