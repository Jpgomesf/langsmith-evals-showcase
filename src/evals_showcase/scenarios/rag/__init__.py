"""Scenario 3 — retrieval-augmented QA."""

from .app import answer, run_rag
from .experiment import run_experiment, seed

__all__ = ["answer", "run_rag", "run_experiment", "seed"]
