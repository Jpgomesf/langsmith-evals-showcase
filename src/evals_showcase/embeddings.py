"""Local embeddings for the RAG scenario.

Uses ``fastembed`` (ONNX, runs locally) so retrieval needs no embeddings API
key — the RAG showcase stays reproducible for anyone who clones the repo.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastembed import TextEmbedding

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


@lru_cache
def _embedder(model_name: str = DEFAULT_MODEL) -> TextEmbedding:
    # Imported lazily: fastembed pulls in onnxruntime and downloads the model on
    # first use, which we only want to pay for in the RAG scenario.
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=model_name)


def embed(texts: list[str], model_name: str = DEFAULT_MODEL) -> list[list[float]]:
    """Embed a list of texts into dense vectors."""
    return [[float(x) for x in vec] for vec in _embedder(model_name).embed(texts)]
