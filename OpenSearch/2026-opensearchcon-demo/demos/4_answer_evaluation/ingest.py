#!/usr/bin/env python3
"""Embed and index the Demo 4 grounding chunks into a real OpenSearch index.

Uses the standard embed-then-index pattern: embed each chunk with a small
sentence-transformers model, create a `knn_vector` index sized to that
model's dimension, and index each chunk with its embedding. Run this ONCE,
ahead of time (not live on stage), so
`evaluate_answer.py` can retrieve context via a real kNN query instead of a
hardcoded Python list.

    python ingest.py

Requires OPENSEARCH_HOST (and usually OPENSEARCH_USER/PASSWORD) to be set,
either via a `.env` file in this directory (see `.env.example`) or real
environment variables.
"""
from __future__ import annotations

from common.config import load_settings
from common.embeddings import EmbeddingModel, to_list
from common.logging import get_logger
from common.opensearch_client import create_client, ensure_vector_index, index_chunks

LOGGER = get_logger(__name__)

# Same two chunks Demo 2's good_structure.py uses. Kept here as the single
# source of truth for what gets indexed; evaluate_answer.py's
# FALLBACK_CHUNKS constant mirrors this exactly in case the cluster is
# unreachable on demo day.
CHUNKS = [
    {
        "id": 1,
        "origin": "Apple Inc. 10-Q filing - Q3 FY2025",
        "text": (
            "Total revenue for Q3 FY2025 was $94.9 billion. Gross margin "
            "for the quarter was 46.9 percent."
        ),
    },
    {
        "id": 2,
        "origin": "Apple Inc. 10-Q filing - Q4 FY2025",
        "text": (
            "Total revenue for Q4 FY2025 was $102.5 billion. Gross margin "
            "for the quarter was 47.3 percent."
        ),
    },
]


def main() -> None:
    settings = load_settings()

    if not settings.opensearch_host:
        raise SystemExit(
            "OPENSEARCH_HOST is not set. Copy .env.example to .env and fill in "
            "your OpenSearch cluster details before running ingest.py."
        )

    client = create_client(settings)
    embedder = EmbeddingModel(settings)
    ensure_vector_index(client, settings.opensearch_index, embedder.dimension)

    texts = [chunk["text"] for chunk in CHUNKS]
    embeddings = [to_list(vec) for vec in embedder.encode(texts)]
    index_chunks(client, settings.opensearch_index, CHUNKS, embeddings)

    LOGGER.info(
        "Ingested %d chunk(s) into index '%s' using embedding model '%s'. "
        "evaluate_answer.py can now retrieve them live via kNN search.",
        len(CHUNKS),
        settings.opensearch_index,
        settings.embedding_model,
    )


if __name__ == "__main__":
    main()
