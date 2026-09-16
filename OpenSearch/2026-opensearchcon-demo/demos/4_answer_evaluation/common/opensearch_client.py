"""OpenSearch client + vector index helpers for Demo 4.

Uses a plain `knn_vector` index + kNN search, scaled down to the one small
index this demo needs. `ingest.py` embeds and indexes the grounding
chunks; the question is embedded and searched the same way in
`evaluate_answer.py`.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from opensearchpy import OpenSearch

from .config import Settings
from .logging import get_logger

LOGGER = get_logger(__name__)


def create_client(settings: Settings) -> OpenSearch:
    """Create an OpenSearch client from Settings."""

    http_auth = None
    if settings.opensearch_user and settings.opensearch_password:
        http_auth = (settings.opensearch_user, settings.opensearch_password)

    LOGGER.info(
        "Connecting to OpenSearch at %s:%s (ssl=%s)",
        settings.opensearch_host,
        settings.opensearch_port,
        settings.opensearch_ssl,
    )
    return OpenSearch(
        hosts=[{"host": settings.opensearch_host, "port": settings.opensearch_port}],
        http_auth=http_auth,
        use_ssl=settings.opensearch_ssl,
        verify_certs=settings.opensearch_ssl,
        ssl_show_warn=False,
        timeout=15,
        max_retries=2,
        retry_on_timeout=True,
    )


def ensure_vector_index(client: OpenSearch, index_name: str, dim: int) -> None:
    """Create the Demo 4 vector index (chunk_id/source/text + embedding), if needed."""

    if client.indices.exists(index=index_name):
        return
    body: Dict[str, Any] = {
        "settings": {"index": {"knn": True}, "number_of_replicas": 0},
        "mappings": {
            "properties": {
                "chunk_id": {"type": "integer"},
                "source": {"type": "keyword"},
                "text": {"type": "text"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": dim,
                    "method": {"name": "hnsw", "space_type": "cosinesimil", "engine": "lucene"},
                },
            }
        },
    }
    LOGGER.info("Creating vector index '%s' (dim=%d)", index_name, dim)
    client.indices.create(index=index_name, body=body)


def index_chunks(
    client: OpenSearch, index_name: str, chunks: List[Dict[str, Any]], embeddings: List[List[float]]
) -> None:
    """Index each chunk with its embedding. Doc id == chunk id, so re-running is idempotent."""

    for chunk, embedding in zip(chunks, embeddings):
        client.index(
            index=index_name,
            id=str(chunk["id"]),
            body={
                "chunk_id": chunk["id"],
                "source": chunk["origin"],
                "text": chunk["text"],
                "embedding": embedding,
            },
            refresh=True,
        )
        LOGGER.info("Indexed chunk %s (%s)", chunk["id"], chunk["origin"])


def knn_search(
    client: OpenSearch, index_name: str, query_vec: List[float], *, k: int = 2
) -> List[Dict[str, Any]]:
    """Run a kNN search and return normalized chunk dicts.

    Returns results shaped like the hardcoded fallback chunks this demo used
    before OpenSearch was wired in: ``{"id", "origin", "text", "score"}``.
    """

    body: Dict[str, Any] = {
        "size": k,
        "query": {"knn": {"embedding": {"vector": query_vec, "k": k}}},
        "_source": ["chunk_id", "source", "text"],
    }
    response = client.search(index=index_name, body=body)
    hits = response.get("hits", {}).get("hits", [])

    results: List[Dict[str, Any]] = []
    for hit in hits:
        source = hit.get("_source", {})
        results.append(
            {
                "id": source.get("chunk_id"),
                "origin": source.get("source", ""),
                "text": source.get("text", ""),
                "score": hit.get("_score"),
            }
        )
    return results


def client_from_settings(settings: Settings) -> Optional[OpenSearch]:
    """Return a connected client, or None if OpenSearch isn't configured.

    Used by evaluate_answer.py to decide whether to attempt live retrieval
    at all before falling back to the built-in sample context.
    """

    if not settings.opensearch_host:
        return None
    return create_client(settings)


__all__ = [
    "create_client",
    "ensure_vector_index",
    "index_chunks",
    "knn_search",
    "client_from_settings",
]
