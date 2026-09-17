"""Central runtime configuration for every stage of this talk repo.

Every stage loads its settings from the same .env file at the repo root, so
the only thing that changes between stages is which index/model fields they
read -- not how they connect to Instaclustr OpenSearch or Bedrock.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

_ENV_LOADED = False


def _load_env_once() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    for candidate in (Path(".env"), Path(__file__).resolve().parent.parent / ".env"):
        if candidate.exists():
            load_dotenv(str(candidate))
            break
    _ENV_LOADED = True


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _get_str(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass
class Settings:
    """Runtime configuration shared by every stage."""

    # --- Instaclustr OpenSearch ---
    opensearch_host: str = "127.0.0.1"
    opensearch_port: int = 9200
    opensearch_user: str = ""
    opensearch_password: str = ""
    opensearch_ssl: bool = True

    # Per-stage default index names (each stage can still override via CLI/env)
    opensearch_hybrid_index: str = "recipes-hybrid"
    opensearch_recipe_bm25_index: str = "recipes-bm25"
    opensearch_recipe_vector_index: str = "recipes-vector"
    # Stage 1/2: baseline recipe-dataset indexes, kept separate so they never
    # collide with Stage 3/4's richer recipe indexes above.
    opensearch_vector_recipes_baseline_index: str = "recipes-vector-baseline"
    opensearch_bm25_recipes_baseline_index: str = "recipes-bm25-baseline"
    search_preference: str = "fix-your-retrieval-fix-your-rag"

    # --- Vector index construction ---
    # Defaults here work on any cluster with the k-NN plugin. They are
    # settings rather than constants so the same code can run against a
    # GPU-enabled cluster, which requires faiss -- see GPU.md.
    #
    # space_type l2 is equivalent to cosine for ranking purposes here because
    # EmbeddingModel.encode() L2-normalizes every vector.
    vector_engine: str = "lucene"
    vector_space_type: str = "l2"
    # Reconstruct vectors from the knn field on read instead of storing them
    # raw in _source. Off by default: the saving is irrelevant at sample size.
    knn_derived_source: bool = False
    knn_remote_build: bool = False
    # Minimum vector-data size per segment before a build is sent to the GPU
    # service. 50mb is the OpenSearch default; at 384 dims that is ~34k docs.
    knn_remote_build_size_min: str = "50mb"
    # 0 forces HNSW graph construction regardless of doc count. The default
    # (15000) silently skips graph building on small indexes, which also means
    # no remote build can ever trigger.
    knn_approximate_threshold: int = 0
    knn_number_of_shards: int = 1
    knn_number_of_replicas: int = 0
    # Disabled during bulk ingest, then restored -- see restore_refresh().
    ingest_refresh_interval: str = "-1"
    ingest_bulk_batch_size: int = 500

    # --- Embeddings (kept small on purpose -- see each stage's README) ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # --- NER service (local, lightweight, spaCy) ---
    ner_service_url: str = "http://127.0.0.1:8000/ner"
    ner_timeout_secs: float = 5.0

    # --- Amazon Bedrock ---
    aws_region: str = "us-east-1"
    bedrock_model_id: str = ""
    bedrock_max_tokens: int = 1024
    bedrock_temperature: float = 0.2

    # --- RAG behavior ---
    rag_top_k: int = 5
    rag_num_candidates: int = 25

    # --- Stage 3 hybrid fusion ---
    # A document matched only by BM25 can never score above hybrid_bm25_weight,
    # so this weight is the ceiling on how far an exact source/name match can
    # climb. 0.3 is enough on the 260-recipe sample; on the full corpus there
    # are thousands of documents with enough vector similarity to clear 0.3,
    # and the named recipe gets buried. See 03_hybrid_native_opensearch/README.
    hybrid_bm25_weight: float = 0.6
    hybrid_vector_weight: float = 0.4
    # How deep each sub-query is searched before the two lists are fused.
    # Leaving this equal to rag_top_k means a document ranked 6th by BM25 is
    # never seen by the combiner at all.
    hybrid_candidate_k: int = 50

    # --- Recipe HTTP fetch settings (unused unless a stage opts into it) ---
    recipe_http_timeout_secs: float = 10.0
    recipe_http_user_agent: str = "fix-your-retrieval-fix-your-rag/1.0"


def load_settings() -> Settings:
    """Load settings from environment variables (populated from .env)."""
    _load_env_once()

    return Settings(
        opensearch_host=_get_str("OPENSEARCH_HOST", Settings.opensearch_host),
        opensearch_port=_get_int("OPENSEARCH_PORT", Settings.opensearch_port),
        opensearch_user=_get_str("OPENSEARCH_USER", Settings.opensearch_user),
        opensearch_password=_get_str("OPENSEARCH_PASSWORD", Settings.opensearch_password),
        opensearch_ssl=_get_bool("OPENSEARCH_SSL", Settings.opensearch_ssl),
        opensearch_hybrid_index=_get_str("OPENSEARCH_HYBRID_INDEX", Settings.opensearch_hybrid_index),
        opensearch_recipe_bm25_index=_get_str("OPENSEARCH_RECIPE_BM25_INDEX", Settings.opensearch_recipe_bm25_index),
        opensearch_recipe_vector_index=_get_str(
            "OPENSEARCH_RECIPE_VECTOR_INDEX", Settings.opensearch_recipe_vector_index
        ),
        opensearch_vector_recipes_baseline_index=_get_str(
            "OPENSEARCH_VECTOR_RECIPES_BASELINE_INDEX", Settings.opensearch_vector_recipes_baseline_index
        ),
        opensearch_bm25_recipes_baseline_index=_get_str(
            "OPENSEARCH_BM25_RECIPES_BASELINE_INDEX", Settings.opensearch_bm25_recipes_baseline_index
        ),
        search_preference=_get_str("SEARCH_PREFERENCE", Settings.search_preference),
        vector_engine=_get_str("VECTOR_ENGINE", Settings.vector_engine),
        vector_space_type=_get_str("VECTOR_SPACE_TYPE", Settings.vector_space_type),
        knn_derived_source=_get_bool("KNN_DERIVED_SOURCE", Settings.knn_derived_source),
        knn_remote_build=_get_bool("KNN_REMOTE_BUILD", Settings.knn_remote_build),
        knn_remote_build_size_min=_get_str(
            "KNN_REMOTE_BUILD_SIZE_MIN", Settings.knn_remote_build_size_min
        ),
        knn_approximate_threshold=_get_int(
            "KNN_APPROXIMATE_THRESHOLD", Settings.knn_approximate_threshold
        ),
        knn_number_of_shards=_get_int("KNN_NUMBER_OF_SHARDS", Settings.knn_number_of_shards),
        knn_number_of_replicas=_get_int("KNN_NUMBER_OF_REPLICAS", Settings.knn_number_of_replicas),
        ingest_refresh_interval=_get_str("INGEST_REFRESH_INTERVAL", Settings.ingest_refresh_interval),
        ingest_bulk_batch_size=_get_int("INGEST_BULK_BATCH_SIZE", Settings.ingest_bulk_batch_size),
        embedding_model=_get_str("EMBEDDING_MODEL", Settings.embedding_model),
        embedding_dimension=_get_int("EMBEDDING_DIMENSION", Settings.embedding_dimension),
        ner_service_url=_get_str("NER_SERVICE_URL", Settings.ner_service_url),
        ner_timeout_secs=_get_float("NER_TIMEOUT_SECS", Settings.ner_timeout_secs),
        aws_region=_get_str("AWS_REGION", Settings.aws_region),
        bedrock_model_id=_get_str("BEDROCK_MODEL_ID", Settings.bedrock_model_id),
        bedrock_max_tokens=_get_int("BEDROCK_MAX_TOKENS", Settings.bedrock_max_tokens),
        bedrock_temperature=_get_float("BEDROCK_TEMPERATURE", Settings.bedrock_temperature),
        rag_top_k=_get_int("RAG_TOP_K", Settings.rag_top_k),
        rag_num_candidates=_get_int("RAG_NUM_CANDIDATES", Settings.rag_num_candidates),
        hybrid_bm25_weight=_get_float("HYBRID_BM25_WEIGHT", Settings.hybrid_bm25_weight),
        hybrid_vector_weight=_get_float("HYBRID_VECTOR_WEIGHT", Settings.hybrid_vector_weight),
        hybrid_candidate_k=_get_int("HYBRID_CANDIDATE_K", Settings.hybrid_candidate_k),
        recipe_http_timeout_secs=_get_float("RECIPE_HTTP_TIMEOUT_SECS", Settings.recipe_http_timeout_secs),
        recipe_http_user_agent=_get_str("RECIPE_HTTP_USER_AGENT", Settings.recipe_http_user_agent),
    )


def require_configured(settings: "Settings", *, need_bedrock: bool = True) -> None:
    """Fail fast with a clear message instead of a raw connection traceback.

    Every stage calls this right after loading settings, so a missing .env
    produces one actionable error instead of a urllib3/boto3 stack trace.
    """
    problems = []
    if settings.opensearch_host in ("", "127.0.0.1", "your-cluster-host.instaclustr.com"):
        problems.append("OPENSEARCH_HOST is not set to a real Instaclustr host")
    if not settings.opensearch_user or not settings.opensearch_password:
        problems.append("OPENSEARCH_USER / OPENSEARCH_PASSWORD are not set")
    if need_bedrock and not settings.bedrock_model_id:
        problems.append("BEDROCK_MODEL_ID is not set")

    if problems:
        raise RuntimeError(
            "Missing configuration in .env:\n  - "
            + "\n  - ".join(problems)
            + "\n\nCopy .env.example to .env at the repo root and fill in your "
            "Instaclustr OpenSearch and AWS Bedrock credentials, then try again."
        )


__all__ = ["Settings", "load_settings", "require_configured", "_get_bool", "_get_int", "_get_float", "_get_str"]
