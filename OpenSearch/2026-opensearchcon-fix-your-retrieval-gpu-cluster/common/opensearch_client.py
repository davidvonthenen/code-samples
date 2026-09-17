"""OpenSearch client + index management shared by every stage.

All stages point at the same Instaclustr-hosted OpenSearch cluster (set once
via OPENSEARCH_HOST/PORT/USER/PASSWORD in .env). Each stage just uses a
different index name and mapping, defined below.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from opensearchpy import OpenSearch
from opensearchpy.exceptions import NotFoundError, TransportError

from .config import Settings, load_settings
from .labels import normalize_values
from .logging import get_logger

LOGGER = get_logger(__name__)

VECTOR_FIELD = "embedding"


# ---------------------------------------------------------------------------
# Vector index settings, driven by .env so the same code runs against a plain
# cluster and the GPU-enabled one.
# ---------------------------------------------------------------------------

def knn_method(settings: Optional[Settings] = None, *, dim: int) -> Dict[str, Any]:
    settings = settings or load_settings()
    return {
        "type": "knn_vector",
        "dimension": dim,
        "method": {
            "name": "hnsw",
            "space_type": settings.vector_space_type,
            "engine": settings.vector_engine,
        },
    }


def knn_index_settings(settings: Optional[Settings] = None) -> Dict[str, Any]:
    settings = settings or load_settings()
    body = {
        "index.knn": True,
        "index.knn.derived_source.enabled": settings.knn_derived_source,
        "index.knn.advanced.approximate_threshold": settings.knn_approximate_threshold,
        "index.number_of_shards": settings.knn_number_of_shards,
        "index.number_of_replicas": settings.knn_number_of_replicas,
        "index.refresh_interval": settings.ingest_refresh_interval,
    }
    # Only sent when GPU builds are wanted. These keys do not exist on
    # OpenSearch versions without the remote build feature, and an unknown
    # index setting fails index creation outright.
    if settings.knn_remote_build:
        body["index.knn.remote_index_build.enabled"] = True
        body["index.knn.remote_index_build.size.min"] = settings.knn_remote_build_size_min
    return body


def text_index_settings(settings: Optional[Settings] = None) -> Dict[str, Any]:
    """Shard/replica settings for the BM25-only indexes (no vectors)."""
    settings = settings or load_settings()
    return {
        "index.number_of_shards": settings.knn_number_of_shards,
        "index.number_of_replicas": settings.knn_number_of_replicas,
        "index.refresh_interval": settings.ingest_refresh_interval,
    }


def restore_refresh(client: OpenSearch, index_name: str, interval: Optional[str] = None) -> None:
    """Re-enable refresh after a bulk load and make everything searchable.

    Ingest disables refresh for throughput. Without this, force-merged or
    newly flushed segments can stay non-searchable indefinitely -- you end up
    querying a graph you did not just build.
    """
    client.indices.put_settings(
        index=index_name, body={"index": {"refresh_interval": interval or "1s"}}
    )
    client.indices.refresh(index=index_name)


def prepare_target_indexes(
    client: OpenSearch,
    index_names: List[str],
    *,
    recreate: bool = False,
    force: bool = False,
) -> bool:
    """Decide whether an ingest should run, clearing the way when asked.

    Returns False when the caller should skip ingest entirely. Re-running an
    ingest over a populated index is almost never what was intended: the
    ensure_* helpers only apply settings at creation, so a second run keeps
    the engine and derived-source choices of the first no matter what .env
    now says, and loading a different corpus size overwrites by _id while
    leaving the rest behind under a doc count that still looks correct.
    """
    if recreate:
        for name in index_names:
            if client.indices.exists(index=name):
                LOGGER.info("Deleting existing index '%s' (--recreate)", name)
                client.indices.delete(index=name)
        return True

    populated = []
    for name in index_names:
        if not client.indices.exists(index=name):
            continue
        count = client.count(index=name).get("count", 0)
        if count:
            populated.append((name, count))

    if not populated:
        return True

    listed = ", ".join(f"'{name}' ({count:,} docs)" for name, count in populated)
    if force:
        LOGGER.warning("Overwriting documents in %s in place (--force)", listed)
        LOGGER.warning("Index settings are NOT reapplied -- mapping and engine stay as created")
        return True

    LOGGER.warning("Already ingested: %s", listed)
    LOGGER.warning(
        "Skipping. Re-run with --recreate to drop and rebuild (needed after "
        "changing vector settings in .env or switching corpus size), or "
        "--force to overwrite documents in place."
    )
    return False


def create_client(settings: Optional[Settings] = None) -> OpenSearch:
    """Create the OpenSearch client used by every stage."""
    settings = settings or load_settings()
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
        ssl_show_warn=settings.opensearch_ssl,
        timeout=60,
        max_retries=3,
        retry_on_timeout=True,
    )


# ---------------------------------------------------------------------------
# Stage 3: single hybrid index (native OpenSearch hybrid pipeline, recipes)
# ---------------------------------------------------------------------------

def ensure_hybrid_index(
    client: OpenSearch, index_name: str, dim: int, settings: Optional[Settings] = None
) -> None:
    """Create the Stage 3 single index with both a text field and a knn_vector field."""
    if client.indices.exists(index=index_name):
        return
    settings = settings or load_settings()
    body = {
        "settings": knn_index_settings(settings),
        "mappings": {
            "properties": {
                "recipe_id": {"type": "keyword"},
                "recipe_name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                "text": {"type": "text"},
                "allergens": {"type": "keyword"},
                "diet_labels": {"type": "keyword"},
                "cuisine_type": {"type": "keyword"},
                VECTOR_FIELD: knn_method(settings, dim=dim),
            }
        },
    }
    LOGGER.info(
        "Creating hybrid index '%s' (engine=%s space=%s remote_build=%s)",
        index_name,
        settings.vector_engine,
        settings.vector_space_type,
        settings.knn_remote_build,
    )
    client.indices.create(index=index_name, body=body)


HYBRID_PIPELINE_NAME = "hybrid-search-pipeline"


def ensure_hybrid_pipeline(
    client: OpenSearch,
    *,
    bm25_weight: Optional[float] = None,
    vector_weight: Optional[float] = None,
    settings: Optional[Settings] = None,
    overwrite: bool = True,
) -> None:
    """Create (or update) the native OpenSearch normalization/combination pipeline.

    `overwrite=False` leaves an existing pipeline alone. Query scripts use that
    so they cannot silently reset weights that ingest.py was told to use.
    """
    settings = settings or load_settings()
    bm25_weight = settings.hybrid_bm25_weight if bm25_weight is None else bm25_weight
    vector_weight = settings.hybrid_vector_weight if vector_weight is None else vector_weight

    if not overwrite:
        try:
            client.transport.perform_request("GET", f"/_search/pipeline/{HYBRID_PIPELINE_NAME}")
            return
        except NotFoundError:
            pass

    client.transport.perform_request(
        "PUT",
        f"/_search/pipeline/{HYBRID_PIPELINE_NAME}",
        body={
            "description": "Normalize and combine BM25 + kNN scores",
            "phase_results_processors": [
                {
                    "normalization-processor": {
                        "normalization": {"technique": "min_max"},
                        "combination": {
                            "technique": "arithmetic_mean",
                            "parameters": {"weights": [bm25_weight, vector_weight]},
                        },
                    }
                }
            ],
        },
    )
    LOGGER.info(
        "Ensured hybrid search pipeline '%s' (bm25=%.2f vector=%.2f)",
        HYBRID_PIPELINE_NAME,
        bm25_weight,
        vector_weight,
    )


def hybrid_search(
    client: OpenSearch,
    index: str,
    text_query: str,
    query_vector: List[float],
    k: int = 10,
    candidate_k: Optional[int] = None,
    settings: Optional[Settings] = None,
):
    """Run one native hybrid (BM25 + kNN) query through the hybrid pipeline.

    Each sub-query is searched `candidate_k` deep; the fused list is then cut
    to `k`. The two are separate on purpose -- min-max normalization rescales
    each sub-query against its own returned window, so the window has to be
    wider than the answer for a keyword-only match to survive fusion.
    """
    settings = settings or load_settings()
    candidate_k = max(candidate_k or settings.hybrid_candidate_k, k)
    body = {
        "size": k,
        # Nothing downstream needs the vector back, and asking for it forces
        # derived-source reconstruction, which is both wasted work and the one
        # read path observed to fail on a cluster. Stages 1 and 4
        # already name their _source fields; this keeps stage 3 consistent.
        "_source": {"excludes": [VECTOR_FIELD]},
        "query": {
            "hybrid": {
                "pagination_depth": candidate_k,
                "queries": [
                    {"match": {"text": text_query}},
                    {"knn": {VECTOR_FIELD: {"vector": query_vector, "k": candidate_k}}},
                ],
            }
        },
    }
    return client.search(index=index, body=body, params={"search_pipeline": HYBRID_PIPELINE_NAME})


# ---------------------------------------------------------------------------
# Stage 1/2/4: BM25-grounding + vector-refine recipe indexes with structured
# allergen/diet/cuisine filters
# ---------------------------------------------------------------------------

RECIPE_SOURCE_FIELDS = [
    "recipe_id",
    "recipe_name",
    "source",
    "url",
    "image_url",
    "servings",
    "calories",
    "chunk_id",
    "chunk_index",
    "chunk_count",
    "text",
    "entities",
    "cautions",
    "cautions_display",
    "health_labels",
    "health_labels_display",
    "diet_labels",
    "diet_labels_display",
    "cuisine_type",
    "meal_type",
    "dish_type",
    "tier",
]


def ensure_recipe_vector_index(
    client: OpenSearch, index_name: str, dim: int, settings: Optional[Settings] = None
) -> None:
    if client.indices.exists(index=index_name):
        return
    settings = settings or load_settings()
    body = {
        "settings": knn_index_settings(settings),
        "mappings": {
            "properties": {
                "recipe_id": {"type": "keyword"},
                "recipe_name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                "source": {"type": "keyword"},
                "url": {"type": "keyword"},
                "image_url": {"type": "keyword"},
                "servings": {"type": "integer"},
                "calories": {"type": "float"},
                "chunk_id": {"type": "keyword"},
                "chunk_index": {"type": "integer"},
                "chunk_count": {"type": "integer"},
                "text": {"type": "text"},
                "cautions": {"type": "keyword"},
                "cautions_display": {"type": "keyword"},
                "diet_labels": {"type": "keyword"},
                "diet_labels_display": {"type": "keyword"},
                "health_labels": {"type": "keyword"},
                "health_labels_display": {"type": "keyword"},
                "cuisine_type": {"type": "keyword"},
                "meal_type": {"type": "keyword"},
                "dish_type": {"type": "keyword"},
                "tier": {"type": "keyword"},
                VECTOR_FIELD: knn_method(settings, dim=dim),
            }
        },
    }
    LOGGER.info(
        "Creating recipe vector index '%s' (engine=%s space=%s remote_build=%s size.min=%s)",
        index_name,
        settings.vector_engine,
        settings.vector_space_type,
        settings.knn_remote_build,
        settings.knn_remote_build_size_min,
    )
    client.indices.create(index=index_name, body=body)


def ensure_recipe_bm25_index(
    client: OpenSearch, index_name: str, settings: Optional[Settings] = None
) -> None:
    if client.indices.exists(index=index_name):
        return
    body = {
        "settings": text_index_settings(settings),
        "mappings": {
            "properties": {
                "recipe_id": {"type": "keyword"},
                "recipe_name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                "source": {"type": "keyword"},
                "url": {"type": "keyword"},
                "image_url": {"type": "keyword"},
                "servings": {"type": "integer"},
                "calories": {"type": "float"},
                "chunk_id": {"type": "keyword"},
                "chunk_index": {"type": "integer"},
                "chunk_count": {"type": "integer"},
                "text": {"type": "text"},
                "entities": {"type": "keyword"},
                "cautions": {"type": "keyword"},
                "cautions_display": {"type": "keyword"},
                "diet_labels": {"type": "keyword"},
                "diet_labels_display": {"type": "keyword"},
                "health_labels": {"type": "keyword"},
                "health_labels_display": {"type": "keyword"},
                "cuisine_type": {"type": "keyword"},
                "meal_type": {"type": "keyword"},
                "dish_type": {"type": "keyword"},
                "tier": {"type": "keyword"},
            }
        }
    }
    LOGGER.info("Creating recipe BM25 index '%s'", index_name)
    client.indices.create(index=index_name, body=body)


# Field aliases exposed to the CLI's generalized --exclude/--require flags.
# Instead of one hardcoded --exclude-caution flag, any of these fields can be
# excluded or required via `--exclude field=value` / `--require field=value`.
FILTERABLE_FIELDS = {
    "caution": "cautions",
    "allergen": "cautions",
    "diet": "diet_labels",
    "health": "health_labels",
    "cuisine": "cuisine_type",
    "meal": "meal_type",
    "dish": "dish_type",
}


def _resolve_field(alias: str) -> str:
    return FILTERABLE_FIELDS.get(alias.strip().lower(), alias.strip())


def parse_field_value_pairs(pairs: Iterable[str]) -> Dict[str, List[str]]:
    """Parse repeated ``field=value`` CLI args (e.g. ``allergen=Sulfites``)."""
    out: Dict[str, List[str]] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"Expected field=value, got: {pair!r}")
        field, value = pair.split("=", 1)
        resolved = _resolve_field(field)
        out.setdefault(resolved, []).append(value.strip())
    return out


def build_recipe_bm25_query(
    query_text: str,
    *,
    k: int,
    entities: Optional[Iterable[str]] = None,
    include_recipe_ids: Optional[Iterable[str]] = None,
    exclude: Optional[Dict[str, List[str]]] = None,
    require: Optional[Dict[str, List[str]]] = None,
    tier: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the BM25 grounding query with generalized structured filters."""
    text = str(query_text or "").strip()
    filters: List[Dict[str, Any]] = []
    must_not: List[Dict[str, Any]] = []
    should: List[Dict[str, Any]] = []

    recipe_ids = [str(x).strip() for x in (include_recipe_ids or []) if str(x).strip()]
    if recipe_ids:
        filters.append({"terms": {"recipe_id": recipe_ids}})

    if tier:
        filters.append({"term": {"tier": tier}})

    for field, values in (exclude or {}).items():
        normalized = normalize_values(values)
        if normalized:
            must_not.append({"terms": {field: normalized}})

    for field, values in (require or {}).items():
        normalized = normalize_values(values)
        for value in normalized:
            filters.append({"term": {field: value}})

    if entities:
        normalized_entities = normalize_values(entities)
        if normalized_entities:
            should.append(
                {
                    "constant_score": {
                        "filter": {"terms": {"entities": normalized_entities}},
                        "boost": 6.0,
                    }
                }
            )

    if text:
        must = [
            {
                "multi_match": {
                    "query": text,
                    "fields": ["recipe_name^4", "entities^3", "text^2", "source"],
                    "type": "best_fields",
                    "operator": "or",
                }
            }
        ]
    else:
        must = [{"match_all": {}}]

    bool_query: Dict[str, Any] = {"must": must, "filter": filters, "must_not": must_not}
    if should:
        bool_query["should"] = should

    return {"size": int(k), "_source": RECIPE_SOURCE_FIELDS, "query": {"bool": bool_query}}


def bm25_search(client: OpenSearch, index_name: str, query_text: str, **kwargs) -> Dict[str, Any]:
    body = build_recipe_bm25_query(query_text, **kwargs)
    try:
        return client.search(index=index_name, body=body, request_timeout=30)
    except TransportError as exc:
        LOGGER.warning("BM25 query failed on %s: %s", index_name, exc)
        return {"hits": {"total": {"value": 0, "relation": "eq"}, "hits": []}, "_error": str(exc)}


def build_recipe_vector_query(
    query_vector: List[float],
    *,
    k: int,
    candidate_k: Optional[int] = None,
    include_recipe_ids: Optional[Iterable[str]] = None,
    exclude: Optional[Dict[str, List[str]]] = None,
    require: Optional[Dict[str, List[str]]] = None,
    tier: Optional[str] = None,
) -> Dict[str, Any]:
    candidate_k = int(candidate_k or max(k, 25))
    filters: List[Dict[str, Any]] = []
    must_not: List[Dict[str, Any]] = []

    recipe_ids = [str(x) for x in (include_recipe_ids or []) if str(x).strip()]
    if recipe_ids:
        filters.append({"terms": {"recipe_id": recipe_ids}})

    if tier:
        filters.append({"term": {"tier": tier}})

    for field, values in (exclude or {}).items():
        normalized = normalize_values(values)
        if normalized:
            must_not.append({"terms": {field: normalized}})

    for field, values in (require or {}).items():
        normalized = normalize_values(values)
        if normalized:
            filters.append({"terms": {field: normalized}})

    # Filters go *inside* the knn clause, not in a surrounding bool.
    #
    # A bool wrapper post-filters: the knn clause returns its candidate_k
    # nearest neighbors across the whole index and the filter then intersects
    # that set. When the filter is selective -- Stage 4 narrows to ~25 grounded
    # recipe_ids out of 39,445 -- the intersection is usually empty, so a
    # correct query returns nothing. Small corpora hide this, because the
    # global neighbors and the filtered set overlap heavily.
    #
    # Inside the knn clause OpenSearch filters during traversal instead, and
    # falls back to exact search when the filter is selective enough to make
    # that cheaper.
    knn_clause: Dict[str, Any] = {"vector": query_vector, "k": candidate_k}
    if filters or must_not:
        knn_clause["filter"] = {"bool": {"filter": filters, "must_not": must_not}}

    return {
        "size": int(k),
        "_source": RECIPE_SOURCE_FIELDS,
        "query": {"knn": {VECTOR_FIELD: knn_clause}},
    }


def recipe_knn_search(client: OpenSearch, index_name: str, query_vector: List[float], **kwargs) -> Dict[str, Any]:
    body = build_recipe_vector_query(query_vector, **kwargs)
    try:
        return client.search(index=index_name, body=body, request_timeout=30)
    except TransportError as exc:
        LOGGER.warning("Vector query failed on %s: %s", index_name, exc)
        return {"hits": {"total": {"value": 0, "relation": "eq"}, "hits": []}, "_error": str(exc)}


def normalize_hits(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten OpenSearch hits into compact dicts for CLI/prompt use."""
    hits = response.get("hits", {}).get("hits", []) or []
    out = []
    for hit in hits:
        src = hit.get("_source", {})
        out.append({"score": float(hit.get("_score") or 0.0), "id": hit.get("_id") or "", **src})
    return out


# ---------------------------------------------------------------------------
# Bulk ingest
# ---------------------------------------------------------------------------

def bulk_index(
    client: OpenSearch,
    index_name: str,
    actions: Iterable[Dict[str, Any]],
    *,
    batch_size: int = 500,
    total: Optional[int] = None,
    desc: str = "Indexing",
) -> int:
    """Bulk-index documents, raising on the first batch that reports errors.

    ``actions`` yields ``{"_id": ..., "_source": {...}}`` dicts. At 39k docs
    the per-document index API would be ~39k round trips; this keeps a full
    corpus load to a couple of minutes.
    """
    from opensearchpy.helpers import streaming_bulk
    from tqdm import tqdm

    def _wrap() -> Iterable[Dict[str, Any]]:
        for action in actions:
            doc = dict(action)
            source = doc.pop("_source")
            yield {"_index": index_name, "_op_type": "index", **doc, **source}

    indexed = 0
    progress = tqdm(total=total, desc=desc, unit="docs")
    try:
        for ok, _ in streaming_bulk(
            client,
            _wrap(),
            chunk_size=batch_size,
            max_retries=3,
            initial_backoff=2,
            request_timeout=120,
            raise_on_error=True,
        ):
            if ok:
                indexed += 1
                progress.update(1)
    finally:
        progress.close()
    return indexed


# ---------------------------------------------------------------------------
# Post-ingest verification
# ---------------------------------------------------------------------------

def assert_vectors_readable(client: OpenSearch, index_name: str) -> None:
    """Fail loudly if documents cannot be read back with their vectors.

    Cheap gate that catches a broken derived-source read path immediately
    after ingest rather than several steps later during a reindex.
    """
    response = client.search(index=index_name, body={"size": 1}, request_timeout=60)
    hits = response.get("hits", {}).get("hits", [])
    if not hits:
        raise RuntimeError(f"'{index_name}' returned no documents -- ingest did not land")
    vector = hits[0].get("_source", {}).get(VECTOR_FIELD)
    if not isinstance(vector, list) or not vector:
        raise RuntimeError(
            f"'{index_name}' returned a document without a readable '{VECTOR_FIELD}'. "
            "If index.knn.derived_source.enabled is true, the reconstruction path "
            "is broken -- re-create the index with KNN_DERIVED_SOURCE=false."
        )
    LOGGER.info("Vector read-back OK on '%s' (%d dims)", index_name, len(vector))


def remote_build_stats(client: OpenSearch) -> Dict[str, int]:
    """Cluster-wide totals of the remote (GPU) index build counters."""
    response = client.transport.perform_request("GET", "/_plugins/_knn/stats")
    totals: Dict[str, int] = {}
    for node in (response.get("nodes") or {}).values():
        stats = node.get("remote_vector_index_build_stats") or {}
        for group in ("client_stats", "repository_stats", "build_stats"):
            for key, value in (stats.get(group) or {}).items():
                if isinstance(value, int):
                    totals[key] = totals.get(key, 0) + value
    return totals


def describe_build_delta(before: Dict[str, int], after: Dict[str, int]) -> str:
    """Human-readable summary of what the remote build counters did."""
    def d(key: str) -> int:
        return after.get(key, 0) - before.get(key, 0)

    builds = d("index_build_success_count")
    failures = d("index_build_failure_count") + d("build_request_failure_count")
    flush_ms = d("remote_index_build_flush_time_in_millis")
    merge_ms = d("remote_index_build_merge_time_in_millis")
    waiting_ms = d("waiting_time_in_ms")

    if builds == 0 and failures == 0:
        return (
            "No remote build triggered. Segment vector data was probably below "
            "KNN_REMOTE_BUILD_SIZE_MIN, or remote build is disabled cluster-side."
        )
    parts = [f"{builds} remote build(s) succeeded"]
    if failures:
        parts.append(f"{failures} FAILED")
    if flush_ms:
        parts.append(f"flush {flush_ms}ms")
    if merge_ms:
        parts.append(f"merge {merge_ms}ms")
    if waiting_ms:
        parts.append(f"{waiting_ms}ms waiting on build service")
    return ", ".join(parts)


def finish_vector_ingest(
    client: OpenSearch,
    index_name: str,
    before: Dict[str, int],
    *,
    force_merge: bool = True,
) -> None:
    """Close out a vector ingest and report whether a GPU build happened.

    Order matters. Force merge produces the single large segment whose vector
    data is compared against size.min, and refresh afterwards is what makes
    that newly built graph the one queries actually use -- skip it and you are
    searching the pre-merge segments instead.
    """
    client.indices.refresh(index=index_name)

    if force_merge:
        LOGGER.info("Force-merging '%s' to one segment", index_name)
        client.indices.forcemerge(
            index=index_name, max_num_segments=1, request_timeout=1800
        )

    restore_refresh(client, index_name)
    assert_vectors_readable(client, index_name)

    # Only meaningful when a GPU build was actually asked for; otherwise the
    # counters are flat by design and saying so just confuses the reader.
    if (load_settings()).knn_remote_build:
        after = remote_build_stats(client)
        LOGGER.info("Remote index build: %s", describe_build_delta(before, after))

    count = client.count(index=index_name).get("count", 0)
    LOGGER.info("'%s' now holds %d documents", index_name, count)


__all__ = [
    "create_client",
    "bulk_index",
    "finish_vector_ingest",
    "knn_index_settings",
    "knn_method",
    "text_index_settings",
    "restore_refresh",
    "prepare_target_indexes",
    "assert_vectors_readable",
    "remote_build_stats",
    "describe_build_delta",
    "ensure_hybrid_index",
    "ensure_hybrid_pipeline",
    "hybrid_search",
    "HYBRID_PIPELINE_NAME",
    "ensure_recipe_vector_index",
    "ensure_recipe_bm25_index",
    "FILTERABLE_FIELDS",
    "parse_field_value_pairs",
    "build_recipe_bm25_query",
    "bm25_search",
    "build_recipe_vector_query",
    "recipe_knn_search",
    "normalize_hits",
    "RECIPE_SOURCE_FIELDS",
    "VECTOR_FIELD",
]
