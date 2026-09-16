#!/usr/bin/env python3
"""Verify a GPU-built vector index returns the same neighbors as a CPU-built one.

A remote build can report success and still produce a graph with degraded
recall, so "the job completed" is not the same as "the index is correct".
This builds a local-CPU control index from the same documents and compares
top-k results question by question.

Compare by score as well as by id: if the corpus contains duplicate vectors,
ids tie arbitrarily and id overlap alone is misleading. A lower top score on
one side means that index genuinely failed to reach a better neighbor.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import load_settings, require_configured
from common.embeddings import EmbeddingModel, to_list
from common.logging import get_logger
from common.opensearch_client import create_client, recipe_knn_search, restore_refresh

LOGGER = get_logger(__name__)

DEFAULT_QUESTIONS = [
    "Does the womensweeklyfood.com.au Fish and Chips recipe contain sulfites?",
    "Does the thegratefulgirlcooks.com Fish and Chips recipe contain sulfites?",
    "vegetarian pasta with tomatoes and basil",
    "gluten free chocolate dessert",
    "spicy grilled shrimp appetizer",
    "low calorie chicken salad",
    "dairy free breakfast smoothie",
    "slow cooker beef stew",
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, help="The GPU-built index to check")
    parser.add_argument(
        "--control-index",
        default=None,
        help="CPU-built control index (default: <index>-cpu-control)",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Reuse an existing control index instead of rebuilding it",
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=50)
    return parser.parse_args(argv)


def build_control(client, source: str, control: str, settings) -> None:
    """Clone `source` into a control index that builds its graph locally."""
    if client.indices.exists(index=control):
        LOGGER.info("Deleting existing control index '%s'", control)
        client.indices.delete(index=control)

    mapping = client.indices.get_mapping(index=source)[source]["mappings"]
    body = {
        "settings": {
            "index.knn": True,
            "index.knn.derived_source.enabled": settings.knn_derived_source,
            "index.knn.remote_index_build.enabled": False,  # the whole point
            "index.knn.advanced.approximate_threshold": 0,
            "index.number_of_shards": settings.knn_number_of_shards,
            "index.number_of_replicas": 0,
            "index.refresh_interval": "-1",
        },
        "mappings": mapping,
    }
    LOGGER.info("Creating CPU control index '%s'", control)
    client.indices.create(index=control, body=body)

    LOGGER.info("Copying documents from '%s' (this reads _source, so it also "
                "confirms vectors are readable)", source)
    client.reindex(
        body={"source": {"index": source}, "dest": {"index": control}},
        params={"wait_for_completion": "true", "requests_per_second": "-1"},
        request_timeout=3600,
    )
    client.indices.refresh(index=control)
    client.indices.forcemerge(index=control, max_num_segments=1, request_timeout=1800)
    restore_refresh(client, control)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    control = args.control_index or f"{args.index}-cpu-control"

    settings = load_settings()
    require_configured(settings, need_bedrock=False)
    client = create_client(settings)
    embedder = EmbeddingModel(settings)

    if not args.skip_build:
        build_control(client, args.index, control, settings)

    gpu_count = client.count(index=args.index).get("count", 0)
    cpu_count = client.count(index=control).get("count", 0)
    if gpu_count != cpu_count:
        LOGGER.warning("Doc counts differ: %s=%d, %s=%d", args.index, gpu_count, control, cpu_count)
    print(f"\nComparing {args.index} ({gpu_count:,} docs) against {control} ({cpu_count:,} docs)\n")

    degraded = 0
    for question in DEFAULT_QUESTIONS:
        vec = to_list(embedder.encode([question])[0])
        gpu = recipe_knn_search(client, args.index, vec, k=args.k, candidate_k=args.candidate_k)
        cpu = recipe_knn_search(client, control, vec, k=args.k, candidate_k=args.candidate_k)

        gpu_hits = gpu["hits"]["hits"]
        cpu_hits = cpu["hits"]["hits"]
        if not gpu_hits or not cpu_hits:
            print(f"  NO HITS  {question[:52]}")
            degraded += 1
            continue

        gpu_ids = [h["_id"] for h in gpu_hits]
        cpu_ids = [h["_id"] for h in cpu_hits]
        overlap = len(set(gpu_ids) & set(cpu_ids)) / max(len(cpu_ids), 1) * 100

        gpu_top = gpu_hits[0]["_score"]
        cpu_top = cpu_hits[0]["_score"]
        # Scores are similarity here, so lower on the GPU side means it never
        # reached a neighbor the CPU build found.
        delta = gpu_top - cpu_top
        flag = "  " if delta >= -1e-6 else "<<"
        if delta < -1e-6:
            degraded += 1

        print(f"{flag} overlap={overlap:5.1f}%  gpu_top={gpu_top:.6f}  "
              f"cpu_top={cpu_top:.6f}  {question[:48]}")

    print()
    if degraded:
        print(f"{degraded}/{len(DEFAULT_QUESTIONS)} queries where the GPU-built index "
              "found a strictly worse nearest neighbor -- graph recall is degraded")
        return 1
    print(f"All {len(DEFAULT_QUESTIONS)} queries matched or beat the CPU control")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
