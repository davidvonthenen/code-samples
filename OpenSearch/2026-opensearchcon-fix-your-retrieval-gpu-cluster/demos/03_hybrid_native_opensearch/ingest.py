#!/usr/bin/env python3
"""Stage 3: embed and index the recipe corpus into a single native
OpenSearch hybrid index (text + embedding in one document, one index).

Same "Recipe & Dietary Safety Assistant" theme and dataset as Stage 1/2,
but here there is no separate BM25 index, no separate vector index, and no
NER microservice to run alongside it. Every recipe's source name is already
baked into `text` (see build_text below), so plain BM25 keyword matching on
that field is enough to disambiguate by source -- the entity-tagging Stage 2
needed a whole Flask service for is unnecessary once BM25 and vector search
are fused into one native OpenSearch query. See README.md for how the hybrid
search pipeline combines the two signals.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from common.config import load_settings, require_configured
from common.corpus import add_corpus_args, add_reingest_args, load_recipes, resolve_data_file
from common.embeddings import EmbeddingModel, to_list
from common.labels import normalize_values
from common.logging import get_logger
from common.opensearch_client import (
    bulk_index,
    create_client,
    ensure_hybrid_index,
    ensure_hybrid_pipeline,
    finish_vector_ingest,
    prepare_target_indexes,
    remote_build_stats,
)

LOGGER = get_logger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest recipes into the hybrid index")
    add_corpus_args(parser, Path(__file__).parent)
    add_reingest_args(parser)
    parser.add_argument("--index-name", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=256, help="Embedding batch size")
    parser.add_argument(
        "--bm25-weight", type=float, default=None, help="Weight given to the BM25 sub-query"
    )
    parser.add_argument(
        "--vector-weight", type=float, default=None, help="Weight given to the kNN sub-query"
    )
    parser.add_argument(
        "--no-force-merge",
        action="store_true",
        help="Skip the post-ingest force merge (which is what triggers a merge-path GPU build)",
    )
    return parser.parse_args(argv)


def build_text(recipe: dict) -> str:
    """Render one recipe into the text that gets embedded AND BM25-matched.

    The source domain is folded directly into the text (not just stored as a
    separate keyword field) so that a question naming a specific source, e.g.
    "the thegratefulgirlcooks.com Fish and Chips recipe", has a literal term
    to match in the same `match` clause the hybrid query runs against.
    """
    lines = recipe.get("ingredient_lines") or []
    cautions = recipe.get("cautions") or []
    health = recipe.get("health_labels") or []
    parts = [
        f"{recipe['recipe_name']} (source: {recipe['source']})",
        "Ingredients: " + "; ".join(lines) if lines else "",
        f"Allergen cautions: {', '.join(cautions)}." if cautions else "Allergen cautions: none listed.",
        f"Health labels: {', '.join(health)}." if health else "",
    ]
    return "\n".join(p for p in parts if p)


def _actions(recipes, embedder, batch_size):
    for start in range(0, len(recipes), batch_size):
        batch = recipes[start : start + batch_size]
        texts = [build_text(r) for r in batch]
        embeddings = embedder.encode(texts)
        for recipe, text, embedding in zip(batch, texts, embeddings):
            yield {
                "_id": recipe["recipe_id"],
                "_source": {
                    "recipe_id": recipe["recipe_id"],
                    "recipe_name": recipe["recipe_name"],
                    "source": recipe.get("source"),
                    "url": recipe.get("url"),
                    "text": text,
                    "allergens": normalize_values(recipe.get("cautions")),
                    "diet_labels": normalize_values(recipe.get("diet_labels")),
                    "cuisine_type": normalize_values(recipe.get("cuisine_type")),
                    "embedding": to_list(embedding),
                },
            }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    data_file = resolve_data_file(args)

    settings = load_settings()
    require_configured(settings, need_bedrock=False)
    index_name = args.index_name or settings.opensearch_hybrid_index

    client = create_client(settings)
    if not prepare_target_indexes(
        client, [index_name], recreate=args.recreate, force=args.force
    ):
        return

    embedder = EmbeddingModel(settings)
    ensure_hybrid_index(client, index_name, embedder.dimension, settings)
    ensure_hybrid_pipeline(
        client,
        bm25_weight=args.bm25_weight,
        vector_weight=args.vector_weight,
        settings=settings,
    )

    recipes = load_recipes(data_file)
    LOGGER.info("Ingesting %d recipes from %s", len(recipes), data_file.name)

    before = remote_build_stats(client)
    total = bulk_index(
        client,
        index_name,
        _actions(recipes, embedder, args.batch_size),
        batch_size=settings.ingest_bulk_batch_size,
        total=len(recipes),
    )
    LOGGER.info("Ingested %d recipes into hybrid index '%s'", total, index_name)

    finish_vector_ingest(client, index_name, before, force_merge=not args.no_force_merge)


if __name__ == "__main__":
    main()
