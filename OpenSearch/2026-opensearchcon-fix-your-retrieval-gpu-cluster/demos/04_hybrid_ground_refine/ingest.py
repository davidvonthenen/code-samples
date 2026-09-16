#!/usr/bin/env python3
"""Stage 4: index the recipe corpus into TWO indexes -- BM25
(`recipes-bm25`) and vector (`recipes-vector`) -- for an externally
orchestrated (not OpenSearch-native) two-phase retrieval pipeline.

Same "Recipe & Dietary Safety Assistant" theme as Stage 1/2/3. Unlike
Stage 3's single native hybrid query, query.py runs BM25 first as a
*grounding* step (with hard structured filters -- see README.md) to build a
provably-safe candidate set, then re-ranks that set by vector similarity in
a second, separate query. Both indexes therefore need the same document
written into them, tagged with the same structured metadata (cautions,
diet_labels, etc.) so filters behave identically in either query.

`entities` (BM25-only) is populated the same way as Stage 2: NER-extracted
entities plus the recipe's own `source` and `recipe_name` folded in, since
spaCy's small model misses niche food-blog domains.
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
from common.ner_client import NERClient
from common.opensearch_client import (
    bulk_index,
    create_client,
    ensure_recipe_bm25_index,
    ensure_recipe_vector_index,
    finish_vector_ingest,
    prepare_target_indexes,
    remote_build_stats,
    restore_refresh,
)

LOGGER = get_logger(__name__)

DEFAULT_TIER = "hot"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest recipes into the BM25 + vector indexes")
    add_corpus_args(parser, Path(__file__).parent)
    add_reingest_args(parser)
    parser.add_argument("--bm25-index-name", type=str, default=None)
    parser.add_argument("--vector-index-name", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=256, help="Embedding batch size")
    parser.add_argument(
        "--no-force-merge",
        action="store_true",
        help="Skip the post-ingest force merge (which is what triggers a merge-path GPU build)",
    )
    return parser.parse_args(argv)


def build_text(recipe: dict) -> str:
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


def _shared_fields(recipe: dict, text: str) -> dict:
    """Fields common to both the BM25 and vector documents for one recipe.

    chunk_id/chunk_index/chunk_count are always a single chunk here (one
    document per recipe, like Stage 1/2) -- they exist in the schema so a
    future stage could split a recipe's instructions into multiple chunks
    without changing the index mapping or query helpers.
    """
    return {
        "recipe_id": recipe["recipe_id"],
        "recipe_name": recipe["recipe_name"],
        "source": recipe.get("source"),
        "url": recipe.get("url"),
        "image_url": recipe.get("image_url"),
        "servings": recipe.get("servings"),
        "calories": recipe.get("calories"),
        "chunk_id": recipe["recipe_id"],
        "chunk_index": 0,
        "chunk_count": 1,
        "text": text,
        "cautions": normalize_values(recipe.get("cautions")),
        "cautions_display": recipe.get("cautions") or [],
        "diet_labels": normalize_values(recipe.get("diet_labels")),
        "diet_labels_display": recipe.get("diet_labels") or [],
        "health_labels": normalize_values(recipe.get("health_labels")),
        "health_labels_display": recipe.get("health_labels") or [],
        "cuisine_type": normalize_values(recipe.get("cuisine_type")),
        "meal_type": normalize_values(recipe.get("meal_type")),
        "dish_type": normalize_values(recipe.get("dish_type")),
        "tier": DEFAULT_TIER,
    }


def _bm25_actions(recipes, ner):
    for recipe in recipes:
        text = build_text(recipe)
        ner_entities = ner.extract_entities(text)
        entities = normalize_values(ner_entities + [recipe["source"], recipe["recipe_name"]])
        yield {"_id": recipe["recipe_id"], "_source": {**_shared_fields(recipe, text), "entities": entities}}


def _vector_actions(recipes, embedder, batch_size):
    for start in range(0, len(recipes), batch_size):
        batch = recipes[start : start + batch_size]
        texts = [build_text(r) for r in batch]
        embeddings = embedder.encode(texts)
        for recipe, text, embedding in zip(batch, texts, embeddings):
            yield {
                "_id": recipe["recipe_id"],
                "_source": {**_shared_fields(recipe, text), "embedding": to_list(embedding)},
            }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    data_file = resolve_data_file(args)

    settings = load_settings()
    require_configured(settings, need_bedrock=False)
    bm25_index = args.bm25_index_name or settings.opensearch_recipe_bm25_index
    vector_index = args.vector_index_name or settings.opensearch_recipe_vector_index

    client = create_client(settings)
    if not prepare_target_indexes(
        client, [bm25_index, vector_index], recreate=args.recreate, force=args.force
    ):
        return

    ner = NERClient(settings)
    embedder = EmbeddingModel(settings)
    ensure_recipe_bm25_index(client, bm25_index, settings)
    ensure_recipe_vector_index(client, vector_index, embedder.dimension, settings)

    recipes = load_recipes(data_file)
    LOGGER.info("Ingesting %d recipes from %s into two indexes", len(recipes), data_file.name)

    # The two indexes are loaded separately rather than interleaved so the
    # vector index's segments -- and therefore its GPU build -- are not
    # entangled with NER latency on the BM25 side.
    bm25_total = bulk_index(
        client,
        bm25_index,
        _bm25_actions(recipes, ner),
        batch_size=settings.ingest_bulk_batch_size,
        total=len(recipes),
        desc="BM25",
    )
    restore_refresh(client, bm25_index)

    before = remote_build_stats(client)
    vector_total = bulk_index(
        client,
        vector_index,
        _vector_actions(recipes, embedder, args.batch_size),
        batch_size=settings.ingest_bulk_batch_size,
        total=len(recipes),
        desc="Vector",
    )
    finish_vector_ingest(client, vector_index, before, force_merge=not args.no_force_merge)

    LOGGER.info(
        "Ingested %d into '%s' and %d into '%s'", bm25_total, bm25_index, vector_total, vector_index
    )


if __name__ == "__main__":
    main()
