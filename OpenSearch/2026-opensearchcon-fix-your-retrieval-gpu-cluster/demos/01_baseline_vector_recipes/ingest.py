#!/usr/bin/env python3
"""Stage 1: embed and index the recipe corpus into Instaclustr OpenSearch.

Teaching goal: vector-only retrieval matches by *semantic similarity*, not
by which recipe is actually the correct one to answer about. See README.md
for the real, naturally-occurring ambiguous pair this stage uses (two
"Fish and Chips" recipes from different sources, only one of which
contains sulfites).
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
    ensure_recipe_vector_index,
    finish_vector_ingest,
    prepare_target_indexes,
    remote_build_stats,
)

LOGGER = get_logger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest recipes into the vector index")
    add_corpus_args(parser, Path(__file__).parent)
    add_reingest_args(parser)
    parser.add_argument("--index-name", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=256, help="Embedding batch size")
    parser.add_argument(
        "--no-force-merge",
        action="store_true",
        help="Skip the post-ingest force merge (which is what triggers a merge-path GPU build)",
    )
    return parser.parse_args(argv)


def build_text(recipe: dict) -> str:
    """Render one recipe into the text that gets embedded and shown to the LLM.

    Cautions/health labels are spelled out in plain English here (not just
    stored as structured fields) so that a question like "does it contain
    soy?" has the literal word "soy" to match against.
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
    """Embed in batches and yield bulk actions one recipe at a time."""
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
                    "source": recipe["source"],
                    "url": recipe.get("url"),
                    "servings": recipe.get("servings"),
                    "calories": recipe.get("calories"),
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
                    "embedding": to_list(embedding),
                },
            }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    data_file = resolve_data_file(args)

    settings = load_settings()
    require_configured(settings, need_bedrock=False)
    index_name = args.index_name or settings.opensearch_vector_recipes_baseline_index

    client = create_client(settings)
    if not prepare_target_indexes(
        client, [index_name], recreate=args.recreate, force=args.force
    ):
        return

    embedder = EmbeddingModel(settings)
    ensure_recipe_vector_index(client, index_name, embedder.dimension, settings)

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
    LOGGER.info("Ingested %d recipes into index '%s'", total, index_name)

    finish_vector_ingest(client, index_name, before, force_merge=not args.no_force_merge)


if __name__ == "__main__":
    main()
