#!/usr/bin/env python3
"""Stage 2: index the recipe corpus for BM25 + NER-entity retrieval.

Every recipe is tagged with entities extracted by the local NER service,
plus its own source name folded in directly: spaCy's small model does not
reliably recognize niche food-blog/brand names (e.g. "Skinnytaste",
"sargento.com") as formal ORG entities, but the source is already known,
deterministic metadata -- so it's added to the same `entities` field NER
populates rather than left to chance.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from common.config import load_settings, require_configured
from common.corpus import add_corpus_args, add_reingest_args, load_recipes, resolve_data_file
from common.labels import normalize_values
from common.logging import get_logger
from common.ner_client import NERClient
from common.opensearch_client import (
    bulk_index,
    create_client,
    ensure_recipe_bm25_index,
    prepare_target_indexes,
    restore_refresh,
)

LOGGER = get_logger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest recipes for BM25 + NER retrieval")
    add_corpus_args(parser, Path(__file__).parent)
    add_reingest_args(parser)
    parser.add_argument("--index-name", type=str, default=None)
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


def _actions(recipes, ner):
    for recipe in recipes:
        text = build_text(recipe)
        ner_entities = ner.extract_entities(text)
        entities = normalize_values(ner_entities + [recipe["source"], recipe["recipe_name"]])
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
                "entities": entities,
                "cautions": normalize_values(recipe.get("cautions")),
                "cautions_display": recipe.get("cautions") or [],
                "diet_labels": normalize_values(recipe.get("diet_labels")),
                "diet_labels_display": recipe.get("diet_labels") or [],
                "health_labels": normalize_values(recipe.get("health_labels")),
                "health_labels_display": recipe.get("health_labels") or [],
                "cuisine_type": normalize_values(recipe.get("cuisine_type")),
                "meal_type": normalize_values(recipe.get("meal_type")),
                "dish_type": normalize_values(recipe.get("dish_type")),
            },
        }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    data_file = resolve_data_file(args)

    settings = load_settings()
    require_configured(settings, need_bedrock=False)
    client = create_client(settings)
    index_name = args.index_name or settings.opensearch_bm25_recipes_baseline_index
    if not prepare_target_indexes(
        client, [index_name], recreate=args.recreate, force=args.force
    ):
        return

    ner = NERClient(settings)
    ensure_recipe_bm25_index(client, index_name, settings)

    recipes = load_recipes(data_file)
    LOGGER.info(
        "Ingesting %d recipes from %s (one NER call each -- the slow part)",
        len(recipes),
        data_file.name,
    )

    count = bulk_index(
        client,
        index_name,
        _actions(recipes, ner),
        batch_size=settings.ingest_bulk_batch_size,
        total=len(recipes),
    )

    restore_refresh(client, index_name)
    LOGGER.info("Ingested %d recipes into '%s'", count, index_name)


if __name__ == "__main__":
    main()
