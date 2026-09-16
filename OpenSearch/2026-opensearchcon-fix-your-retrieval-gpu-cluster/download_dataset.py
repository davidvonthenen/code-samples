#!/usr/bin/env python3
"""Download the full recipe corpus so vector indexes are large enough to
trigger a GPU (remote) index build.

The curated 260-recipe sample each stage ships with produces roughly 390 KB of
vector data -- far below the 50 MB default at which OpenSearch hands index
construction to the remote build service. The full dataset is ~39,447 recipes,
or ~57.8 MiB of vector data at 384 dimensions, which clears that default
without lowering any threshold.

Recipes that already appear in the curated sample keep their original
recipe_id, so the deliberately ambiguous "Fish and Chips" pair the demos rely
on stays addressable and is not duplicated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from common.logging import get_logger

LOGGER = get_logger(__name__)

DATASET = "datahiveai/recipes-with-nutrition"
ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "recipes" / "recipes_full.json"
CURATED = ROOT / "demos" / "01_baseline_vector_recipes" / "recipes" / "recipes_sample.json"

# Upstream stores these as JSON-encoded strings rather than native lists.
LIST_FIELDS = (
    "diet_labels",
    "health_labels",
    "cautions",
    "cuisine_type",
    "meal_type",
    "dish_type",
    "ingredient_lines",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only keep the first N recipes (for a quick smoke test)",
    )
    return parser.parse_args(argv)


def _as_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return [str(value)]
    return [str(v) for v in parsed] if isinstance(parsed, list) else []


def _stable_id(url: str, recipe_name: str, source: str) -> str:
    key = f"{url}|{recipe_name}|{source}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


def _curated_ids() -> Dict[tuple, str]:
    """Map (recipe_name, source) -> the recipe_id the curated sample uses."""
    if not CURATED.exists():
        LOGGER.warning("Curated sample not found at %s; using generated ids only", CURATED)
        return {}
    curated = json.loads(CURATED.read_text(encoding="utf-8"))
    return {(r["recipe_name"], r["source"]): r["recipe_id"] for r in curated}


def _convert(row: Dict[str, Any], known_ids: Dict[tuple, str]) -> Optional[Dict[str, Any]]:
    name = (row.get("recipe_name") or "").strip()
    source = (row.get("source") or "").strip()
    if not name or not source:
        return None

    lines = _as_list(row.get("ingredient_lines"))
    if not lines:
        return None

    url = row.get("url") or ""
    recipe_id = known_ids.get((name, source)) or _stable_id(url, name, source)

    return {
        "recipe_id": recipe_id,
        "recipe_name": name,
        "source": source,
        "url": url,
        "image_url": row.get("image_url") or "",
        "servings": row.get("servings"),
        "calories": row.get("calories"),
        "cautions": _as_list(row.get("cautions")),
        "diet_labels": _as_list(row.get("diet_labels")),
        "health_labels": _as_list(row.get("health_labels")),
        "cuisine_type": _as_list(row.get("cuisine_type")),
        "meal_type": _as_list(row.get("meal_type")),
        "dish_type": _as_list(row.get("dish_type")),
        "ingredient_lines": lines,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset

    LOGGER.info("Downloading %s (this pulls ~100MB of parquet on first run)", DATASET)
    dataset = load_dataset(DATASET, split="train")

    known_ids = _curated_ids()
    LOGGER.info("Preserving recipe_id for %d curated recipes", len(known_ids))

    recipes: List[Dict[str, Any]] = []
    seen: set = set()
    skipped = 0
    for row in dataset:
        converted = _convert(row, known_ids)
        if converted is None:
            skipped += 1
            continue
        if converted["recipe_id"] in seen:
            skipped += 1
            continue
        seen.add(converted["recipe_id"])
        recipes.append(converted)
        if args.limit and len(recipes) >= args.limit:
            break

    out_path.write_text(json.dumps(recipes), encoding="utf-8")

    matched = sum(1 for r in recipes if (r["recipe_name"], r["source"]) in known_ids)
    vector_mb = len(recipes) * 384 * 4 / (1024 * 1024)
    LOGGER.info(
        "Wrote %d recipes to %s (%.1f MB on disk)",
        len(recipes),
        out_path,
        out_path.stat().st_size / (1024 * 1024),
    )
    LOGGER.info("Skipped %d rows (missing fields or duplicate id)", skipped)
    LOGGER.info("Curated recipes carried over with original ids: %d/%d", matched, len(known_ids))
    LOGGER.info(
        "Projected vector data per index: %.1f MiB at 384 dims -- %s the 50mb default",
        vector_mb,
        "clears" if vector_mb > 50 else "BELOW",
    )


if __name__ == "__main__":
    main()
