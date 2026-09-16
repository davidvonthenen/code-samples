#!/usr/bin/env python
"""Print all three retrieval lanes for one question or the scripted set."""

from __future__ import annotations

import argparse

from common import ensure_schema, wait_for_cassandra
from retrieval import (
    DEFAULT_QUERIES,
    Hit,
    RetrievalResult,
    filtered_where,
    retrieve_all,
)


def describe(hit: Hit) -> str:
    vehicle = (
        f" [{hit.model or '?'} {hit.model_year or '?'}]"
        if hit.model or hit.model_year
        else ""
    )
    notes = []
    if hit.score is not None:
        notes.append(f"score={hit.score:.4f}")
    if hit.matched is not None:
        notes.append(f"matched {hit.matched} token{'s' if hit.matched != 1 else ''}")
    suffix = f"  {'  '.join(notes)}" if notes else ""
    return f"[{hit.category}] {hit.id}{vehicle} {hit.title}{suffix}"


def print_lane(heading: str, hits: list[Hit]) -> None:
    print(f"\n--- {heading} ---")
    if not hits:
        print("  (no hits)")
        return
    for rank, hit in enumerate(hits, start=1):
        print(f"  {rank}. {describe(hit)}")


def print_result(result: RetrievalResult) -> None:
    print("\n" + "=" * 72)
    print(f"QUERY: {result.query}")
    print(f"Exact tokens: {', '.join(result.tokens) or '(none detected)'}")
    print(
        f"Selective token: {result.keyword or '(none)'}   "
        f"Inferred category: {result.category or '(none)'}"
    )

    where = filtered_where(
        result.keyword, result.category, result.model, result.model_year
    )
    # Hybrid leads, then the two lanes it combines. That lane is skipped when
    # nothing in the question is selective enough to filter on, so the numbering
    # is derived rather than hardcoded.
    lanes = []
    if where:
        lanes.append((f"Hybrid: WHERE {where} + ANN", result.filtered_hits))
    lanes += [
        ("Keyword grounding: keywords CONTAINS", result.keyword_hits),
        ("Vector ANN: ORDER BY embedding ANN OF", result.vector_hits),
    ]
    for number, (heading, hits) in enumerate(lanes, start=1):
        print_lane(f"{number}) {heading}", hits)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cassandra hybrid retrieval demo."
    )
    parser.add_argument(
        "query",
        nargs="*",
        help="Question to run. If omitted, runs the three scripted questions.",
    )
    parser.add_argument(
        "--model",
        help="Customer's truck model, applied to the hybrid lane (1500, 2500).",
    )
    parser.add_argument(
        "--model-year",
        type=int,
        help="Customer's model year, applied to the hybrid lane.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session = wait_for_cassandra()
    ensure_schema(session)
    queries = [" ".join(args.query)] if args.query else DEFAULT_QUERIES

    for query in queries:
        print_result(
            retrieve_all(
                session,
                query,
                model=args.model,
                model_year=args.model_year,
            )
        )


if __name__ == "__main__":
    main()
