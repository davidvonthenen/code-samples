#!/usr/bin/env python
"""Score every retrieval lane on the same task: look a recall up by its number.

Each recall document is worded from the same template, so the recall ID is the
only thing that distinguishes it from eleven near-twins. That makes this a clean
test of which lanes can resolve an identifier.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from common import ensure_schema, wait_for_cassandra
from retrieval import (
    MAX_LIMIT,
    filtered_ann,
    infer_category,
    keyword_search,
    selective_token,
    vector_search,
)

CORPUS_PATH = Path(__file__).resolve().parents[1] / "data" / "corpus.json"
RECALL_ID = re.compile(r"\b(\d{2}V-\d{3})\b")

LANES = ("keyword", "vector", "hybrid")


def recall_ids() -> list[tuple[str, str]]:
    """Return (doc_id, recall_id) for every recall document in the corpus."""
    pairs = []
    for doc in json.loads(CORPUS_PATH.read_text()):
        match = RECALL_ID.search(doc["title"])
        if match:
            pairs.append((doc["id"], match.group(1)))
    return pairs


def rank_of(doc_id: str, hits) -> int | None:
    ids = [hit.id for hit in hits]
    return ids.index(doc_id) + 1 if doc_id in ids else None


def main() -> None:
    session = wait_for_cassandra()
    ensure_schema(session)

    pairs = recall_ids()
    if not pairs:
        print(f"No recall documents found in {CORPUS_PATH}")
        return

    total = len(pairs)
    ranks: dict[str, list[int | None]] = {lane: [] for lane in LANES}

    print(f"Asking for {total} recall IDs by number, LIMIT {MAX_LIMIT}")
    print("Rank of the exactly matching document in each lane:\n")
    print("recall     " + "".join(f"{lane:>15}" for lane in LANES))

    for doc_id, recall_id in pairs:
        query = f"What is recall {recall_id}?"
        keyword_hits = keyword_search(session, query, MAX_LIMIT)
        vector_hits = vector_search(session, query, MAX_LIMIT)
        lane_hits = {
            "keyword": keyword_hits,
            "vector": vector_hits,
            "hybrid": filtered_ann(
                session,
                query,
                keyword=selective_token(query),
                category=infer_category(query),
                limit=MAX_LIMIT,
            ),
        }
        cells = []
        for lane in LANES:
            rank = rank_of(doc_id, lane_hits[lane])
            ranks[lane].append(rank)
            cells.append(f"{rank}" if rank else f">{MAX_LIMIT}")
        print(f"{recall_id:<10} " + "".join(f"{cell:>15}" for cell in cells))

    print()
    for lane in LANES:
        found = [rank for rank in ranks[lane] if rank is not None]
        at_1 = sum(1 for rank in found if rank == 1)
        at_3 = sum(1 for rank in found if rank <= 3)
        mean_rank = f"{sum(found) / len(found):.2f}" if found else "n/a"
        missed = total - len(found)
        print(
            f"{lane:<14} "
            f"recall@1 = {at_1}/{total} ({at_1 / total:.0%})   "
            f"recall@3 = {at_3}/{total} ({at_3 / total:.0%})   "
            f"mean rank = {mean_rank}"
            + (f"   ({missed} outside top {MAX_LIMIT})" if missed else "")
        )

    print(
        "\nMean rank counts only the documents a lane returned, so a lane that "
        "misses is flattered by it."
    )


if __name__ == "__main__":
    main()
