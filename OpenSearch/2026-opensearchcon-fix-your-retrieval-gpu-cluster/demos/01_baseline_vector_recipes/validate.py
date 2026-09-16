#!/usr/bin/env python3
"""Pass/fail smoke test for Stage 1. Run after ingest.py."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from query import ask

TARGET_SOURCE = "thegratefulgirlcooks.com"
QUESTION = f"Does the {TARGET_SOURCE} Fish and Chips recipe contain sulfites?"


def _rank_of_source(hits, source: str) -> int | None:
    for rank, hit in enumerate(hits, start=1):
        if hit.get("source") == source:
            return rank
    return None


def test_stage1_cannot_find_the_named_recipe() -> None:
    """The question names a source, and vector-only search still misses it.

    This is the failure the rest of the talk exists to fix, so the test has to
    assert the failure actually happens. Checking only that hits came back
    would pass just as happily if retrieval started working, and the first
    demo would quietly stop demonstrating anything.

    Holds on both corpora: on the 260-recipe sample the other Fish and Chips
    recipe outranks it, and on the full corpus neither one surfaces at all.
    """
    answer, hits = ask(QUESTION, top_k=5, num_candidates=25)
    assert hits, "Expected at least one retrieved recipe -- did ingest.py run?"
    assert answer, "Expected a non-empty answer from Bedrock"

    rank = _rank_of_source(hits, TARGET_SOURCE)
    assert rank is None, (
        f"Vector-only search retrieved the {TARGET_SOURCE} recipe at rank {rank}, "
        "so this stage no longer demonstrates the retrieval failure the talk is "
        f"built on. Got: {[h.get('source') for h in hits]}. Check that ingest.py "
        "loaded the expected corpus."
    )

    print(
        f"[PASS] Stage 1 missed the {TARGET_SOURCE} recipe as expected; "
        f"returned {len(hits)} semantically similar recipes instead "
        f"({', '.join(h.get('source', '?') for h in hits)})"
    )


if __name__ == "__main__":
    test_stage1_cannot_find_the_named_recipe()
