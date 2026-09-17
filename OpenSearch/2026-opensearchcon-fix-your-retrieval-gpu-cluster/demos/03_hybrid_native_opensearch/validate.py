#!/usr/bin/env python3
"""Pass/fail smoke test for Stage 3. Run after ingest.py."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from query import ask


def _rank_of_source(hits, source: str) -> int | None:
    for rank, hit in enumerate(hits, start=1):
        if hit.get("_source", {}).get("source") == source:
            return rank
    return None


def test_stage3_disambiguates_similar_questions() -> None:
    """Each question names a source; hybrid search has to retrieve *that* recipe.

    Checking only that hits came back is not enough: the whole point of this
    stage is that fusing BM25 with vector search pulls the named recipe into
    the results, and that can fail while still returning five hits.
    """
    cases = [
        ("thegratefulgirlcooks.com", "Does the thegratefulgirlcooks.com Fish and Chips recipe contain sulfites?"),
        ("womensweeklyfood.com.au", "Does the womensweeklyfood.com.au Fish and Chips recipe contain sulfites?"),
    ]

    for source, question in cases:
        answer, hits = ask(question, top_k=5)
        assert hits, f"Expected hybrid hits for {source!r} -- did ingest.py run?"
        assert answer, f"Expected a non-empty Bedrock answer for {source!r}"
        rank = _rank_of_source(hits, source)
        assert rank is not None, (
            f"Hybrid search did not retrieve the {source} recipe at all. "
            f"Got: {[h['_source'].get('source') for h in hits]}. "
            "A keyword-only match cannot score above HYBRID_BM25_WEIGHT, so if that "
            "weight is too low the named recipe loses to merely-similar documents."
        )
        print(f"[ok] {source} retrieved at rank {rank} of {len(hits)}")

    print("[PASS] Stage 3 retrieved the named recipe for both disambiguation questions")


if __name__ == "__main__":
    test_stage3_disambiguates_similar_questions()
