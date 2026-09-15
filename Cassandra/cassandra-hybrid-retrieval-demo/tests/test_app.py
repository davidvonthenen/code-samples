from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from presenter import QueryRequest, app  # noqa: E402
from retrieval import (  # noqa: E402
    DEFAULT_QUERIES,
    extract_keyword_tokens,
    filtered_ann_cql,
    infer_category,
    selective_token,
)


class TokenTests(unittest.TestCase):
    def test_keeps_identifiers_numbers_and_known_terms(self) -> None:
        self.assertEqual(
            extract_keyword_tokens("What is recall 24V-330 for a Summit 1500?"),
            ["recall", "24v-330", "summit", "1500"],
        )

    def test_deduplicates_without_reordering(self) -> None:
        self.assertEqual(
            extract_keyword_tokens("Tow the Summit 1500: tow or towing?"),
            ["tow", "summit", "1500", "towing"],
        )

    def test_selective_token_prefers_the_identifier(self) -> None:
        self.assertEqual(selective_token("What is recall 24V-330?"), "24v-330")

    def test_selective_token_ignores_bare_nouns_and_years(self) -> None:
        """`keywords CONTAINS 'recall'` matches all twelve recalls, so filtering
        on it would buy nothing."""
        self.assertIsNone(selective_token("Is there a 2024 recall?"))

    def test_infers_scripted_categories(self) -> None:
        cases = {
            "Is there a tailgate recall?": "recalls",
            "How much can it tow?": "towing",
            "When should I change the oil?": "maintenance",
            "What does the warranty cover?": "warranty",
            "What is the payload?": "specs",
            "Tell me about this truck": None,
        }
        for query, expected in cases.items():
            with self.subTest(query=query):
                self.assertEqual(infer_category(query), expected)


class FilteredAnnTests(unittest.TestCase):
    def test_renders_the_statement_with_values_inlined(self) -> None:
        cql = filtered_ann_cql(keyword="24v-330", category="recalls", model_year=2024)
        self.assertIn(
            "WHERE keywords CONTAINS '24v-330' AND category = 'recalls' "
            "AND model_year = 2024",
            cql,
        )
        self.assertIn("ORDER BY embedding ANN OF ?", cql)

    def test_no_filters_means_no_statement(self) -> None:
        self.assertIsNone(filtered_ann_cql())


class PresenterTests(unittest.TestCase):
    def test_exposes_only_retrieval_routes(self) -> None:
        paths = {route.path for route in app.routes}
        self.assertIn("/api/query", paths)
        self.assertNotIn("/api/generate", paths)

    def test_rejects_blank_input(self) -> None:
        with self.assertRaises(ValueError):
            QueryRequest(query="")


class CorpusTests(unittest.TestCase):
    def setUp(self) -> None:
        self.docs = json.loads((ROOT / "data" / "corpus.json").read_text())

    def test_scripted_queries_cover_the_three_failure_modes(self) -> None:
        self.assertEqual(len(DEFAULT_QUERIES), 3)
        self.assertIn("24V-330", DEFAULT_QUERIES[0])
        self.assertIn("tailgate", DEFAULT_QUERIES[1])
        self.assertIn("tow", DEFAULT_QUERIES[2])

    def test_documents_are_unique(self) -> None:
        ids = [doc["id"] for doc in self.docs]
        self.assertEqual(len(ids), len(set(ids)))

    def test_enough_recalls_to_show_the_failure(self) -> None:
        """Below roughly eight similar recalls, ANN ranks identifiers correctly
        every time and the demo's central point disappears."""
        recalls = [doc for doc in self.docs if doc["category"] == "recalls"]
        self.assertGreaterEqual(len(recalls), 8)

    def test_every_recall_id_is_a_keyword(self) -> None:
        """The keyword lane can only ground an identifier that was indexed."""
        for doc in self.docs:
            match = re.search(r"\b(\d{2}V-\d{3})\b", doc["title"])
            if match:
                with self.subTest(doc=doc["id"]):
                    self.assertIn(match.group(1).lower(), doc["keywords"])

    def test_keywords_are_lowercase_for_exact_matching(self) -> None:
        for doc in self.docs:
            with self.subTest(doc=doc["id"]):
                self.assertEqual(
                    doc["keywords"], [word.lower() for word in doc["keywords"]]
                )


if __name__ == "__main__":
    unittest.main()
