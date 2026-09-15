#!/usr/bin/env python
"""Embed the sample corpus and load it into Cassandra."""

from __future__ import annotations

import json
from pathlib import Path

from cassandra.query import SimpleStatement

from common import TABLE, embed_texts, ensure_schema, wait_for_cassandra

CORPUS_PATH = Path(__file__).resolve().parents[1] / "data" / "corpus.json"


def main() -> None:
    docs = json.loads(CORPUS_PATH.read_text())
    print(f"Loaded {len(docs)} docs from {CORPUS_PATH}")

    session = wait_for_cassandra()
    ensure_schema(session)

    print("Embedding with sentence-transformers (first run downloads the model)...")
    embeddings = embed_texts([f"{doc['title']}\n{doc['body']}" for doc in docs])

    # Truncate so re-seeding is idempotent.
    session.execute(f"TRUNCATE {TABLE}")

    insert = session.prepare(
        f"""
        INSERT INTO {TABLE} (
            id, title, body, category, keywords, model, model_year, embedding
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
    )
    for doc, embedding in zip(docs, embeddings):
        session.execute(
            insert,
            (
                doc["id"],
                doc["title"],
                doc["body"],
                doc["category"],
                set(doc["keywords"]),
                doc.get("model"),
                doc.get("model_year"),
                embedding,
            ),
        )
        print(f"  inserted {doc['id']}")

    count = session.execute(SimpleStatement(f"SELECT COUNT(*) FROM {TABLE}")).one()[0]
    print(f"Seed complete. Row count: {count}")


if __name__ == "__main__":
    main()
