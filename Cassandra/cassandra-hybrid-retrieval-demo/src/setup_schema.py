#!/usr/bin/env python
"""Create the keyspace, table, and SAI indexes for the hybrid retrieval demo."""

from common import KEYSPACE, TABLE, ensure_schema, wait_for_cassandra


def main() -> None:
    print("Waiting for Cassandra...")
    session = wait_for_cassandra()
    print("Applying schema and SAI indexes...")
    ensure_schema(session)
    print(f"Done. Keyspace: {KEYSPACE}  Table: {TABLE}")
    print("Indexes: keywords, category, model, model_year, embedding (ANN)")


if __name__ == "__main__":
    main()
