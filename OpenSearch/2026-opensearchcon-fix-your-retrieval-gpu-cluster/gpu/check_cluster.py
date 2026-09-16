#!/usr/bin/env python3
"""Pre-flight check: is this cluster actually able to run GPU index builds?

Run this before ingesting. If the remote build service is not wired up
cluster-side, every index setting downstream is inert and you will not find
out until after a full corpus load.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import load_settings, require_configured
from common.corpus import FULL_CORPUS
from common.opensearch_client import create_client

OK = "  ok   "
WARN = " warn  "
FAIL = " FAIL  "

_UNITS = {"b": 1, "kb": 1024, "mb": 1024**2, "gb": 1024**3, "tb": 1024**4}


def parse_size(value: str) -> int:
    text = str(value).strip().lower()
    for suffix in ("tb", "gb", "mb", "kb", "b"):
        if text.endswith(suffix):
            return int(float(text[: -len(suffix)]) * _UNITS[suffix])
    return int(float(text))


def flat_cluster_settings(client) -> Dict[str, Any]:
    response = client.transport.perform_request(
        "GET", "/_cluster/settings?include_defaults=true&flat_settings=true"
    )
    merged: Dict[str, Any] = {}
    for scope in ("defaults", "persistent", "transient"):
        merged.update(response.get(scope) or {})
    return merged


def main() -> int:
    settings = load_settings()
    require_configured(settings, need_bedrock=False)
    client = create_client(settings)

    failures = 0

    health = client.cluster.health()
    print(f"[{OK}] cluster '{health['cluster_name']}' is {health['status']} "
          f"({health['number_of_data_nodes']} data nodes)")

    cluster = flat_cluster_settings(client)
    enabled = str(cluster.get("knn.remote_index_build.enabled", "false")).lower() == "true"
    repository = cluster.get("knn.remote_index_build.repository", "")
    endpoint = cluster.get("knn.remote_index_build.service.endpoint", "")
    poll = cluster.get("knn.remote_index_build.poll.interval", "?")
    size_max = cluster.get("knn.remote_index_build.size.max", "?")

    if enabled:
        print(f"[{OK}] knn.remote_index_build.enabled = true")
    else:
        print(f"[{FAIL}] knn.remote_index_build.enabled is not true -- no GPU build can trigger")
        failures += 1

    if repository:
        print(f"[{OK}] vector repository = '{repository}'")
        try:
            client.transport.perform_request("GET", f"/_snapshot/{repository}")
            print(f"[{OK}] repository '{repository}' is registered")
        except Exception as exc:  # noqa: BLE001 - surfacing any transport error is the point
            print(f"[{FAIL}] repository '{repository}' is not readable: {exc}")
            failures += 1
    else:
        print(f"[{FAIL}] knn.remote_index_build.repository is unset")
        failures += 1

    if endpoint:
        print(f"[{OK}] build service endpoint = {endpoint}")
    else:
        print(f"[{FAIL}] knn.remote_index_build.service.endpoint is unset")
        failures += 1

    print(f"[{OK}] poll interval = {poll}, cluster size.max = {size_max}")

    # --- local settings that decide whether a build can trigger ---
    print()
    print(f"[{OK}] engine={settings.vector_engine} space={settings.vector_space_type} "
          f"derived_source={settings.knn_derived_source} replicas={settings.knn_number_of_replicas}")

    if settings.vector_engine != "faiss":
        print(f"[{FAIL}] VECTOR_ENGINE is '{settings.vector_engine}' -- the remote build "
              "service only supports faiss")
        failures += 1

    if not settings.knn_remote_build:
        print(f"[{WARN}] KNN_REMOTE_BUILD is false -- indexes will build locally on CPU")

    if settings.knn_approximate_threshold != 0:
        print(f"[{WARN}] KNN_APPROXIMATE_THRESHOLD={settings.knn_approximate_threshold}: "
              "indexes smaller than this build no graph at all, so no remote build triggers")

    # --- how much data is needed to clear the threshold ---
    per_doc = settings.embedding_dimension * 4
    needed = parse_size(settings.knn_remote_build_size_min) // per_doc
    print(f"[{OK}] size.min={settings.knn_remote_build_size_min} needs ~{needed:,} docs "
          f"at {settings.embedding_dimension} dims ({per_doc} bytes/doc)")

    if FULL_CORPUS.exists():
        import json

        count = len(json.loads(FULL_CORPUS.read_text(encoding="utf-8")))
        verdict = OK if count >= needed else FAIL
        print(f"[{verdict}] full corpus has {count:,} recipes "
              f"({count * per_doc / 1024**2:.1f} MiB of vector data)")
        if count < needed:
            failures += 1
    else:
        print(f"[{WARN}] full corpus not downloaded yet -- the 260-recipe sample "
              f"({260 * per_doc / 1024:.0f} KB) will not trigger a build. "
              "Run: python download_dataset.py")

    print()
    print("PASS -- ready to ingest" if failures == 0 else f"{failures} blocking problem(s) found")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
