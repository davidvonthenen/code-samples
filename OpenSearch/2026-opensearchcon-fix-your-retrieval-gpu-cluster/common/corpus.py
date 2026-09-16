"""Recipe corpus selection shared by every stage's ingest script.

Two corpora exist:

* the full ~39k-recipe corpus written by ``download_dataset.py`` -- what the
  demos are documented and tuned against, and large enough that a vector
  index clears the 50mb default at which OpenSearch hands index construction
  to the remote (GPU) build service;
* the curated 260-recipe sample each stage ships with -- a fast smoke test,
  too small to reproduce the retrieval failures the session is about.

The sample is what you get by *omitting* ``--full``, so anything that reports
results should pass it explicitly; the two produce very different behavior.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
FULL_CORPUS = ROOT / "recipes" / "recipes_full.json"


def add_corpus_args(parser: argparse.ArgumentParser, stage_dir: Path) -> None:
    parser.add_argument(
        "--data-file",
        type=str,
        default=str(stage_dir / "recipes" / "recipes_sample.json"),
        help="Recipe JSON to ingest (default: this stage's curated 260-recipe sample)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help=f"Use the full corpus at {FULL_CORPUS.relative_to(ROOT)} instead "
        "(run download_dataset.py first). Required to trigger a GPU build.",
    )


def add_reingest_args(parser: argparse.ArgumentParser) -> None:
    """Flags every ingest script shares for an already-populated index."""
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete the target index/indexes first, then rebuild. Needed after "
        "changing vector settings in .env or switching between corpora, since "
        "index settings are only applied at creation.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ingest into an already-populated index, overwriting documents in "
        "place without recreating it.",
    )


def resolve_data_file(args: argparse.Namespace) -> Path:
    if getattr(args, "full", False):
        if not FULL_CORPUS.exists():
            raise FileNotFoundError(
                f"{FULL_CORPUS} not found. Run:  python download_dataset.py"
            )
        return FULL_CORPUS
    path = Path(args.data_file)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. See README.md.")
    return path


def load_recipes(path: Path) -> List[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = [
    "add_corpus_args",
    "add_reingest_args",
    "resolve_data_file",
    "load_recipes",
    "FULL_CORPUS",
]
