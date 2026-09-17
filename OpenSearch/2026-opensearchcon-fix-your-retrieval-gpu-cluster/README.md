# OpenSearchCon 2026: Fix Your Retrieval, Fix Your RAG

Welcome to the landing page for the session `Fix Your Retrieval, Fix Your RAG` at `OpenSearchCon 2026`.

## What to Expect

This repo walks through four runnable stages of a Recipe & Dietary Safety
Assistant, each fixing one concrete retrieval failure of the stage before
it:

- Section 1: Vector search retrieves what's similar, not what's true
- Section 2: Fixing that with BM25 + NER entity matching works, but needs an extra microservice
- Section 3: Fusing BM25 + vector server-side in one OpenSearch query removes that extra service
- Section 4: Even hybrid scores can't guarantee a hard exclusion like "no sulfites" -- that needs real filters, not scores

## Prerequisites

- A Mac or Linux developer laptop (Windows users should use a VM or cloud instance)
- Python 3.10 or higher
  - (Recommended) Use miniconda, uv, or venv for an isolated environment
- An [Instaclustr](https://www.instaclustr.com/) OpenSearch cluster (free tier works; every stage shares one cluster, just different index names)
- An AWS account with **Bedrock model access enabled** for an Anthropic Claude model

## Installation

```bash
git clone https://github.com/instaclustr/code-samples.git
cd code-samples/OpenSearch/2026-opensearchcon-fix-your-retrieval-gpu-cluster
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.5.0/en_core_web_sm-3.5.0.tar.gz
```

The spaCy model is installed by URL because `python -m spacy download` fails
on newer pip with `Invalid requirement: '==en_core_web_sm'`. Sections 2 and 4
need it; Sections 1 and 3 do not.

Then configure your credentials:

```bash
cp .env.example .env
# edit .env: OPENSEARCH_HOST/USER/PASSWORD, AWS_ACCESS_KEY_ID/SECRET_ACCESS_KEY, AWS_REGION, BEDROCK_MODEL_ID
```

## Download the Corpus

```bash
python download_dataset.py
```

Fetches all 39,445 recipes from
[`datahiveai/recipes-with-nutrition`](https://huggingface.co/datasets/datahiveai/recipes-with-nutrition)
into `recipes/recipes_full.json`. Every section reads from it, so this only
needs to run once.

## Session Materials

- Section 1: [Baseline Vector RAG](./demos/01_baseline_vector_recipes/README.md)
- Section 2: [Baseline BM25 + NER RAG](./demos/02_baseline_bm25_ner_recipes/README.md)
- Section 3: [Hybrid RAG (Native OpenSearch)](./demos/03_hybrid_native_opensearch/README.md)
- Section 4: [Hybrid RAG (Ground + Refine)](./demos/04_hybrid_ground_refine/README.md)

The instructions and purpose for each stage are contained within their
respective folders.

## The Corpus

All four sections run against the same 39,445 recipes and take `--full` on
ingest. Retrieval failures are only convincing at that size, because a
realistic corpus contains many near-misses rather than a handful: Section 1
answers a question about a specific Fish and Chips recipe by retrieving
*potato chips*.

Loading is the slow part -- roughly 6, 12, 6 and 17 minutes for Sections 1
through 4, since Sections 2 and 4 make one NER call per recipe. Plan on about
40 minutes to load all four, then demo against the loaded indexes.

Ingest refuses to write into an index that already holds documents, so a
second run prints `Already ingested` and stops rather than silently mixing
data. Add `--recreate` to drop the index and reload it.

Each section also ships a 260-recipe sample in its `recipes/` folder, which is
what ingest loads if you omit `--full`. It is too small to reproduce the
retrieval failures this session is about, and too small to trigger a GPU
build, so it is not part of the walkthrough. It exists as a fast smoke test,
and it is the baseline the docs compare against when explaining why Section
3's BM25 weight is tuned the way it is.

## Running on a GPU-Accelerated Cluster

Vector index construction can be offloaded to OpenSearch's Dedicated Vector
Index Builder. Nothing in the four sections changes; index settings are read
from `.env`. See [GPU.md](./GPU.md) for the configuration and the verification
tooling under `gpu/`.

## License

Apache License 2.0. See [LICENSE](./LICENSE).
