# Running on a GPU-Accelerated OpenSearch Cluster

The four sections in this repo are unchanged when vector index construction is
offloaded to OpenSearch's Dedicated Vector Index Builder. Index settings come
from `.env`, so the same code runs against a laptop cluster and a GPU-enabled
one.

This file is not needed for the session. It documents the configuration and
what was measured on an Instaclustr preprod cluster.

## Why corpus size decides whether a GPU build happens

OpenSearch only sends a segment to the remote build service once its vector
data exceeds `index.knn.remote_index_build.size.min`, which defaults to 50 MB.
At 384 dimensions each document contributes 384 x 4 = 1,536 bytes, so roughly
34,000 documents are needed to reach that.

| Corpus | Vector data | vs 50 MB default |
| --- | --- | --- |
| 260 (curated sample) | 390 KB | 0.7% |
| 39,445 (full) | 57.8 MiB | 116% |

The curated sample that ships in each section cannot trigger a build at any
realistic threshold, which is why every section is run with `--full`.

## Configuration

```bash
VECTOR_ENGINE=faiss                 # the only engine the build service supports
VECTOR_SPACE_TYPE=l2                # embeddings are L2-normalized, so ranking matches cosine
KNN_REMOTE_BUILD=true
KNN_REMOTE_BUILD_SIZE_MIN=50mb      # OpenSearch default
KNN_APPROXIMATE_THRESHOLD=0         # the default (15000) skips graph construction on small
                                    # indexes, so no remote build can trigger
KNN_DERIVED_SOURCE=true             # vectors reconstructed on read rather than stored raw
                                    # in _source; roughly 2.6x smaller indexes
KNN_NUMBER_OF_REPLICAS=0            # each replica builds its own graph under document
                                    # replication, multiplying builds as well as storage
```

Which indexes are GPU-eligible:

| Section | Index | GPU build |
| --- | --- | --- |
| 1 | `recipes-vector-baseline` | yes |
| 2 | `recipes-bm25-baseline` | no vectors |
| 3 | `recipes-hybrid` | yes |
| 4 | `recipes-bm25` / `recipes-vector` | no / yes |

## Tooling

```bash
python gpu/check_cluster.py                          # pre-flight
python download_dataset.py                           # full corpus
cd demos/01_baseline_vector_recipes && python ingest.py --full
python gpu/compare_recall.py --index recipes-vector-baseline
```

**`gpu/check_cluster.py`** confirms `knn.remote_index_build.enabled`, that the
vector repository is registered and readable, and that the service endpoint is
set, then checks `.env` for settings that would silently prevent a build. Run
it first: if the cluster side is not configured, every index setting is inert.

**`gpu/compare_recall.py`** clones a GPU-built index into a control index with
`remote_index_build.enabled: false`, so the control builds its graph locally on
CPU from identical documents, then compares top-k results. It compares scores
as well as ids, because a lower top score on the GPU side means that index
never reached a neighbor the CPU build found -- something id overlap alone can
hide.

Each vector ingest also ends by force-merging to one segment, restoring
`refresh_interval` and refreshing, asserting a document reads back with its
vector, and diffing the k-NN remote build counters. The refresh matters: skip
it and the newly built segment stays non-searchable while queries hit the
pre-merge segments, so you measure a graph you did not just build.
