# Cassandra Hybrid Retrieval — SAI Keywords + Vector ANN

A local **Apache Cassandra 5** demo that answers one question three different
ways against a single table: SAI keyword matching, **JVector ANN**, and native
CQL that filters and vector-orders in one statement. It is retrieval only — no
language model — so you can judge the context a RAG pipeline would send before
generation hides the evidence.

This is the hybrid follow-up to [cassandra-vector-demo](../cassandra-vector-demo/README.md), which shows the approximate nearest neighbor (ANN) lane alone on the same dataset.

## What it demonstrates

For this demo, imagine that you own a fictional **2024 Summit 1500**. Ask a support question and compare the different methods of retrieval:

- **Embeddings cannot resolve identifiers.** *"What is recall 24V-330?"* puts the
matching recall at rank 9 of 12 in the ANN lane. `keywords CONTAINS '24v-330'`
puts it first.
- **Filters fix it.** SAI predicates plus ANN ordering in one CQL statement
return a single row, the right one. `keywords CONTAINS '24v-330'` excludes the
eleven wrong recalls before ANN ranks anything, so the identifier gates the
results instead of competing with semantic similarity.
- **Vectors are still good at meaning.** *"Is there a recall on the tailgate?"*
ranks the tailgate-latch recall first at 0.7767, ahead of eleven near-twins.

The twelve recall documents are written from one template, varying only in the
recall number, the affected component and its symptom, the remedy, and the model
and year. That is deliberate: it leaves the identifier as the only signal a
lookup-by-number can rely on.

```
Question
   │
   ├── extract exact tokens ──▶ keywords CONTAINS ?          → lexical hits
   │                            (ranked by overlap in app)
   │
   ├── embed with all-MiniLM-L6-v2 (384-dim, local)
   │      └──────────────────▶ ORDER BY embedding ANN OF ?   → neighbors
   │
   └── selective token + category + customer model/year
          └───────────────────▶ WHERE ... ORDER BY ANN OF ?  → filtered neighbors
```

## Prerequisites

- **Docker / Docker Compose**
- **Python 3.10+** (Python 3.14 may fail installing the embedding stack). The
`python3` shipped with macOS command line tools is 3.9 and is too old —
`truststore` has no 3.9 wheel, so `pip install` fails with
`No matching distribution found`. Name a newer interpreter explicitly, e.g.
`python3.13 -m venv .venv`.
- About **4GB** RAM and **2GB** free disk for Cassandra, Python packages, and the
first-run model download
- Internet access the first time `seed.py` downloads
`sentence-transformers/all-MiniLM-L6-v2` (a Hugging Face Hub token reminder is
optional; set `HF_TOKEN` only if you want higher Hub rate limits)

This demo and `cassandra-vector-demo` both bind Cassandra to `127.0.0.1:9042`, so
run one at a time. `docker compose stop` in the other project's directory frees
the port and keeps its data. To run both at once instead, publish a different
host port and point the app at it with `CASSANDRA_PORT`.

## Running

```bash
cd cassandra-hybrid-retrieval-demo
docker compose up -d
docker compose ps          # wait until cassandra is healthy (1–2 min on first boot)

python3.13 -m venv .venv    # any 3.10-3.13 interpreter works
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt

python src/setup_schema.py
python src/seed.py
python src/demo.py
```

If `seed.py` fails with `CERTIFICATE_VERIFY_FAILED` on a TLS-inspecting corporate
network, reinstall from `requirements.txt` so `truststore` can use the operating
system certificate store.

`python src/demo.py` with no arguments runs all three scripted questions. Pass
`--model` and `--model-year` to apply customer context to the hybrid lane.

Presenter UI at [http://127.0.0.1:8000](http://127.0.0.1:8000):

```bash
uvicorn presenter:app --app-dir src --host 127.0.0.1 --port 8000
```

Stop / reset:

```bash
docker compose down          # keep data volume
docker compose down -v       # wipe Cassandra data
```

## Try these questions

```bash
python src/demo.py "What is recall 24V-330?"
python src/demo.py "Is there a recall on the tailgate?"
python src/demo.py "How much can the Summit 1500 tow?"
python src/demo.py --model 1500 --model-year 2024 "How much can the Summit 1500 tow?"
```

**"What is recall 24V-330?"** — the identifier case, and the reason this sample
exists:


| Lane          | Top hit                                | Where the correct document landed |
| ------------- | -------------------------------------- | --------------------------------- |
| Keyword (SAI) | `recall-24v-330`, matched 2 tokens     | rank 1                            |
| Vector ANN    | `recall-23v-011` at 0.7871             | rank 9 of 12, not in the top 5    |
| Hybrid        | `recall-24v-330` at 0.7332, single row | rank 1                            |


One detail worth pointing at. In the keyword lane, ranks 2 through 5 all matched
exactly one token (`recall`), so that tail is not a ranking at all — SAI returns
matches, and the overlap count is the app's own ordering.

**"Is there a recall on the tailgate?"** — do not oversell the failure. Vector ANN
gets this right at 0.7767 against 0.7032 for the runner-up, and every lane agrees.
Descriptive language is what embeddings are for.

**"How much can the Summit 1500 tow?"** — four confident chunks with four
different numbers, all within 0.08: the 2024 1500 (11,300 lb), the 2023 1500
(9,100 lb), the 2024 **2500** (17,600 lb), and 2024 1500 **payload** (1,750 lb).
The question never said which model year, and no amount of ranking can infer it.
The keyword lane is no better — it puts the 2023 truck first, on a three-token
tie the app breaks by first-seen order rather than by relevance. Adding
`--model 1500 --model-year 2024`, metadata you
already have in a customer record, reduces the hybrid lane to one row:
`summit-1500-2024-tow` at 0.8990.

Scores in the ANN lanes are Cassandra's `similarity_cosine`, which rescales
cosine into `[0, 1]` as `(1 + cos θ) / 2`, so unrelated text lands near 0.5
rather than 0 and only roughly 0.7 and above is meaningful.

## Measure it

```bash
python src/evaluate.py
```

Asks `What is recall X?` for all twelve recall IDs at `LIMIT 10` and reports where
each lane ranked the matching document:

```text
keyword        recall@1 = 12/12 (100%)   recall@3 = 12/12 (100%)   mean rank = 1.00
vector         recall@1 = 2/12 (17%)     recall@3 = 8/12 (67%)     mean rank = 3.58
hybrid         recall@1 = 12/12 (100%)   recall@3 = 12/12 (100%)   mean rank = 1.00
```

ANN alone puts the right recall first twice out of twelve. Grounding on the
identifier gets it every time, whether through SAI matching alone or through SAI
filters with ANN ordering. That is the measurement to run before shipping vector
search as your only retrieval path.

## Project structure

```
cassandra-hybrid-retrieval-demo/
├── src/
│   ├── common.py          # Cassandra connect, schema, embeddings
│   ├── setup_schema.py    # Keyspace + SAI keyword, metadata, and ANN indexes
│   ├── seed.py            # Embed + insert corpus
│   ├── retrieval.py       # Keyword, ANN, filtered ANN
│   ├── demo.py            # CLI
│   ├── evaluate.py        # recall@1 per lane over every recall ID
│   └── presenter.py       # FastAPI presenter
├── static/
│   └── presenter.html     # Three-column UI + CQL statement panel
├── data/
│   └── corpus.json        # Fictional Summit truck knowledge base
├── tests/
│   └── test_app.py
├── docker-compose.yml     # Cassandra 5.0.9 on localhost:9042
├── requirements.txt
└── README.md
```

Run the tests with `python -m unittest discover -s tests`. They cover token extraction, category inference, the filtered-ANN statement, and dataset invariants, and need no running Cassandra.

## What Cassandra is not

- **SAI is not BM25.** Apache Cassandra's SAI text options are limited to
`normalize`, `case_sensitive`, and `ascii`. Lucene analyzers via
`index_analyzer`, the `:` term-match operator, and `ORDER BY BM25` are not part
of open source Apache Cassandra. Treat this lane as exact grounding, not a
drop-in replacement for a search cluster.
- **Not a graph engine.** Apache Cassandra has no native GraphRAG; citation or
related-id walks would be application code.
- **Not a warehouse.** This is top-k retrieval for a prompt, not analytics over a dataset.
- You still add a search sidecar when you need real BM25 relevance, facets, or
highlighting. The point is not to delete your search cluster; it is to avoid
standing one up just to match identifiers next to operational data.

## Additional materials

- [cassandra-vector-demo](../cassandra-vector-demo/README.md) — the ANN-only precursor sample
- [Storage-Attached Indexing (SAI)](https://cassandra.apache.org/doc/latest/cassandra/developing/cql/indexing/sai/sai-concepts.html) — how Cassandra attaches keyword and vector search to one table
- Prefer Cassandra **5.0.7+** (this sample uses **5.0.9**) for filtered-ANN correctness and latency fixes — see [CASSANDRA-20086](https://issues.apache.org/jira/browse/CASSANDRA-20086)

