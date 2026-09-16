# Section 3: Hybrid RAG (Native OpenSearch Search Pipeline)

This project fuses BM25 keyword matching and vector semantic search
**server-side, in a single OpenSearch query**, instead of running two
separate retrieval systems and reconciling them yourself the way Section 2
did. Section 1 showed vector search can silently drop the correct document
when two recipes share a name. Section 2 fixed that with exact
keyword/entity matching, but needed a whole extra NER microservice running
alongside it. This section shows you don't have to choose, and don't need
the extra service.

The main scripts are:

- `ingest.py`: embeds each recipe, writes it into a single hybrid index (text + embedding), and creates the search pipeline.
- `query.py`: runs one native `hybrid` query (BM25 `match` + kNN) and generates an answer via Bedrock.

## Installation Prerequisite Software

Please see the README.md at the root of the repo.

## How it works

Unlike Section 1/2, there is only one index here (`recipes-hybrid`), and
each document has both a `text` field (for BM25) and an `embedding` field
(for kNN) -- see `ensure_hybrid_index()` in `common/opensearch_client.py`.

Querying uses OpenSearch's `hybrid` query clause with two sub-queries:

```json
{
  "query": {
    "hybrid": {
      "queries": [
        { "match": { "text": "<question>" } },
        { "knn": { "embedding": { "vector": [...], "k": 10 } } }
      ]
    }
  }
}
```

A search pipeline (`hybrid-search-pipeline`, created once by `ingest.py`
via `ensure_hybrid_pipeline()`) post-processes the two result sets before
they're merged:

1. `normalization-processor` rescales each sub-query's raw scores onto a comparable `[0, 1]` range via min-max normalization -- BM25 scores and cosine/L2 kNN scores otherwise live on completely different scales and can't be added together meaningfully.
2. `arithmetic_mean` combination then takes a weighted average of the two normalized scores: BM25 weight 0.6, vector weight 0.4 by default.

### Why the BM25 weight is 0.6, not 0.3

The weights are a ceiling, not a preference. Min-max normalization puts each
sub-query's best hit at 1.0, so a document that only the BM25 side found
scores at most `bm25_weight`, and a document only the vector side found scores
at most `vector_weight`.

That matters here because the recipe each question names has *no* vector
signal at all -- ask about `thegratefulgirlcooks.com`'s Fish and Chips on the
full corpus and it is nowhere in the vector top 50, while BM25 ranks it 2nd.
With the weights at 0.3/0.7 its ceiling was 0.3, and thousands of merely
fish-adjacent recipes cleared 0.3 on similarity alone. It landed 5th of 5,
behind a chicken recipe and a sandwich. At 0.6/0.4 it lands 2nd.

This is a scale effect, not a bug: on the 260-recipe sample 0.3 worked fine,
because there was nothing else close enough to outrank it. Tune fusion weights
against the corpus you will actually serve.

`HYBRID_CANDIDATE_K` (default 50) is the second half of the same problem. Each
sub-query is searched that deep before the lists are fused, and the fused list
is then cut to `RAG_TOP_K`. Leave the two equal and a document BM25 ranks 6th
is never handed to the combiner in the first place.

The same disambiguation trick Section 2 relied on (folding the recipe's
`source` domain directly into the indexed `text`) is what gives BM25 a
literal term to lock onto here too -- but there's no NER service
extracting it into a separate field, since the raw text match is doing the
work.

Weights are tunable from the CLI (`--bm25-weight` / `--vector-weight` in
`ingest.py`, which is what (re)creates the pipeline) if you want to see how
leaning further toward BM25 or vector shifts the ranking, or from `.env` via
`HYBRID_BM25_WEIGHT` / `HYBRID_VECTOR_WEIGHT`. `query.py` deliberately does
not rewrite an existing pipeline, so whatever you ingested with is what you
query with. Retrieval depth is `--candidate-k` / `HYBRID_CANDIDATE_K`.

## Dataset

Same 39,445-recipe corpus as Section 1/2 -- including the same real, verified
ambiguous pair: two "Fish and Chips" recipes (`thegratefulgirlcooks.com`,
no cautions vs. `womensweeklyfood.com.au`, Sulfites) where Section 1's
vector search demonstrably drops the first recipe from its top-5 results
entirely.

## Step 1: Ingest the Recipe Dataset

```bash
cd demos/03_hybrid_native_opensearch
python ingest.py --full
```

No NER service needs to be running for this stage -- that's the point.

About 6 minutes. Add `--recreate` if the index is already loaded.

## Step 2: Query the RAG Pipeline

```bash
python query.py --question "Does the thegratefulgirlcooks.com Fish and Chips recipe contain sulfites?"
python query.py --question "Does the womensweeklyfood.com.au Fish and Chips recipe contain sulfites?"
```

## Step 3: Validate

```bash
python validate.py
```

## Takeaways

Both questions retrieve their own named recipe in the top 2 hybrid hits
(score = normalized/weighted combination, not raw BM25 or cosine). On the
260-recipe sample:

```
QUESTION: Does the thegratefulgirlcooks.com Fish and Chips recipe contain sulfites?
  - Fish and chips | source=womensweeklyfood.com.au | score=0.6498
  - Fish and Chips | source=thegratefulgirlcooks.com | score=0.6254
  ...
ANSWER: ...does not contain sulfites. The allergen cautions for this recipe list "none listed."

QUESTION: Does the womensweeklyfood.com.au Fish and Chips recipe contain sulfites?
  - Fish and chips | source=womensweeklyfood.com.au | score=1.0000
  ...
ANSWER: Yes, ...does contain sulfites. The recipe lists "Sulfites" under its allergen cautions.
```

And on the full 39,445-recipe corpus, where the competition is far harsher:

```
QUESTION: Does the thegratefulgirlcooks.com Fish and Chips recipe contain sulfites?
  - Grilled Curry Chicken Legs | source=Penzeys                  | score=0.6000
  - Fish and Chips             | source=thegratefulgirlcooks.com | score=0.5900
  ...

QUESTION: Does the womensweeklyfood.com.au Fish and Chips recipe contain sulfites?
  - Fish and chips | source=womensweeklyfood.com.au | score=0.7238
  ...
```

Compare this to Section 1, where vector search dropped
`thegratefulgirlcooks.com`'s recipe from the top-5 entirely. Here it comes
back purely because BM25's exact-text match on the source domain
contributes a large enough combined score -- no separate NER service needed
to force it there the way Section 2 did.

Note what the full-corpus run shows about *how* it comes back. That recipe
is not in the vector top 50 at all; every point of its score comes from the
keyword side. That is the honest version of the hybrid story: fusion is not
"semantic search, slightly improved," it is two independent retrievers where
either one can carry a document the other misses entirely.
