# Section 4: Hybrid RAG (Ground + Refine)

This project solves a gap that Section 3's native hybrid pipeline can't:
**guaranteed hard exclusions**, e.g. "never recommend a recipe with
sulfites." Vector similarity and even Section 3's weighted BM25+vector
hybrid are still *scoring* mechanisms -- they can rank a sulfite-containing
recipe lower, but nothing in a similarity score can *guarantee* a negation
like "no sulfites" is honored, because "no sulfites" and "contains
sulfites" can still embed close together, and BM25 keyword matching has no
built-in concept of exclusion either. This section fixes that with
explicit, generalized `--exclude`/`--require` structured filters applied
as real OpenSearch `must_not`/`filter` clauses -- which are boolean, not
scored, so they can't be outvoted by relevance.

The main scripts are:

- `ingest.py`: writes each recipe into two indexes -- BM25 (`recipes-bm25`) and vector (`recipes-vector`) -- tagged with structured metadata (cautions, diet_labels, cuisine_type, meal_type, dish_type).
- `query.py`: runs a two-phase ground-then-refine retrieval and generates an answer via Bedrock.

## Installation Prerequisite Software

Please see the README.md at the root of the repo.

## How it works -- externally orchestrated two-phase retrieval

Unlike Section 3's single native `hybrid` query + search pipeline, this
section runs two separate queries across two separate indexes, glued
together by client-side (Python) logic:

1. **GROUND** -- `bm25_search()` against `recipes-bm25`: a keyword + NER-entity query (same style as Section 2) that also applies any `--exclude`/`--require` filters as hard `must_not`/`filter` clauses. This produces a candidate set that is *provably* free of anything excluded -- not just down-ranked.
2. **REFINE** -- `recipe_knn_search()` against `recipes-vector`, restricted via `include_recipe_ids` to exactly the grounding set's candidate IDs (plus the same filters again, for defense-in-depth). This re-ranks the already-safe candidates by semantic similarity to the question, so the final top-k is both constraint-compliant *and* relevant.

The generalized field aliases (`allergen`→`cautions`, `diet`→`diet_labels`,
`cuisine`→`cuisine_type`, etc., see `FILTERABLE_FIELDS` in
`common/opensearch_client.py`) mean any of these can be excluded or
required from the CLI -- not just one hardcoded flag.

## Dataset

Same 39,445-recipe corpus as Section 1/2/3. Of those, **31,949 carry a
"Sulfites" caution (81%)** -- a large enough majority that a plain,
unfiltered "fish and chips" query legitimately surfaces a
sulfite-containing recipe as the top result, making it a genuine (not
contrived) test of the exclusion guarantee below.

## Step 1: Start the NER Service

**IN A NEW TERMINAL:** this section reuses the same NER service as Section 2:

```bash
cd demos/02_baseline_bm25_ner_recipes
python ner_service.py
```

## Step 2: Ingest the Recipe Dataset

Back in your original terminal:

```bash
cd demos/04_hybrid_ground_refine
python ingest.py --full
```

The slowest section, roughly 17 minutes: one NER call per recipe *and* an
embedding for every recipe, across two indexes. They are loaded one after the
other rather than interleaved, so the vector index is not entangled with NER
latency on the BM25 side. Add `--recreate` if the indexes are already loaded.

## Step 3: Query the RAG Pipeline

```bash
# No hard filter -- both "Fish and Chips" recipes are eligible
python query.py --question "Suggest a British fish and chips recipe."

# Hard-exclude anything with sulfites
python query.py --question "Suggest a British fish and chips recipe." --exclude allergen=Sulfites

# --require works the same way, e.g.:
python query.py --question "Suggest a dinner idea." --require cuisine=italian --require diet=Low-Carb
```

## Step 4: Validate

With `ner_service.py` still running:

```bash
python validate.py
```

`validate.py` asserts the exclusion guarantee directly: every recipe in
both the grounding set and the refined set must NOT carry the excluded
caution, for every recipe in the corpus that does carry it.

## Takeaways

**Without a filter**, the sulfite-containing recipe legitimately wins both
stages -- this is the *correct*, unforced behavior when nothing says to
avoid it:

```
GROUNDING (BM25, top 2 of 25):
  - Fish and Chips | source=thegratefulgirlcooks.com | cautions=[]          | score=22.1129
  - Fish and chips | source=womensweeklyfood.com.au  | cautions=['Sulfites'] | score=22.1129
REFINED (vector rerank, top 2 of 5):
  - Fish and chips | source=womensweeklyfood.com.au  | cautions=['Sulfites'] | score=0.7941
  - Fish and Chips | source=thegratefulgirlcooks.com | cautions=[]          | score=0.7691
ANSWER: presents BOTH recipes, sulfites and all, as valid options.
```

**With `--exclude allergen=Sulfites`**, the sulfite-containing recipe is
gone from *both* the 25-candidate grounding set and the final refined
results -- not just pushed down:

```
GROUNDING (BM25, top result of 25):
  - Fish and Chips | source=thegratefulgirlcooks.com | cautions=[] | score=22.1129
REFINED (vector rerank, top result of 4):
  - Fish and Chips | source=thegratefulgirlcooks.com | cautions=[] | score=0.7691
ANSWER: recommends ONLY thegratefulgirlcooks.com's recipe, correctly noting it has no allergen cautions.
```

This is the practical difference between "hybrid scoring" (Section 3,
where sulfite-free wasn't even an option the system could be told to
enforce) and "hybrid scoring + hard constraints" (Section 4): the
exclusion isn't a preference the ranking might honor, it's a guarantee.
