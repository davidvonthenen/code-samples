# Section 2: Baseline BM25 + NER RAG

This project is the "obvious" fix for Section 1's failure: add keyword and
entity matching so retrieval isn't relying on semantic similarity alone.
BM25 + entity tagging is deterministic and auditable -- the same query
keyword either matches a document's text or it doesn't.

The main scripts are:

- `ner_service.py`: a local spaCy NER service, called by `ingest.py` and `query.py`.
- `ingest.py`: tags each recipe with entities from the NER service and writes it to a BM25 index.
- `query.py`: extracts entities from the question, runs a BM25 + entity search, and generates an answer via Bedrock.

## Installation Prerequisite Software

Please see the README.md at the root of the repo.

## Dataset

Same 39,445-recipe corpus as Section 1 -- including the same real, verified
ambiguous pair: two "Fish and Chips" recipes (thegratefulgirlcooks.com, no
cautions vs. womensweeklyfood.com.au, Sulfites) that Section 1's vector
search demonstrably confuses.

Each recipe is tagged at ingest time with entities from the local NER
service, plus its own `source` and `recipe_name` folded directly into the
same `entities` field. This is a deliberate design choice, confirmed by
testing: spaCy's small NER model tags no useful entity for
"thegratefulgirlcooks.com" (it only spuriously tags the unrelated word
"chips" as `ORG`), so relying on NER alone would not have disambiguated
these two recipes. The source is already known, deterministic metadata --
there's no reason to leave a disambiguation signal you already have to
chance.

## Step 1: Start the NER Service

**IN A NEW TERMINAL:** run the following from this directory (don't forget
to activate your virtual environment):

```bash
cd demos/02_baseline_bm25_ner_recipes
python ner_service.py
```

## Step 2: Ingest the Recipe Dataset

Back in your original terminal:

```bash
cd demos/02_baseline_bm25_ner_recipes
python ingest.py --full
```

About 12 minutes: this stage makes one NER call per recipe. Add `--recreate`
if the index is already loaded.

## Step 3: Query the RAG Pipeline

```bash
python query.py --question "Does the thegratefulgirlcooks.com Fish and Chips recipe contain sulfites?"
python query.py --question "Does the womensweeklyfood.com.au Fish and Chips recipe contain sulfites?"
```

## Step 4: Validate

With `ner_service.py` still running:

```bash
python validate.py
```

## Takeaways

Compare these answers against Section 1's. BM25 + entity matching
correctly separates the two recipes because it matches the literal term
"thegratefulgirlcooks.com" or "womensweeklyfood.com.au" rather than
relying on "similar meaning" -- closing the exact gap Section 1 exposed.
Verified output: it correctly answers "no sulfites" for
thegratefulgirlcooks.com and "yes, sulfites" for womensweeklyfood.com.au,
where Section 1's vector search couldn't even retrieve the first recipe.

That said, notice what this fix *cost*: a whole separate `ner_service.py`
process has to be running just to disambiguate two documents. Section 3
gets the same result without it.
