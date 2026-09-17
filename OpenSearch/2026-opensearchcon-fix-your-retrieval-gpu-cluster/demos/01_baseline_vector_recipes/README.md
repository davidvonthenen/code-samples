# Section 1: Baseline Vector RAG

This project stores a curated recipe dataset as vectors in OpenSearch and
answers dietary-safety questions using pure vector k-NN retrieval plus
Amazon Bedrock for generation. It demonstrates the failure this whole
session is about: vector search retrieves by *semantic similarity*, not by
*factual correctness*.

The main scripts are:

- `ingest.py`: embeds each recipe and writes it, plus its metadata, to OpenSearch.
- `query.py`: embeds the question, retrieves top-k vector hits, builds a grounded prompt, and prints the answer plus retrieved hit metadata.

## Installation Prerequisite Software

Please see the README.md at the root of the repo.

## Dataset

All 39,445 recipes from
[`datahiveai/recipes-with-nutrition`](https://huggingface.co/datasets/datahiveai/recipes-with-nutrition)
on HuggingFace. Run `python download_dataset.py` from the repo root once
before ingesting.

The corpus contains a real, naturally-occurring ambiguous pair: two
completely different recipes that both happen to be named "Fish and Chips":

- Recipe A: `thegratefulgirlcooks.com` -- no allergen cautions listed
- Recipe B: `womensweeklyfood.com.au` -- **Sulfites** (from the beer in its batter)

This pair was chosen after empirically verifying it actually breaks vector
retrieval, not just because the names match: with the `all-MiniLM-L6-v2`
embedding model, the question about Recipe A scores *higher* against
Recipe B's embedding (0.62) than against its own (0.49) -- a genuine
ranking flip, not just a close call.

## Step 1: Ingest the Recipe Dataset

```bash
cd demos/01_baseline_vector_recipes
python ingest.py --full
```

Embeds and indexes all 39,445 recipes, about 6 minutes. Add `--recreate` if
the index is already loaded.

## Step 2: Query the RAG Pipeline

```bash
python query.py --question "Does the thegratefulgirlcooks.com Fish and Chips recipe contain sulfites?"
python query.py --question "Does the womensweeklyfood.com.au Fish and Chips recipe contain sulfites?"
```

## Step 3: Validate

```bash
python validate.py
```

This passes when the thegratefulgirlcooks.com recipe is *absent* from the
results. That gap is the failure the rest of the session fixes, so a failure
here means this section stopped demonstrating anything.

## Takeaways

Asking about **thegratefulgirlcooks.com**'s recipe (the correct answer is
"no sulfites"): vector search's top-5 hits don't include that recipe at
all. You get a Mediterranean fish stew, BBQ potato chips, a sandwich, an
air-fryer fish and chips from a different source, and salmon fishcakes --
all semantically plausible, none of them the document that answers the
question. Because
this stage's system prompt instructs the model to say "I don't know" when
the context doesn't support an answer, you'll see the assistant correctly
decline rather than hallucinate -- but the *retrieval* still silently
failed to find the one document that actually answers the question. With a
looser system prompt (or a model that ignores it), this is exactly the
shape of failure that produces a confidently wrong safety answer instead.

Asking about **womensweeklyfood.com.au**'s recipe works fine, since that
recipe legitimately ranks near the top of the corpus regardless.

A dropped or wrong retrieval here isn't just an inaccurate trivia fact --
it's the kind of gap that matters for someone with a sulfite sensitivity.
Section 2's BM25 + entity retrieval is designed to close this exact gap.
