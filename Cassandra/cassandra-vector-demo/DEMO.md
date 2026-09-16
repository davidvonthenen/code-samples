# Cassandra vector demo notes

Presenter: http://127.0.0.1:8000

## First-time setup

This app has its own virtual environment. Create it once before the first demo:

```bash
cd cassandra-vector-demo
python3.13 -m venv .venv   # macOS system python3 is 3.9 and is too old
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Before the demo

```bash
cd cassandra-vector-demo
docker compose up -d
docker compose ps
source .venv/bin/activate
python src/seed.py
uvicorn presenter:app --app-dir src --host 127.0.0.1 --port 8000
```

## Story

1. **Meaning works:** “Is there a recall on the tailgate?” retrieves the
   tailgate-latch recall first at 0.7767, ahead of eleven other recalls.
2. **Identifiers are weak:** “What is recall 24V-330?” does not return the
   matching recall in the top five at all — it lands at rank 9 of 12. Across all
   twelve recall IDs, recall@1 is 2/12 (`python src/evaluate.py`).
3. **Missing metadata stays ambiguous:** “How much can the Summit 1500 tow?”
   returns plausible but conflicting chunks across model years, models, and
   payload versus towing.

The app intentionally shows only ANN neighbors and similarity scores. Do not imply
that it calls an LLM, performs keyword matching, or filters by model metadata.

Two things to state accurately if asked:

- The scores are `similarity_cosine`, which is `(1 + cos θ) / 2`, not raw cosine.
  0.5 means orthogonal. Do not read 0.6 as "moderately relevant."
- Embeddings are not blind to identifiers. `What is recall 23V-011?` ranks first.
  The signal is weak, not absent, and near-identical documents outvote it.

CLI fallback:

```bash
python src/demo.py
python src/demo.py "What is recall 24V-330?"
```
