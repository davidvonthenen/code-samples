# Demo 4: Prompt Evaluation — "How Do You Evaluate What 'Good' Is?"

Session section: **Prompt Evaluation.** Once a RAG pipeline produces an
answer, how do you know it's actually *good* — grounded in the retrieved
context, not just plausible-sounding?

Unlike Demos 1–3, this demo's grounding context is retrieved **live from a
real OpenSearch index** rather than a hardcoded Python list — a standard
embed-then-kNN-search pattern (`EmbeddingModel` + `ensure_vector_index` +
`knn_search`). This is the part of the session that actually exercises
OpenSearch, closing the loop between "here's what was retrieved" and
"here's whether the answer is trustworthy."

If OpenSearch isn't configured (no `OPENSEARCH_HOST`) or a query fails for
any reason, `evaluate_answer.py` transparently falls back to a small
built-in copy of the same chunks (`FALLBACK_CHUNKS`) — a network/credentials
hiccup on stage degrades gracefully instead of crashing the demo. The
evaluation harness itself stays lightweight and dependency-free otherwise:
aside from the OpenSearch query + embedding step, it only talks to the same
local OpenAI-compatible endpoint the other demos use.

It checks two things for each candidate answer, against the chunks retrieved
for the question (the same Q3/Q4 Apple scenario introduced in
`../2_good_structure/good_structure.py`):

1. **Faithfulness / groundedness (LLM-as-judge).** The local model itself is
   prompted to act as a strict judge: is every claim in the answer directly
   supported by the retrieved context, or did it introduce outside
   knowledge? This is the same idea RAGAS calls "Faithfulness" and TruLens
   calls "Groundedness."
2. **Citation / attribution overlap (heuristic).** A simple token-overlap
   check: what fraction of the answer's numbers/words are literally traceable
   back to a source chunk? This won't catch a cleverly-paraphrased
   hallucination, but it catches the most common live-demo failure — a
   number that simply isn't anywhere in the retrieved context. Loosely
   analogous to RAGAS's "Context Precision."

Run against a **good** answer (grounded, matches the right quarter) and a
**bad** answer (plausible, but cites the wrong quarter's number plus an
unsupported claim) side by side, so the contrast in verdicts is visible live.

## For production: reach for a real framework instead

This script is a teaching aid, not a recommendation to hand-roll evaluation
in production. At scale, use one of the following instead:

| Tool | What it adds over this demo |
|---|---|
| [RAGAS](https://github.com/explodinggradients/ragas) | Faithfulness, Answer Relevancy, Context Precision/Recall — the industry-standard RAG metric suite |
| [DeepEval](https://github.com/confident-ai/deepeval) | G-Eval + hallucination/faithfulness metrics with pytest-style assertions, built for CI regression testing |
| [TruLens](https://github.com/truera/trulens) | "Groundedness"/"Answer Relevance" feedback functions wired into app-level tracing |
| [Arize Phoenix](https://github.com/Arize-ai/phoenix) | Open-source LLM observability + eval traces across a whole pipeline |
| [Giskard](https://github.com/Giskard-AI/giskard) | Automated scanning for hallucination, bias, and robustness issues in RAG pipelines |
| [promptfoo](https://github.com/promptfoo/promptfoo) | Config-driven prompt/RAG regression testing, CI-friendly |

## Run it

Requires:
- The local model server from Demo 1 to already be running
  (`../1_prompt_repetition/slm_service.py`, listening on
  `http://127.0.0.1:8001`).
- An OpenSearch cluster reachable via `.env` (copy `.env.example` to `.env`
  and fill in `OPENSEARCH_HOST`/`USER`/`PASSWORD`). If you skip this,
  `evaluate_answer.py` still runs fine using its built-in fallback context.

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in your OpenSearch cluster details

python ingest.py           # one-time: embed + index the grounding chunks
python evaluate_answer.py  # embeds the question, retrieves via kNN, evaluates both answers
```

Watch the console output: it prints whether context was retrieved live from
OpenSearch or fell back to the built-in sample, so the audience can see
which path ran.
