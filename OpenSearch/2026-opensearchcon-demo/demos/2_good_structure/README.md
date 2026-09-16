# Demo 2: Deconstructing RAG Prompts (a.k.a. "Reconstructing RAG Prompts")

Session section: **Deconstructing RAG Prompts — Research-Based Prompt
Structure for RAG.** This is the "positive" counterpart to the Adversarial
Prompts / Contextual Contradiction demo (`../3_contradiction/`): before
showing what breaks when a RAG prompt is built badly, this demo shows what a
correctly-structured prompt looks like, and isolates *structure* (not
retrieval quality) as the variable.

Two of the "Common Mistakes" from the talk are demonstrated here in
combination:

1. **Cross-Chunk Bleeding** — bounded data/info without delimiters. When
   retrieved chunks are concatenated with nothing marking where one ends and
   the next begins, an auto-regressive LLM can blend facts across chunk
   boundaries (e.g. attaching the wrong quarter's gross margin to the wrong
   quarter's revenue figure).
2. **Causal Misalignment** (a.k.a. "premature interrogation") — asking the
   question *before* presenting the context. A non-reasoning LLM reads
   left-to-right; if the question comes first, the model starts attending to
   "what's being asked" before the facts it needs are even in its context
   window. Context should always come first, question last.

Both scripts use the **same already-grounded** four OpenSearch hits, in the
**same order** (a reranker/score cutoff is assumed to have already run —
that lesson is `../3_contradiction/`'s job), so the only thing that changes
between `bad_structure.py` and `good_structure.py` is how the prompt is
built.

The four hits simulate a naive chunker that split a financial table
row-by-row (revenue, revenue, margin, margin) instead of quarter-by-quarter,
and the two margin hits came back in **reverse** quarter order. Neither
sentence names its own quarter — that's only encoded in each hit's
`source` label. Without delimiters, Qwen 2.5 7B reliably pairs the facts
**positionally** (1st revenue with 1st margin, like zipping two parallel
lists) and confidently reports Q4's margin (47.3%) for the Q3 question —
this was verified live against the running model, not theoretical. With the
`<chunk id="..." source="...">` labels present, the same model correctly
uses the `source` attribution instead of guessing from list position.

| | `bad_structure.py` | `good_structure.py` |
|---|---|---|
| Chunk boundaries | None — raw text concatenated | Explicit `<chunk id="..." source="...">` delimiter tags (human-readable, not JSON — see below) |
| Question position | First, before any context | Last, after all context |
| Expected answer | 46.9% (reliably reports 47.3% in practice) | 46.9% |

## Why delimiters, and why not JSON?

The talk's "JSON or Raw Payload Dump" point applies here too: LLMs are
**language** models, and heavy JSON/structured-payload dumps introduce
token bloat that interferes with chain-of-thought and correlates with
5–20% higher failure rates in the cited research. The fix isn't "more
structure," it's "human-readable structure" — lightweight, XML-style
delimiter tags around each chunk, not a nested JSON blob.

Reference: ["Let Me Speak Freely? A Study on the Impact of Format
Restrictions on Performance of Large Language
Models"](https://arxiv.org/abs/2408.02442) (`../../docs/2408.02442v1.pdf`).

## Run it

Requires the local model server from Demo 1 to already be running
(`../1_prompt_repetition/slm_service.py`, listening on
`http://127.0.0.1:8001`).

```bash
pip install -r requirements.txt
python bad_structure.py
python good_structure.py
```

Watch the model's answer flip from a blended/wrong margin (bad structure) to
the correct 46.9% (good structure) with identical underlying data.
