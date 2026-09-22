"""Prompt Evaluation: "How do you evaluate what 'good' is?"

Grounding context is now retrieved LIVE from a real OpenSearch index (run
`ingest.py` once beforehand) via a kNN vector search -- a standard
embed-then-search pattern (`EmbeddingModel` + `knn_search`). This is the part
of the session that actually exercises OpenSearch, closing the loop between
"here's what was retrieved" and "here's whether the answer is trustworthy."
If OpenSearch isn't configured or isn't reachable, this falls back to a
small built-in copy of the same chunks (`FALLBACK_CHUNKS`) so a
network/credentials hiccup on stage degrades gracefully instead of crashing
the demo.

The evaluation harness itself stays LIGHTWEIGHT and dependency-free: aside
from the OpenSearch query, it only talks to the same local OpenAI-compatible
endpoint the other demos use.

For production RAG systems, reach for a real evaluation framework instead of
hand-rolling this:

  - RAGAS         -- Faithfulness, Answer Relevancy, Context Precision/Recall
                     https://github.com/explodinggradients/ragas
  - DeepEval       -- G-Eval, Faithfulness/Hallucination metrics, pytest-style
                     assertions for CI            https://github.com/confident-ai/deepeval
  - TruLens        -- "Groundedness" & "Answer Relevance" feedback functions,
                     works as an app-level tracing/eval layer
                     https://github.com/truera/trulens
  - Arize Phoenix  -- Open-source LLM observability + eval traces
                     https://github.com/Arize-ai/phoenix
  - Giskard        -- Scan RAG pipelines for hallucination/bias/robustness
                     https://github.com/Giskard-AI/giskard
  - promptfoo      -- Config-driven prompt/RAG regression testing in CI
                     https://github.com/promptfoo/promptfoo

Those tools compute the same two families of signal this script demos by
hand: (1) is the answer *faithful/grounded* in the retrieved context (an
LLM-as-judge check), and (2) does the answer *cite* content that's actually
traceable back to a source chunk (an attribution/overlap check).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

from openai import OpenAI

from common.config import load_settings
from common.embeddings import EmbeddingModel, to_list
from common.logging import get_logger
from common.opensearch_client import client_from_settings, knn_search

LOGGER = get_logger(__name__)

# Connecting to the local inference engine (e.g., llama.cpp / MLX server
# started by ../1_prompt_repetition/slm_service.py). We reuse the
# SAME local model as judge -- no extra model download or API key needed for
# the live demo.
client = OpenAI(base_url="http://127.0.0.1:8001/v1", api_key="not-needed")

# Live-demo safety net: if OPENSEARCH_HOST isn't configured, or the cluster
# is unreachable when this runs, we fall back to this hardcoded copy of the
# exact same chunks `ingest.py` indexes -- so a network hiccup on stage
# degrades gracefully instead of crashing the demo.
FALLBACK_CHUNKS = [
    {
        "id": 1,
        "origin": "Apple Inc. 10-Q filing - Q3 FY2025",
        "text": (
            "Total revenue for Q3 FY2025 was $94.9 billion. Gross margin "
            "for the quarter was 46.9 percent."
        ),
    },
    {
        "id": 2,
        "origin": "Apple Inc. 10-Q filing - Q4 FY2025",
        "text": (
            "Total revenue for Q4 FY2025 was $102.5 billion. Gross margin "
            "for the quarter was 47.3 percent."
        ),
    },
]

QUESTION = "What was Apple's gross margin for the quarter with $94.9 billion in revenue?"


def retrieve_context_chunks(question: str) -> List[Dict[str, Any]]:
    """Retrieve grounding chunks via a live OpenSearch kNN vector search.

    Embeds `question` with the same model `ingest.py` used to embed the
    chunks, then runs a kNN search -- mirroring
    `01_baseline_vector_rag/query.py`'s `ask()`. Falls back to
    `FALLBACK_CHUNKS` (identical data to what `ingest.py` indexes) if
    OpenSearch isn't configured or the query fails for any reason -- this
    evaluation demo should never hard-fail on a network or credentials
    problem mid-talk.
    """

    settings = load_settings()
    search_client = None
    try:
        search_client = client_from_settings(settings)
    except Exception as exc:  # pragma: no cover - defensive, live-demo path
        LOGGER.warning("Could not create OpenSearch client (%s); using fallback context.", exc)

    if search_client is None:
        print("\n[OpenSearch not configured -- using built-in fallback context]")
        return FALLBACK_CHUNKS

    try:
        embedder = EmbeddingModel(settings)
        query_vec = to_list(embedder.encode([question])[0])
        hits = knn_search(search_client, settings.opensearch_index, query_vec, k=2)
        if not hits:
            print("\n[OpenSearch returned no hits -- using built-in fallback context]")
            return FALLBACK_CHUNKS
        print(f"\n[Retrieved {len(hits)} chunk(s) live from OpenSearch index '{settings.opensearch_index}' via kNN search]")
        return hits
    except Exception as exc:  # pragma: no cover - defensive, live-demo path
        LOGGER.warning("OpenSearch query failed (%s); using fallback context.", exc)
        print("\n[OpenSearch query failed -- using built-in fallback context]")
        return FALLBACK_CHUNKS

# Two candidate answers to evaluate side by side:
#   - "good": grounded, matches the Q3 chunk exactly.
#   - "bad":  plausible-sounding but pulls the WRONG quarter's number AND
#             adds an unsupported claim -- a stand-in for what an ungrounded
#             or badly-structured pipeline (see Demo 2 / Demo 3) might return.
CANDIDATE_ANSWERS: Dict[str, str] = {
    "good": "Apple's gross margin for the $94.9 billion quarter (Q3 FY2025) was 46.9 percent.",
    "bad": (
        "Apple's gross margin for that quarter was 47.3 percent, reflecting "
        "strong iPhone demand in international markets."
    ),
}


def build_bounded_context(context_chunks: List[Dict[str, Any]]) -> str:
    """Render the context chunks with the same delimiter scheme as Demo 2."""

    return "\n".join(
        f'<chunk id="{chunk["id"]}" source="{chunk["origin"]}">\n{chunk["text"]}\n</chunk>'
        for chunk in context_chunks
    )


def llm_judge_faithfulness(question: str, context: str, answer: str) -> str:
    """Ask the local model to act as a faithfulness/groundedness judge.

    This mirrors what RAGAS calls "Faithfulness" and TruLens calls
    "Groundedness": every claim in ANSWER must be traceable to CONTEXT, not
    to the model's own outside knowledge.
    """

    judge_messages = [
        {
            "role": "system",
            "content": (
                "You are a strict evaluation judge for a RAG pipeline. Given "
                "CONTEXT and ANSWER, decide whether every factual claim in "
                "ANSWER is directly supported by CONTEXT. Respond in exactly "
                "this format:\nVERDICT: SUPPORTED or UNSUPPORTED\nREASON: "
                "<one short sentence>"
            ),
        },
        {
            "role": "user",
            "content": (
                f"CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\n"
                f"ANSWER TO EVALUATE:\n{answer}"
            ),
        },
    ]

    response = client.chat.completions.create(
        model="local-llm",
        messages=judge_messages,
        user="opensearchcon-demo-eval",
        temperature=0.0,
        max_tokens=64,
    )
    return (response.choices[0].message.content or "").strip()


_WORD_RE = re.compile(r"[A-Za-z0-9.%]+")


def citation_overlap_score(answer: str, context_chunks: List[Dict[str, Any]]) -> float:
    """A simple, dependency-free attribution heuristic.

    Extracts numeric/word tokens from ANSWER and reports what fraction of
    them literally appear in at least one source chunk's text. This is a
    crude stand-in for what RAGAS calls "Context Precision" -- it will not
    catch paraphrased hallucinations, but it catches the common case in a
    live demo: a number that simply isn't anywhere in the retrieved context.
    """

    answer_tokens = {token.lower() for token in _WORD_RE.findall(answer) if len(token) > 2}
    if not answer_tokens:
        return 0.0

    source_text = " ".join(chunk["text"] for chunk in context_chunks).lower()
    matched = [token for token in answer_tokens if token in source_text]
    return len(matched) / len(answer_tokens)


def main() -> None:
    context_chunks = retrieve_context_chunks(QUESTION)
    context = build_bounded_context(context_chunks)

    print("\nCONTEXT (bounded chunks, same scheme as Demo 2):")
    print(context)
    print(f"\nQUESTION:\n{QUESTION}")

    for label, answer in CANDIDATE_ANSWERS.items():
        print(f"\n{'=' * 60}\nEVALUATING '{label.upper()}' ANSWER:\n{answer}")

        verdict = llm_judge_faithfulness(QUESTION, context, answer)
        print(f"\n[LLM-as-judge faithfulness check]\n{verdict}")

        overlap = citation_overlap_score(answer, context_chunks)
        print(f"\n[Citation/attribution overlap] {overlap:.0%} of answer tokens traceable to a source chunk")


if __name__ == "__main__":
    main()
