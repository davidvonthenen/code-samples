from openai import OpenAI

# Connecting to the local inference engine (e.g., llama.cpp / MLX server started
# by ../1_prompt_repetition/slm_service.py).
client = OpenAI(base_url="http://127.0.0.1:8001/v1", api_key="not-needed")

# Four ALREADY-GROUNDED OpenSearch hits (same simulated-hit universe as the
# Contextual Contradiction demo, ../3_contradiction/). Assume a reranker/score
# cutoff has already run -- all four chunks are genuinely relevant to the
# question. The failure mode demonstrated here is NOT retrieval quality, it
# is prompt STRUCTURE:
#
#   1. Cross-Chunk Bleeding  -- a naive chunker split each quarter's revenue
#      and gross margin into SEPARATE hits (e.g. splitting a financial table
#      row-by-row instead of quarter-by-quarter), and they came back grouped
#      by field (both revenue hits, then both margin hits) rather than by
#      quarter, with the margins in reverse order. With no delimiters or
#      source labels, the model has no way to tell WHICH margin belongs to
#      WHICH revenue -- it tends to pair them positionally (1st revenue with
#      1st margin, like zipping two parallel lists), grabbing Q4's margin for
#      the Q3 question.
#   2. Causal Misalignment   -- aka "premature interrogation": the question is
#      placed BEFORE the model has seen any context. A non-reasoning LLM is
#      auto-regressive/left-to-right, so it starts "thinking" about the
#      answer before the relevant facts have even entered its context.
retrieved_chunks = [
    {
        "origin": "Apple Inc. 10-Q filing - Q3 FY2025 - Revenue",
        "text": "Total revenue for the quarter was $94.9 billion.",
    },
    {
        "origin": "Apple Inc. 10-Q filing - Q4 FY2025 - Revenue",
        "text": "Total revenue for the quarter was $102.5 billion.",
    },
    {
        "origin": "Apple Inc. 10-Q filing - Q4 FY2025 - Gross Margin",
        "text": "Gross margin for the quarter was 47.3 percent.",
    },
    {
        "origin": "Apple Inc. 10-Q filing - Q3 FY2025 - Gross Margin",
        "text": "Gross margin for the quarter was 46.9 percent.",
    },
]

question = "What was Apple's gross margin for the quarter with $94.9 billion in revenue?"

# MISTAKE 1: Cross-Chunk Bleeding. No delimiters, no source/quarter labels --
# just the raw text of all four hits smashed together. Nothing marks which
# margin sentence belongs to which revenue sentence.
undelimited_context = " ".join(chunk["text"] for chunk in retrieved_chunks)

# MISTAKE 2: Causal Misalignment. Question first, context second -- the
# opposite of "context follow by question" that a left-to-right LLM needs.
messages = [
    {
        "role": "system",
        "content": (
            "You are a financial question-answering component in a RAG "
            "pipeline. Use only the retrieved context. Return one concise "
            "answer. Do not use outside knowledge."
        ),
    },
    {
        "role": "user",
        "content": f"""Question: {question}\n\nRetrieved context:\n{undelimited_context}""",
    },
]

print("\nBAD STRUCTURE:")
print("  - No chunk delimiters/source labels (Cross-Chunk Bleeding)")
print("  - Question placed BEFORE context (Causal Misalignment)")
print(f"\nPrompt sent to model:\n{messages[-1]['content']}")

response = client.chat.completions.create(
    model="local-llm",
    messages=messages,
    user="opensearchcon-demo",
    temperature=0.0,
    max_tokens=64,
)

assistant_text = response.choices[0].message.content or ""
print(f"\nSLM Response (Bad Structure):\n{assistant_text}")
print("\nExpected answer: 46.9 percent (the margin attached to the $94.9B/Q3 chunk)")
