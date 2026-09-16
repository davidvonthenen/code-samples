from openai import OpenAI

# Connecting to the local inference engine (e.g., llama.cpp / MLX server started
# by ../1_prompt_repetition/slm_service.py).
client = OpenAI(base_url="http://127.0.0.1:8001/v1", api_key="not-needed")

# SAME four already-grounded OpenSearch hits as bad_structure.py, in the SAME
# (field-grouped, not quarter-grouped) order. Only the STRUCTURE of the
# prompt changes -- this isolates presentation/labeling as the variable, not
# retrieval quality or hit order.
retrieved_chunks = [
    {
        "id": 1,
        "origin": "Apple Inc. 10-Q filing - Q3 FY2025 - Revenue",
        "text": "Total revenue for the quarter was $94.9 billion.",
    },
    {
        "id": 2,
        "origin": "Apple Inc. 10-Q filing - Q4 FY2025 - Revenue",
        "text": "Total revenue for the quarter was $102.5 billion.",
    },
    {
        "id": 3,
        "origin": "Apple Inc. 10-Q filing - Q4 FY2025 - Gross Margin",
        "text": "Gross margin for the quarter was 47.3 percent.",
    },
    {
        "id": 4,
        "origin": "Apple Inc. 10-Q filing - Q3 FY2025 - Gross Margin",
        "text": "Gross margin for the quarter was 46.9 percent.",
    },
]

question = "What was Apple's gross margin for the quarter with $94.9 billion in revenue?"

# FIX 1: Bounded chunks. Every chunk gets an explicit delimiter tag with a
# stable id + source attribute -- a human-readable, XML-style boundary marker
# (per "Let Me Speak Freely?", arXiv:2408.02442 -- LLMs are LANGUAGE models;
# heavy JSON/raw-payload dumps cost 5-20% failure rates vs. human-readable
# delimited text). The `source` label carries the quarter+field attribution
# that the raw text alone doesn't -- no positional/index guessing required to
# know which margin belongs to which quarter's revenue.
bounded_context = "\n".join(
    f'<chunk id="{chunk["id"]}" source="{chunk["origin"]}">\n{chunk["text"]}\n</chunk>'
    for chunk in retrieved_chunks
)

# FIX 2: Causal alignment. Context FIRST, question LAST. A non-reasoning,
# auto-regressive LLM reads left-to-right -- it needs every relevant fact in
# its context *before* it starts attending to what's being asked, otherwise
# it starts "reasoning" over an empty/partial context (Causal Misalignment /
# "premature interrogation").
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
        "content": f"""Retrieved context:\n{bounded_context}\n\nQuestion: {question}""",
    },
]

print("\nGOOD STRUCTURE:")
print("  - Bounded chunks with explicit id/source delimiters (fixes Cross-Chunk Bleeding)")
print("  - Context provided BEFORE the question (fixes Causal Misalignment)")
print(f"\nPrompt sent to model:\n{messages[-1]['content']}")

response = client.chat.completions.create(
    model="local-llm",
    messages=messages,
    user="opensearchcon-demo",
    temperature=0.0,
    max_tokens=64,
)

assistant_text = response.choices[0].message.content or ""
print(f"\nSLM Response (Good Structure):\n{assistant_text}")
print("\nExpected answer: 46.9 percent (the margin attached to the $94.9B/Q3 chunk)")
