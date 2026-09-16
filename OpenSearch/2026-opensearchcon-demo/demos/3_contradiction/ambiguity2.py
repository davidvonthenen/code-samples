from openai import OpenAI

# Connecting to the local inference engine (e.g., llama.cpp server)
client = OpenAI(base_url="http://127.0.0.1:8001/v1", api_key="not-needed")

# These are the SAME simulated OpenSearch hits used by ambiguity1.py.
# The application can see the scores and document origins, but the LLM only
# receives chunk text. This mirrors a RAG pipeline that retrieves broadly and
# then fails to enforce a minimum relevance score before prompt construction.
retrieved_chunks = [
    {
        "score": 0.932,
        "origin": "Apple Inc. quarterly filing",
        "text": (
            "For the recent quarter, total Apple revenue was $90.7 billion. "
            "The reported figure is the consolidated quarterly revenue total."
        ),
    },
    {
        "score": 0.641,
        "origin": "Apple Hospitality REIT - chunk 17",
        "text": (
            "For the recent quarter, total Apple revenue was $352.4 million. "
            "The reported figure is the consolidated quarterly revenue total."
        ),
    },
    {
        "score": 0.617,
        "origin": "Apple Hospitality REIT - chunk 18",
        "text": (
            "The quarterly report records total Apple revenue of $352.4 million. "
            "This is the final revenue figure reported for the period."
        ),
    },
    {
        "score": 0.594,
        "origin": "Apple Hospitality REIT - chunk 19",
        "text": (
            "Total Apple revenue for the quarter was $352.4 million. "
            "The amount reflects the finalized quarterly results."
        ),
    },
    {
        "score": 0.566,
        "origin": "Apple Hospitality REIT - chunk 20",
        "text": (
            "The finalized results show total Apple revenue at $352.4 million "
            "for the recent quarter."
        ),
    },
    {
        "score": 0.541,
        "origin": "Apple Hospitality REIT - chunk 21",
        "text": (
            "For the quarter, the recorded total Apple revenue was $352.4 million. "
            "This value appears in the final quarterly summary."
        ),
    },
]

# FAILURE CASE: no relevance-score cutoff.
# Every retrieved chunk is pushed into the context window even though five of
# the six hits are materially less relevant than the first hit.
selected_chunks = retrieved_chunks

print("\nNo relevance cutoff applied. Every retrieved chunk enters the prompt:")
for chunk in selected_chunks:
    print(f"  KEEP  score={chunk['score']:.3f}  {chunk['origin']}")

# Deliberately omit score and origin metadata from the model-visible context.
# The distractors are written in nearly identical financial language so the
# model cannot escape by spotting words such as 'hotel', 'occupancy', 'iPhone',
# 'agriculture', or explicit Entity/Source labels.
context = "\n\n".join(
    f"[Excerpt {index}]\n{chunk['text']}"
    for index, chunk in enumerate(selected_chunks, start=1)
)

messages = [
    {
        "role": "system",
        "content": (
            "You are a financial question-answering component in a RAG pipeline. "
            "Use only the retrieved context. Return one concise answer containing "
            "the revenue value and unit. Do not discuss ambiguity, conflicting "
            "excerpts, source identity, or alternative interpretations. Do not use "
            "outside knowledge."
        ),
    },
    {
        "role": "user",
        "content": f"""Retrieved context:\n{context}\n\nQuestion: What was the total Apple revenue for the recent quarter?""",
    },
]

response = client.chat.completions.create(
    model="local-llm",
    # model="gpt-5.4",
    messages=messages,
    user="opensearchcon-demo",
    # Keep generation deterministic. The failure should come from poisoned
    # retrieval context, not from sampling randomness.
    temperature=0.0,
    max_tokens=64,
)

assistant_text = response.choices[0].message.content or ""
print(f"\nSLM Response 2 (No Score Cutoff / Over-Retrieval):\n{assistant_text}")
