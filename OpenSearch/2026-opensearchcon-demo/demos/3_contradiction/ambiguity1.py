from openai import OpenAI

# Connecting to the local inference engine (e.g., llama.cpp server)
client = OpenAI(base_url="http://127.0.0.1:8001/v1", api_key="not-needed")

# Simulated OpenSearch hits. The score is intentionally kept OUT of the LLM
# prompt. In a healthy RAG pipeline, the application uses the score to decide
# which chunks are allowed into the model's context window.
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

# The correct pipeline rejects semantically related but insufficiently relevant
# chunks before prompt construction.
SCORE_CUTOFF = 0.80
selected_chunks = [
    chunk for chunk in retrieved_chunks if chunk["score"] >= SCORE_CUTOFF
]

print(f"\nApplying relevance cutoff: score >= {SCORE_CUTOFF:.2f}")
for chunk in retrieved_chunks:
    disposition = "KEEP" if chunk in selected_chunks else "DROP"
    print(f"  {disposition:4}  score={chunk['score']:.3f}  {chunk['origin']}")

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
    temperature=0.0,
    max_tokens=64,
)

assistant_text = response.choices[0].message.content or ""
print(f"\nSLM Response 1 (Score Cutoff Applied):\n{assistant_text}")
