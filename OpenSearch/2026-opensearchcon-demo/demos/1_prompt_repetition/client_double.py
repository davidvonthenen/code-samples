from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8001/v1", api_key="not-needed")
messages = []

prompt = """
Here's a list (potentially with repetitions) of names:
Carlos Davis, Dale Sims, Carlos Davis, Dale Sims, Stephen Cruz, Dale Sims, Finnian Ross, Stephen Cruz, Stephen Cruz, Gregory Collins, Dale Sims, Stephen Cruz, Carlos Davis, Stephen Cruz, Dale Sims, Dale Sims, Stephen Cruz, Stephen Cruz, Leonard Kalman, Bruce Phillips, Raymond Roberts, Dale White, Leonard Kalman, Finnian Ross, James Wright, Finnian Ross, Raymond Roberts, Dale Sims, Dale Sims, Leonard Kalman, Dale Sims, Carlos Davis, Leonard Kalman, Bruce Phillips, Dale Sims, Raymond Roberts, Gregory Collins, Gregory Collins, Dale Sims, Finnian Ross

What is the single name that appears right between Carlos Davis and Bruce Phillips?

Carlos Davis, Dale Sims, Carlos Davis, Dale Sims, Stephen Cruz, Dale Sims, Finnian Ross, Stephen Cruz, Stephen Cruz, Gregory Collins, Dale Sims, Stephen Cruz, Carlos Davis, Stephen Cruz, Dale Sims, Dale Sims, Stephen Cruz, Stephen Cruz, Leonard Kalman, Bruce Phillips, Raymond Roberts, Dale White, Leonard Kalman, Finnian Ross, James Wright, Finnian Ross, Raymond Roberts, Dale Sims, Dale Sims, Leonard Kalman, Dale Sims, Carlos Davis, Leonard Kalman, Bruce Phillips, Dale Sims, Raymond Roberts, Gregory Collins, Gregory Collins, Dale Sims, Finnian Ross

What is the single name that appears right between Carlos Davis and Bruce Phillips?
"""

messages = []
messages.append({"role": "user", "content": prompt})
response = client.chat.completions.create(
    model="local-llm",
    messages=messages,
    user="opensearchcon-demo",
    temperature=0.0,
)
assistant_text = response.choices[0].message.content or ""
messages.append({"role": "assistant", "content": assistant_text})
print(f"\nSLM Response 2:\n{assistant_text}")
