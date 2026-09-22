# Demo 3: The Hoarder's Prompt and Adversarial Ambiguity

In the world of retrieval-augmented generation, there is a hilarious tendency to treat the language model's context window like a storage unit, shoving every available OpenSearch hit into the prompt. This "hoarder's prompt" acts as an unintentional adversarial attack, where excessive, marginally relevant data overwhelms the model. By intentionally injecting ambiguity through over-retrieval, we can observe how distractors fracture attention, allowing us to refactor our retrieval thresholds and radically improve prompt quality.

### Execution Steps

* Using a Python virtual environment (such as miniconda, uv, venv, etc), run `pip install -r requirements.txt`.
* In one terminal, start the local inference engine by executing `python slm_service.py` to bring the REST server online.
* In a second terminal, execute `python ambiguity1.py`. This simulates a healthy RAG pipeline by applying a strict relevance cutoff (`score >= 0.80`), ensuring only semantically precise chunks construct the prompt.
* Execute `python ambiguity2.py`. This script deliberately omits the relevance cutoff, forcing all retrieved chunks-including five highly similar distractors-into the model-visible context.
