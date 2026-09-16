# Demo 1: Anchoring Attention with Prompt Duplication

The core idea behind this demonstration is that language models can suffer from severe attention degradation when processing extensive context, but appending a complete copy of your prompt at the very end of the input can theoretically improve response quality. We test this mechanism by feeding a highly repetitive list of names to a local Small Language Model (SLM) to evaluate how context layout influences its ability to pinpoint a specific target within the noise.

### Steps to Run the Demo

* Using a Python virtual environment (such as miniconda, uv, venv, etc), run `pip install -r requirements.txt`.
* In one terminal, tnitialize the local language model server by running `python slm_service.py`.
* In a second terminal, Execute the baseline test by running `python client_single.py`, which sends a single instance of the prompt to the model.
* Execute the experimental test by running `python client_double.py`, which sends a duplicated instance of the prompt to the model.

Reflecting on my own work with multi-cloud provisioning and natural language understanding, it is remarkable how much the physical layout of information dictates performance. We spend vast amounts of time optimizing our OpenSearch vector databases and Podman containers, yet the raw presentation of the text itself remains one of our most critical, yet frequently ignored, tuning levers. Fire up your local runtime, execute these scripts, and observe the mechanics of context processing firsthand!