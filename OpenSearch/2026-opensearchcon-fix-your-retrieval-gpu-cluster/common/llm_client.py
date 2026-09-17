"""Amazon Bedrock client -- replaces the reference workshop's local GGUF model.

Every stage calls ``generate()`` in-process. There is no separate
``llm_service.py`` to start/stop; auth is via standard AWS credential
resolution (env vars, shared config/credentials file, or an assumed role).
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Optional

import boto3

from .config import Settings, load_settings
from .logging import get_logger

LOGGER = get_logger(__name__)


@lru_cache(maxsize=1)
def _client(region: str):
    return boto3.client("bedrock-runtime", region_name=region)


def generate(
    prompt: str,
    *,
    settings: Optional[Settings] = None,
    system: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> str:
    """Generate one completion from the configured Bedrock model.

    Uses the Anthropic Messages API request/response shape, which is what
    Bedrock expects for Anthropic Claude models. If you standardize on a
    different model family (Llama, Nova, Mistral, ...), adjust the request
    body shape here -- the rest of the codebase only depends on this
    function's signature, not on Bedrock's wire format.
    """
    settings = settings or load_settings()
    if not settings.bedrock_model_id:
        raise RuntimeError(
            "BEDROCK_MODEL_ID is not set. Add it to your .env, e.g. "
            "BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0"
        )

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": int(max_tokens if max_tokens is not None else settings.bedrock_max_tokens),
        "temperature": float(temperature if temperature is not None else settings.bedrock_temperature),
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        body["system"] = system

    client = _client(settings.aws_region)
    LOGGER.info("Invoking Bedrock model '%s' in %s", settings.bedrock_model_id, settings.aws_region)
    response = client.invoke_model(modelId=settings.bedrock_model_id, body=json.dumps(body))
    payload = json.loads(response["body"].read())

    content = payload.get("content", [])
    if content and isinstance(content, list):
        return "".join(part.get("text", "") for part in content if part.get("type") == "text").strip()
    return str(payload).strip()


__all__ = ["generate"]
