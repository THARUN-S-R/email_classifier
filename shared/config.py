from __future__ import annotations
import os, logging

logger = logging.getLogger("email_classifier.config")

def warn_if_missing_llm_keys() -> None:
    provider_keys = [
        "OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LITELLM_API_KEY"
    ]
    if not any(os.getenv(k) for k in provider_keys):
        logger.warning("No LLM API key found in env. Set one of: %s", ", ".join(provider_keys))
