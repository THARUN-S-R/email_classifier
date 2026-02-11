from __future__ import annotations
import os
import logging

logger = logging.getLogger("email_classifier.config")

# Collections
THREAD_CLASS = "EmailThread"
DETAIL_CLASS = "EmailThreadDetail"
DAILY_SUMMARY_CLASS = "DailySummary"

# Retrieval defaults
THREAD_FETCH_LIMIT = int(os.getenv("THREAD_FETCH_LIMIT", "10"))
DETAIL_FETCH_LIMIT = int(os.getenv("DETAIL_FETCH_LIMIT", "8"))
SUMMARY_FETCH_LIMIT = int(os.getenv("SUMMARY_FETCH_LIMIT", "5"))
FETCH_CANDIDATES_LIMIT = int(os.getenv("FETCH_CANDIDATES_LIMIT", "50"))

# LLM safety thresholds
CONFIDENCE_MIN = float(os.getenv("CONFIDENCE_MIN", "0.75"))

# Truncation
MAX_DETAIL_CHARS = int(os.getenv("MAX_DETAIL_CHARS", "4000"))
MAX_LLM_THREAD_CHARS = int(os.getenv("MAX_LLM_THREAD_CHARS", "2200"))

def warn_if_missing_llm_keys() -> None:
    provider_keys = [
        "OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LITELLM_API_KEY"
    ]
    if not any(os.getenv(k) for k in provider_keys):
        logger.warning("No LLM API key found in env. Set one of: %s", ", ".join(provider_keys))
