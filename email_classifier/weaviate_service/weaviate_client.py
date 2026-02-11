from __future__ import annotations

import atexit
import logging
import os
from urllib.parse import urlparse

import weaviate

_CLIENT: weaviate.WeaviateClient | None = None
logger = logging.getLogger("email_classifier.weaviate_client")


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if not v:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _connect_custom() -> weaviate.WeaviateClient:
    try:
        url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
        parsed = urlparse(url)

        http_host = os.getenv("WEAVIATE_HTTP_HOST", parsed.hostname or "localhost")
        http_port = _env_int("WEAVIATE_HTTP_PORT", parsed.port or 8080)
        http_secure = (
            os.getenv("WEAVIATE_HTTP_SECURE", "false").lower() == "true" or parsed.scheme == "https"
        )

        grpc_host = os.getenv("WEAVIATE_GRPC_HOST", http_host)
        grpc_port = _env_int("WEAVIATE_GRPC_PORT", 50051)
        grpc_secure = os.getenv("WEAVIATE_GRPC_SECURE", "false").lower() == "true"

        skip_init_checks = os.getenv("WEAVIATE_SKIP_INIT_CHECKS", "true").lower() == "true"

        return weaviate.connect_to_custom(
            http_host=http_host,
            http_port=http_port,
            http_secure=http_secure,
            grpc_host=grpc_host,
            grpc_port=grpc_port,
            grpc_secure=grpc_secure,
            skip_init_checks=skip_init_checks,
        )
    except Exception as e:
        logger.exception("Failed to create Weaviate client: %s", e)
        raise RuntimeError(f"Failed to create Weaviate client: {e}") from e


def get_client() -> weaviate.WeaviateClient:
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    _CLIENT = _connect_custom()
    try:
        atexit.register(_CLIENT.close)
    except Exception as e:
        logger.warning("Failed to register Weaviate client close handler: %s", e)
    return _CLIENT


def get_fresh_client() -> weaviate.WeaviateClient:
    """Per-call client for concurrent operations."""
    return _connect_custom()
