from __future__ import annotations
import os
from urllib.parse import urlparse
import weaviate
import atexit

_CLIENT: weaviate.WeaviateClient | None = None

def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if not v:
        return default
    try:
        return int(v)
    except ValueError:
        return default

def get_client() -> weaviate.WeaviateClient:
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
    parsed = urlparse(url)

    http_host = os.getenv("WEAVIATE_HTTP_HOST", parsed.hostname or "localhost")
    http_port = _env_int("WEAVIATE_HTTP_PORT", parsed.port or 8080)
    http_secure = os.getenv("WEAVIATE_HTTP_SECURE", "false").lower() == "true" or parsed.scheme == "https"

    grpc_host = os.getenv("WEAVIATE_GRPC_HOST", http_host)
    grpc_port = _env_int("WEAVIATE_GRPC_PORT", 50051)
    grpc_secure = os.getenv("WEAVIATE_GRPC_SECURE", "false").lower() == "true"

    _CLIENT = weaviate.connect_to_custom(
        http_host=http_host,
        http_port=http_port,
        http_secure=http_secure,
        grpc_host=grpc_host,
        grpc_port=grpc_port,
        grpc_secure=grpc_secure,
    )
    atexit.register(_CLIENT.close)
    return _CLIENT
