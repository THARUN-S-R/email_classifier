from __future__ import annotations
import os
from urllib.parse import urlparse
import weaviate

def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if not v:
        return default
    try:
        return int(v)
    except ValueError:
        return default

def get_client() -> weaviate.WeaviateClient:
    url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
    parsed = urlparse(url)

    http_host = os.getenv("WEAVIATE_HTTP_HOST", parsed.hostname or "localhost")
    http_port = _env_int("WEAVIATE_HTTP_PORT", parsed.port or 8080)
    http_secure = os.getenv("WEAVIATE_HTTP_SECURE", "false").lower() == "true" or parsed.scheme == "https"

    grpc_host = os.getenv("WEAVIATE_GRPC_HOST", http_host)
    grpc_port = _env_int("WEAVIATE_GRPC_PORT", 50051)
    grpc_secure = os.getenv("WEAVIATE_GRPC_SECURE", "false").lower() == "true"

    return weaviate.connect_to_custom(
        http_host=http_host,
        http_port=http_port,
        http_secure=http_secure,
        grpc_host=grpc_host,
        grpc_port=grpc_port,
        grpc_secure=grpc_secure,
    )
