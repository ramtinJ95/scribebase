from __future__ import annotations

import hashlib
import json

from scribebase.config import EmbeddingConfig


EMBEDDING_PROFILE_VERSION = 1


def embedding_profile_fingerprint(config: EmbeddingConfig, dimension: int) -> str:
    """Return a stable identity for every setting that defines the embedding space."""
    if dimension <= 0:
        raise ValueError("Embedding dimension must be positive")
    payload = {
        "version": EMBEDDING_PROFILE_VERSION,
        "provider": config.provider,
        "model": config.model,
        "dimension": dimension,
        "query_instruction": config.query_instruction,
        "document_instruction": config.document_instruction,
        "normalize": config.normalize,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"v{EMBEDDING_PROFILE_VERSION}:{hashlib.sha256(encoded.encode()).hexdigest()}"
