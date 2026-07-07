from __future__ import annotations

import math
from typing import Iterable

import httpx

from scribebase.config import EmbeddingConfig


class LlamaCppEmbeddingClient:
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.base_url = config.base_url.rstrip("/")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = httpx.post(
            f"{self.base_url}/embeddings",
            json={
                "model": self.config.model,
                "input": texts,
                "encoding_format": "float",
            },
            timeout=self.config.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json().get("data", [])
        ordered = sorted(data, key=lambda item: item.get("index", 0))
        embeddings = [item["embedding"] for item in ordered]
        if self.config.normalize:
            embeddings = [_normalize(vec) for vec in embeddings]
        return embeddings

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([f"{self.config.query_instruction}{query}"])[0]

    def embed_batches(self, texts: list[str]) -> Iterable[list[list[float]]]:
        batch_size = self.config.batch_size
        for start in range(0, len(texts), batch_size):
            yield self.embed_texts(texts[start : start + batch_size])

    def detect_dimension(self) -> int:
        return len(self.embed_texts(["dimension test"])[0])

    def check_health(self) -> tuple[bool, str]:
        try:
            response = httpx.get(f"{self.base_url}/models", timeout=5)
            if response.status_code < 500:
                model_ids = _model_ids(response.json())
                suffix = f"; server models={', '.join(model_ids[:3])}" if model_ids else ""
                return True, f"/v1/models reachable; configured model={self.config.model}{suffix}"
        except Exception:
            pass
        try:
            dim = self.detect_dimension()
            return True, f"embeddings reachable, dimension={dim}"
        except Exception as exc:
            return False, str(exc)


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if not norm:
        return vec
    return [x / norm for x in vec]


def _model_ids(payload: dict) -> list[str]:
    ids: list[str] = []
    for item in payload.get("data", []):
        model_id = item.get("id")
        if model_id:
            ids.append(str(model_id))
    return ids
