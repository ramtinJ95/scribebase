import httpx
import pytest

from scribebase.config import EmbeddingConfig
from scribebase.embeddings.llamacpp_client import LlamaCppEmbeddingClient
from scribebase.errors import DependencyUnavailableError


class FakeResponse:
    def raise_for_status(self) -> None:
        pass

    def json(self):
        return {
            "data": [
                {"index": 1, "embedding": [0.0, 4.0]},
                {"index": 0, "embedding": [3.0, 4.0]},
            ]
        }


class FakeModelsResponse:
    def __init__(self, model_ids: list[str], status_code: int = 200) -> None:
        self.model_ids = model_ids
        self.status_code = status_code

    @property
    def is_success(self) -> bool:
        return 200 <= self.status_code < 300

    def json(self):
        return {"data": [{"id": model_id} for model_id in self.model_ids]}


def test_embedding_client_parses_openai_style_response(monkeypatch) -> None:
    calls = []

    def fake_post(url, json, timeout):
        calls.append((url, json, timeout))
        return FakeResponse()

    monkeypatch.setattr("scribebase.embeddings.llamacpp_client.httpx.post", fake_post)
    client = LlamaCppEmbeddingClient(EmbeddingConfig(normalize=False))
    embeddings = client.embed_texts(["a", "b"])
    assert embeddings == [[3.0, 4.0], [0.0, 4.0]]
    assert calls[0][0] == "http://localhost:8080/v1/embeddings"
    assert calls[0][1]["encoding_format"] == "float"


def test_query_embedding_uses_instruction(monkeypatch) -> None:
    captured = {}

    def fake_post(url, json, timeout):
        captured.update(json)
        return FakeResponse()

    monkeypatch.setattr("scribebase.embeddings.llamacpp_client.httpx.post", fake_post)
    client = LlamaCppEmbeddingClient(EmbeddingConfig(normalize=False, query_instruction="Q: "))
    client.embed_query("working memory")
    assert captured["input"] == ["Q: working memory"]


def test_batch_embedding_uses_document_instruction(monkeypatch) -> None:
    captured_inputs = []

    def fake_post(url, json, timeout):
        captured_inputs.append(json["input"])
        return FakeResponse()

    monkeypatch.setattr("scribebase.embeddings.llamacpp_client.httpx.post", fake_post)
    client = LlamaCppEmbeddingClient(
        EmbeddingConfig(normalize=False, document_instruction="passage: ", batch_size=2)
    )
    list(client.embed_batches(["one", "two", "three"]))
    assert captured_inputs == [["passage: one", "passage: two"], ["passage: three"]]


def test_query_embedding_does_not_get_document_instruction(monkeypatch) -> None:
    captured = {}

    def fake_post(url, json, timeout):
        captured.update(json)
        return FakeResponse()

    monkeypatch.setattr("scribebase.embeddings.llamacpp_client.httpx.post", fake_post)
    client = LlamaCppEmbeddingClient(
        EmbeddingConfig(normalize=False, query_instruction="query: ", document_instruction="passage: ")
    )
    client.embed_query("working memory")
    assert captured["input"] == ["query: working memory"]


def test_dimension_probe_is_unprefixed(monkeypatch) -> None:
    captured = {}

    def fake_post(url, json, timeout):
        captured.update(json)
        return FakeResponse()

    monkeypatch.setattr("scribebase.embeddings.llamacpp_client.httpx.post", fake_post)
    client = LlamaCppEmbeddingClient(
        EmbeddingConfig(normalize=False, document_instruction="passage: ")
    )
    client.detect_dimension()
    assert captured["input"] == ["dimension test"]


def test_transport_failure_is_typed_as_dependency_unavailable(monkeypatch) -> None:  # noqa: ANN001
    def fail_post(*_args, **_kwargs):  # noqa: ANN002, ANN003
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr("scribebase.embeddings.llamacpp_client.httpx.post", fail_post)

    with pytest.raises(DependencyUnavailableError, match="connection refused"):
        LlamaCppEmbeddingClient(EmbeddingConfig()).embed_texts(["text"])


def test_health_check_propagates_malformed_embedding_url(monkeypatch) -> None:  # noqa: ANN001
    def invalid_url(*_args, **_kwargs):  # noqa: ANN002, ANN003
        raise httpx.UnsupportedProtocol("missing URL scheme")

    monkeypatch.setattr("scribebase.embeddings.llamacpp_client.httpx.get", invalid_url)

    with pytest.raises(httpx.UnsupportedProtocol, match="missing URL scheme"):
        LlamaCppEmbeddingClient(EmbeddingConfig()).check_health()


def test_health_check_requires_configured_model_alias(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(
        "scribebase.embeddings.llamacpp_client.httpx.get",
        lambda *_args, **_kwargs: FakeModelsResponse(["Qwen3-Embedding-4B-Q4_K_M.gguf"]),
    )

    ok, message = LlamaCppEmbeddingClient(EmbeddingConfig()).check_health()

    assert ok is False
    assert "is not advertised" in message


def test_health_check_accepts_configured_model_alias(monkeypatch) -> None:  # noqa: ANN001
    config = EmbeddingConfig()
    monkeypatch.setattr(
        "scribebase.embeddings.llamacpp_client.httpx.get",
        lambda *_args, **_kwargs: FakeModelsResponse([config.model]),
    )

    ok, message = LlamaCppEmbeddingClient(config).check_health()

    assert ok is True
    assert config.model in message
