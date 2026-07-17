from __future__ import annotations

import httpx

from scribebase.config import default_config
from scribebase.ocr.health import check_ocr_provider_health


def test_glm_ocr_health_validates_model_and_multimodal_capability(monkeypatch) -> None:
    responses = {
        "http://localhost:8082/health": {"status": "ok"},
        "http://localhost:8082/v1/models": {
            "data": [{"id": "GLM-OCR"}],
            "models": [{"model": "GLM-OCR", "capabilities": ["multimodal"]}],
        },
        "http://localhost:8082/props": {
            "model_alias": "GLM-OCR",
            "modalities": {"vision": True},
        },
    }

    def fake_get(url, **_kwargs):  # noqa: ANN001, ANN202
        return httpx.Response(200, json=responses[url], request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)
    config = default_config()

    ok, message = check_ocr_provider_health(
        config.ocr.default_provider,
        config.ocr.providers[config.ocr.default_provider],
    )

    assert ok is True
    assert "model=GLM-OCR" in message
    assert "multimodal=True" in message


def test_unavailable_glm_ocr_has_actionable_error_and_no_fallback(monkeypatch) -> None:
    def unavailable(url, **_kwargs):  # noqa: ANN001, ANN202
        raise httpx.ConnectError("connection refused", request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", unavailable)
    config = default_config()

    ok, message = check_ocr_provider_health(
        config.ocr.default_provider,
        config.ocr.providers[config.ocr.default_provider],
    )

    assert ok is False
    assert "glm_ocr" in message
    assert "http://localhost:8082/v1" in message
    assert "No OCR fallback will be used" in message
    assert "--mmproj ./models/ocr/mmproj-GLM-OCR-Q8_0.gguf" in message


def test_glm_ocr_health_rejects_server_without_vision(monkeypatch) -> None:
    responses = {
        "http://localhost:8082/health": {"status": "ok"},
        "http://localhost:8082/v1/models": {"data": [{"id": "GLM-OCR"}]},
        "http://localhost:8082/props": {
            "model_alias": "GLM-OCR",
            "modalities": {"vision": False},
        },
    }

    def fake_get(url, **_kwargs):  # noqa: ANN001, ANN202
        return httpx.Response(200, json=responses[url], request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)
    config = default_config()

    ok, message = check_ocr_provider_health(
        config.ocr.default_provider,
        config.ocr.providers[config.ocr.default_provider],
    )

    assert ok is False
    assert "does not advertise vision/multimodal capability" in message
