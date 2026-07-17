from __future__ import annotations

import json

import httpx
import pytest

from scribebase.config import OCRProviderConfig, default_config
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


@pytest.mark.parametrize(
    ("health_payload", "expected"),
    [
        ({"status": "error"}, "/health status is 'error'"),
        ([], "/health response must be a JSON object"),
    ],
)
def test_glm_ocr_health_rejects_unhealthy_payload(
    monkeypatch,
    health_payload,
    expected,
) -> None:  # noqa: ANN001
    responses = {
        "http://localhost:8082/health": health_payload,
        "http://localhost:8082/v1/models": {"data": [{"id": "GLM-OCR"}]},
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

    assert ok is False
    assert expected in message


def test_glm_ocr_health_rejects_non_json_health_response(monkeypatch) -> None:
    def fake_get(url, **_kwargs):  # noqa: ANN001, ANN202
        if url.endswith("/health"):
            return httpx.Response(200, text="not json", request=httpx.Request("GET", url))
        raise AssertionError("models and props must not be checked after invalid health")

    monkeypatch.setattr(httpx, "get", fake_get)
    config = default_config()

    ok, message = check_ocr_provider_health(
        config.ocr.default_provider,
        config.ocr.providers[config.ocr.default_provider],
    )

    assert ok is False
    assert "/health response must be valid JSON" in message


@pytest.mark.parametrize(
    ("models_payload", "props_payload", "expected"),
    [
        (None, {"model_alias": "GLM-OCR", "modalities": {"vision": True}}, "JSON object"),
        ({"data": None}, {"model_alias": "GLM-OCR", "modalities": {}}, "array of objects"),
        ({"data": [None]}, {"model_alias": "GLM-OCR", "modalities": {}}, "array of objects"),
        ({"data": [{"id": "GLM-OCR"}]}, [], "/props response"),
    ],
)
def test_glm_ocr_health_reports_malformed_json(
    monkeypatch,
    models_payload,
    props_payload,
    expected,
) -> None:  # noqa: ANN001
    responses = {
        "http://localhost:8082/health": {"status": "ok"},
        "http://localhost:8082/v1/models": models_payload,
        "http://localhost:8082/props": props_payload,
    }

    def fake_get(url, **_kwargs):  # noqa: ANN001, ANN202
        return httpx.Response(
            200,
            content=json.dumps(responses[url]).encode(),
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr(httpx, "get", fake_get)
    config = default_config()

    ok, message = check_ocr_provider_health(
        config.ocr.default_provider,
        config.ocr.providers[config.ocr.default_provider],
    )

    assert ok is False
    assert expected in message
    assert "No OCR fallback will be used" in message


def test_ocr_health_rejects_missing_command_executable(tmp_path) -> None:
    provider = OCRProviderConfig(command=str(tmp_path / "missing-adapter"))

    ok, message = check_ocr_provider_health("custom", provider)

    assert ok is False
    assert "Missing OCR executable" in message


def test_ocr_health_rejects_non_executable_command(tmp_path) -> None:
    executable = tmp_path / "ocr-adapter"
    executable.write_text("#!/bin/sh\n")
    executable.chmod(0o644)
    provider = OCRProviderConfig(command=str(executable))

    ok, message = check_ocr_provider_health("custom", provider)

    assert ok is False
    assert "OCR executable is not executable" in message
