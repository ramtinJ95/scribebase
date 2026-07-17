from __future__ import annotations

import shlex
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

from scribebase.config import OCRProviderConfig


GLM_OCR_START_COMMAND = (
    "llama-server --model ./models/ocr/GLM-OCR-Q8_0.gguf "
    "--mmproj ./models/ocr/mmproj-GLM-OCR-Q8_0.gguf --alias GLM-OCR "
    "--ctx-size 8192 --parallel 1 --cache-ram 0 -ngl 0 "
    "--host 127.0.0.1 --port 8082"
)


def check_ocr_provider_health(
    provider_name: str,
    provider: OCRProviderConfig | None,
) -> tuple[bool, str]:
    if provider is None:
        return False, f"OCR provider is not configured: {provider_name}"
    command_ok, command_message = _check_command(provider)
    if not command_ok:
        return False, command_message
    if not provider.base_url:
        return True, f"provider={provider_name}; adapter configured"

    root_url = _server_root(provider.base_url)
    try:
        health = httpx.get(f"{root_url}/health", timeout=5)
        health.raise_for_status()
        try:
            health_payload = health.json()
        except ValueError as exc:
            raise ValueError("/health response must be valid JSON") from exc
        _validate_health(health_payload)
        models = httpx.get(f"{provider.base_url.rstrip('/')}/models", timeout=5)
        models.raise_for_status()
        props = httpx.get(f"{root_url}/props", timeout=5)
        props.raise_for_status()
        models_payload = models.json()
        props_payload = props.json()
        model_ids = _model_ids(models_payload)
        model_alias = _model_alias(props_payload)
        has_vision = _has_vision(models_payload, props_payload)
        if provider.model_name and provider.model_name not in model_ids:
            raise ValueError(
                f"configured model {provider.model_name!r} is not loaded; "
                f"server models={model_ids}"
            )
        if provider.model_name and model_alias != provider.model_name:
            raise ValueError(
                f"server model alias is {model_alias!r}, expected {provider.model_name!r}"
            )
        if provider.require_multimodal and not has_vision:
            raise ValueError(
                "server does not advertise vision/multimodal capability; "
                "load the required mmproj"
            )
    except (httpx.HTTPError, TypeError, ValueError) as exc:
        return False, _unavailable_message(provider_name, provider, str(exc))
    return (
        True,
        f"provider={provider_name}; model={provider.model_name}; "
        f"multimodal={provider.require_multimodal}; server={root_url}",
    )


def ensure_ocr_provider_ready(provider_name: str, provider: OCRProviderConfig) -> None:
    ok, message = check_ocr_provider_health(provider_name, provider)
    if not ok:
        raise RuntimeError(message)


def _check_command(provider: OCRProviderConfig) -> tuple[bool, str]:
    try:
        parts = shlex.split(
            provider.command.format(
                input_image="x",
                output_md="y",
                output_json="z",
                page_number=1,
                source_id="s",
                base_url=provider.base_url or "",
                model_name=provider.model_name or "",
            )
        )
    except Exception as exc:
        return False, f"Invalid OCR command template: {exc}"
    if not parts:
        return False, "Invalid OCR command template: command is empty"
    executable = parts[0]
    if "/" in executable:
        executable_path = Path(executable)
        if not executable_path.is_file():
            return False, f"Missing OCR executable: {executable}"
    elif shutil.which(executable) is None:
        return False, f"Missing OCR executable on PATH: {executable}"
    for part in parts[1:]:
        if part.endswith(".py") or part.startswith("./"):
            if not Path(part).exists():
                return False, f"Missing OCR adapter path: {part}"
            break
    return True, "adapter configured"


def _server_root(base_url: str) -> str:
    parsed = urlsplit(base_url)
    return urlunsplit((parsed.scheme, parsed.netloc, "", "", "")).rstrip("/")


def _model_ids(payload: Any) -> set[str]:
    data = _object_list(payload, "data")
    models = _object_list(payload, "models")
    ids = {
        str(item[key])
        for item in data + models
        for key in ("id", "model", "name")
        if item.get(key)
    }
    return ids


def _validate_health(payload: Any) -> None:
    if not isinstance(payload, dict):
        raise ValueError("/health response must be a JSON object")
    if payload.get("status") != "ok":
        raise ValueError(f"/health status is {payload.get('status')!r}, expected 'ok'")


def _model_alias(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        raise ValueError("/props response must be a JSON object")
    value = payload.get("model_alias")
    return str(value) if value is not None else None


def _has_vision(models_payload: Any, props_payload: Any) -> bool:
    if not isinstance(props_payload, dict):
        raise ValueError("/props response must be a JSON object")
    modalities = props_payload.get("modalities", {})
    if not isinstance(modalities, dict):
        raise ValueError("/props modalities must be a JSON object")
    if modalities.get("vision") is True:
        return True
    for item in _object_list(models_payload, "models"):
        capabilities = item.get("capabilities", [])
        if not isinstance(capabilities, list):
            raise ValueError("/v1/models capabilities must be a JSON array")
        if "multimodal" in capabilities:
            return True
    return False


def _object_list(payload: Any, key: str) -> list[dict]:
    if not isinstance(payload, dict):
        raise ValueError("/v1/models response must be a JSON object")
    value = payload.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"/v1/models {key} must be an array of objects")
    return value


def _unavailable_message(
    provider_name: str,
    provider: OCRProviderConfig,
    reason: str,
) -> str:
    hint = f" Start the dedicated GLM-OCR service with: {GLM_OCR_START_COMMAND}"
    return (
        f"OCR provider {provider_name!r} is unavailable at {provider.base_url}: {reason}."
        " No OCR fallback will be used."
        f"{hint if provider_name == 'glm_ocr' else ''}"
    )
