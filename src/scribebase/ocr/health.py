from __future__ import annotations

import shlex
from pathlib import Path
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
        models = httpx.get(f"{provider.base_url.rstrip('/')}/models", timeout=5)
        models.raise_for_status()
        props = httpx.get(f"{root_url}/props", timeout=5)
        props.raise_for_status()
        models_payload = models.json()
        props_payload = props.json()
    except (httpx.HTTPError, ValueError) as exc:
        return False, _unavailable_message(provider_name, provider, str(exc))

    model_ids = _model_ids(models_payload)
    if provider.model_name and provider.model_name not in model_ids:
        return False, _unavailable_message(
            provider_name,
            provider,
            f"configured model {provider.model_name!r} is not loaded; server models={model_ids}",
        )
    if provider.model_name and props_payload.get("model_alias") != provider.model_name:
        return False, _unavailable_message(
            provider_name,
            provider,
            f"server model alias is {props_payload.get('model_alias')!r}, "
            f"expected {provider.model_name!r}",
        )
    if provider.require_multimodal and not _has_vision(models_payload, props_payload):
        return False, _unavailable_message(
            provider_name,
            provider,
            "server does not advertise vision/multimodal capability; load the required mmproj",
        )
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
    for part in parts[1:]:
        if part.endswith(".py") or part.startswith("./"):
            if not Path(part).exists():
                return False, f"Missing OCR adapter path: {part}"
            break
    return True, "adapter configured"


def _server_root(base_url: str) -> str:
    parsed = urlsplit(base_url)
    return urlunsplit((parsed.scheme, parsed.netloc, "", "", "")).rstrip("/")


def _model_ids(payload: dict) -> set[str]:
    ids = {
        str(item[key])
        for item in payload.get("data", []) + payload.get("models", [])
        for key in ("id", "model", "name")
        if item.get(key)
    }
    return ids


def _has_vision(models_payload: dict, props_payload: dict) -> bool:
    if props_payload.get("modalities", {}).get("vision") is True:
        return True
    return any(
        "multimodal" in item.get("capabilities", [])
        for item in models_payload.get("models", [])
    )


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
