#!/usr/bin/env python3
"""GLM-OCR adapter for ScribeBase's shell OCR provider.

Expected runtime: a local llama.cpp server exposing the OpenAI-compatible chat API.
Recommended server:

llama-server \
  -m ./models/ocr/GLM-OCR-Q8_0.gguf \
  --mmproj ./models/ocr/mmproj-GLM-OCR-Q8_0.gguf \
  --alias GLM-OCR \
  -ngl 0 \
  --port 8082
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_PROMPT = """Convert this document image to clean Markdown.
Preserve headings, bullets, tables, code blocks, and command output.
Return only Markdown."""


def _data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def _post_json(url: str, payload: dict, timeout: int) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OCR server returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Cannot reach OCR server at {url}: {exc.reason}") from exc


def _extract_markdown(response: dict) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected OCR response shape: {response!r}") from exc
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        content = "\n".join(parts)
    text = str(content).strip()
    if not text:
        raise RuntimeError("OCR server returned empty content")
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Rendered page image path.")
    parser.add_argument("--output", required=True, help="Output Markdown path.")
    parser.add_argument("--output-json", help="Optional raw JSON response path.")
    parser.add_argument("--base-url", help="llama.cpp OpenAI-compatible base URL.")
    parser.add_argument("--model", help="Model alias exposed by llama.cpp.")
    args = parser.parse_args()

    image = Path(args.input)
    output = Path(args.output)
    raw_output = Path(args.output_json) if args.output_json else output.with_suffix(".json")
    if not image.exists():
        raise SystemExit(f"Input image does not exist: {image}")

    base_url = (
        args.base_url or os.getenv("SCRIBEBASE_OCR_BASE_URL", "http://localhost:8082/v1")
    ).rstrip("/")
    model = args.model or os.getenv("SCRIBEBASE_OCR_MODEL", "GLM-OCR")
    prompt = os.getenv("SCRIBEBASE_OCR_PROMPT", DEFAULT_PROMPT)
    max_tokens = int(os.getenv("SCRIBEBASE_OCR_MAX_TOKENS", "4096"))
    timeout = int(os.getenv("SCRIBEBASE_OCR_TIMEOUT_SECONDS", "900"))

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": _data_url(image)}},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }

    try:
        response = _post_json(f"{base_url}/chat/completions", payload, timeout)
        markdown = _extract_markdown(response)
    except Exception as exc:
        print(f"GLM-OCR adapter failed: {exc}", file=sys.stderr)
        print(
            "Start the OCR server with: llama-server -m ./models/ocr/GLM-OCR-Q8_0.gguf "
            "--mmproj ./models/ocr/mmproj-GLM-OCR-Q8_0.gguf --alias GLM-OCR "
            "-ngl 0 --port 8082",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown + "\n")
    raw_output.write_text(json.dumps(response, indent=2))


if __name__ == "__main__":
    main()
