from __future__ import annotations

import os
from pathlib import Path

import httpx

from scribebase.config import LLMConfig


class OpenAICompatibleChatClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.base_url = config.base_url.rstrip("/")

    @property
    def api_key(self) -> str | None:
        return os.getenv(self.config.api_key_env)

    def available(self) -> bool:
        return self.config.enabled and bool(self.api_key)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        if not self.api_key:
            raise RuntimeError(f"Missing API key env var: {self.config.api_key_env}")
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.config.model,
                "temperature": self.config.temperature,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
            timeout=180,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


def save_markdown(output_dir: Path, stem: str, content: str) -> Path:
    from datetime import datetime
    from scribebase.source_registry import slugify

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{slugify(stem)[:60]}.md"
    path.write_text(content)
    return path
