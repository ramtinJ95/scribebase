from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

from scribebase.config import OCRProviderConfig
from scribebase.models import OCRResult


class ShellOCRProvider:
    def __init__(self, config: OCRProviderConfig, name: str = "shell"):
        self.config = config
        self.name = name

    def format_command(self, image_path: Path, output_md_path: Path, metadata: dict) -> list[str]:
        output_json = output_md_path.with_suffix(".json")
        values = {
            "input_image": str(image_path),
            "output_md": str(output_md_path),
            "output_json": str(output_json),
            "page_number": str(metadata.get("page_number", "")),
            "source_id": str(metadata.get("source_id", "")),
            "base_url": self.config.base_url or "",
            "model_name": self.config.model_name or "",
        }
        command = self.config.command.format(**values)
        return shlex.split(command)

    def ocr_image(self, image_path: Path, output_md_path: Path, metadata: dict) -> OCRResult:
        output_md_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = self.format_command(image_path, output_md_path, metadata)
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=self.config.timeout_seconds,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "OCR command failed "
                f"(exit {completed.returncode}): {completed.stderr.strip() or completed.stdout.strip()}"
            )
        if not output_md_path.exists() or not output_md_path.read_text().strip():
            raise RuntimeError(f"OCR command did not write non-empty Markdown: {output_md_path}")
        return OCRResult(
            markdown_path=output_md_path,
            text=output_md_path.read_text(),
            provider=self.name,
            model=self.config.model_name,
            raw_output_path=output_md_path.with_suffix(".json")
            if output_md_path.with_suffix(".json").exists()
            else None,
        )
