from __future__ import annotations

from pathlib import Path
from typing import Protocol

from scribebase.models import OCRResult


class OCRProvider(Protocol):
    name: str

    def ocr_image(
        self,
        image_path: Path,
        output_md_path: Path,
        metadata: dict,
    ) -> OCRResult: ...
