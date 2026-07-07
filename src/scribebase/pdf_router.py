from __future__ import annotations

import re

from .config import PDFDetectionConfig
from .models import TextQuality


def evaluate_text_quality(text: str, config: PDFDetectionConfig | None = None) -> TextQuality:
    config = config or PDFDetectionConfig()
    stripped = text.strip()
    chars = len(stripped)
    words = re.findall(r"\b[\w'-]+\b", stripped, flags=re.UNICODE)
    alnum = sum(ch.isalnum() for ch in stripped)
    replacement = stripped.count("�")
    flags: list[str] = []
    alpha_ratio = alnum / chars if chars else 0.0
    replacement_ratio = replacement / chars if chars else 0.0
    avg_word_length = sum(len(w) for w in words) / len(words) if words else 0.0
    line_count = len([line for line in stripped.splitlines() if line.strip()])

    if chars < config.min_chars_per_page:
        flags.append("too_few_chars")
    if alpha_ratio < config.min_alpha_ratio:
        flags.append("low_alpha_ratio")
    if replacement_ratio > config.max_replacement_char_ratio:
        flags.append("replacement_chars")
    if avg_word_length > 35:
        flags.append("long_average_word")
    if line_count == 0:
        flags.append("no_lines")

    return TextQuality(
        char_count=chars,
        word_count=len(words),
        alpha_ratio=alpha_ratio,
        replacement_char_ratio=replacement_ratio,
        avg_word_length=avg_word_length,
        line_count=line_count,
        is_true_text=not flags,
        flags=flags,
    )
