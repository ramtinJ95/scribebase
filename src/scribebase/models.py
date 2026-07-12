from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


SourceType = Literal[
    "book",
    "notes",
    "paper",
    "article",
    "transcript",
    "documentation",
    "snippet",
    "other",
]
Language = Literal["en", "sv", "mixed", "unknown"]


def normalize_tags(tags: list[str] | str | None) -> list[str]:
    if tags is None:
        return []
    if isinstance(tags, str):
        values = tags.split(",")
    elif isinstance(tags, list):
        if not all(isinstance(value, str) for value in tags):
            raise ValueError("tags must be a comma-separated string or a list of strings")
        values = [part for value in tags for part in value.split(",")]
    else:
        raise ValueError("tags must be a comma-separated string or a list of strings")
    return [value.strip() for value in values if value.strip()]


class ExtractionSummary(BaseModel):
    pages_total: int = 0
    pages_extracted_with_pymupdf4llm: int = 0
    pages_ocr: int = 0
    ocr_provider: str | None = None
    ocr_model: str | None = None


class EmbeddingSummary(BaseModel):
    embedding_model: str | None = None
    embedding_dimension: int | None = None
    embedding_base_url: str | None = None
    indexed_in_weaviate: bool = False
    weaviate_collection: str | None = None
    index_operation_id: str | None = None


class GenericMetadata(BaseModel):
    tags: list[str] = Field(default_factory=list)
    origin: str | None = None
    publisher: str | None = None
    author: str | None = None
    created_at_source: datetime | None = None
    updated_at_source: datetime | None = None
    retrieved_at: datetime | None = None
    url: str | None = None
    canonical_url: str | None = None
    external_id: str | None = None
    collection: str | None = None
    summary: str | None = None

    @field_validator("tags", mode="before")
    @classmethod
    def _normalize_tags(cls, value: list[str] | str | None) -> list[str]:
        return normalize_tags(value)


class SourceMetadataInput(GenericMetadata):
    model_config = ConfigDict(extra="ignore")

    title: str | None = None
    source_type: SourceType | None = None
    course: str | None = None
    chapter: str | None = None
    language: Language | None = None


class SourceManifest(GenericMetadata):
    schema_version: str = "1.0"
    source_id: str
    title: str
    source_type: SourceType = "other"
    course: str | None = None
    chapter: str | None = None
    language: Language = "unknown"
    identity_key: str | None = None
    content_sha256: str | None = None
    original_path: str
    data_dir: str
    created_at: datetime
    updated_at: datetime
    extraction_summary: ExtractionSummary = Field(default_factory=ExtractionSummary)
    embedding_summary: EmbeddingSummary = Field(default_factory=EmbeddingSummary)


class PageMetadata(BaseModel):
    source_id: str
    page_number: int
    page_index: int
    input_type: Literal["pdf_page", "image", "markdown", "text"]
    text_layer_detected: bool
    extraction_method: Literal["pymupdf4llm", "pymupdf", "ocr", "markdown", "text", "failed"]
    ocr_provider: str | None = None
    ocr_model: str | None = None
    image_path: str | None = None
    markdown_path: str
    char_count: int = 0
    word_count: int = 0
    quality_flags: list[str] = Field(default_factory=list)


class OCRResult(BaseModel):
    markdown_path: Path
    text: str
    provider: str
    model: str | None = None
    confidence: float | None = None
    warnings: list[str] = Field(default_factory=list)
    raw_output_path: Path | None = None


class TextQuality(BaseModel):
    char_count: int
    word_count: int
    alpha_ratio: float
    replacement_char_ratio: float
    avg_word_length: float
    line_count: int
    is_true_text: bool
    flags: list[str] = Field(default_factory=list)


class Chunk(GenericMetadata):
    chunk_id: str
    source_id: str
    source_type: str
    title: str
    course: str | None = None
    chapter: str | None = None
    section: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    chunk_index: int
    text: str
    file_path: str
    extraction_method: str
    ocr_model: str | None = None
    language: str | None = None
    embedding_model: str | None = None
    embedding_dimension: int | None = None
    chunker_version: str = "v1"
    created_at: datetime | None = None


class SearchFilters(BaseModel):
    source_id: str | None = None
    title: str | None = None
    source_type: str | None = None
    course: str | None = None
    chapter: str | None = None
    section: str | None = None
    tags: list[str] = Field(default_factory=list)
    origin: str | None = None
    publisher: str | None = None
    author: str | None = None
    url: str | None = None
    canonical_url: str | None = None
    external_id: str | None = None
    collection: str | None = None
    created_at_source_after: datetime | None = None
    created_at_source_before: datetime | None = None
    updated_at_source_after: datetime | None = None
    updated_at_source_before: datetime | None = None
    retrieved_at_after: datetime | None = None
    retrieved_at_before: datetime | None = None
    page_start: int | None = None
    page_end: int | None = None
    language: str | None = None

    @field_validator("tags", mode="before")
    @classmethod
    def _normalize_tags(cls, value: list[str] | str | None) -> list[str]:
        return normalize_tags(value)


class SearchResult(BaseModel):
    chunk: Chunk
    score: float | None = None
    explain_score: str | None = None
