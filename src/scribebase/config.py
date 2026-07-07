from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class WeaviateConfig(BaseModel):
    url: str = "http://localhost:8081"
    collection: str = "StudyChunk"
    vector_name: str = "text_vector"


class EmbeddingConfig(BaseModel):
    provider: str = "llamacpp"
    base_url: str = "http://localhost:8080/v1"
    model: str = "Qwen3-Embedding-0.6B-GGUF"
    timeout_seconds: int = 120
    batch_size: int = 16
    query_instruction: str = (
        "Instruct: Given a study question, retrieve relevant textbook passages that answer it\n"
        "Query: "
    )
    normalize: bool = True
    dimension: int | None = None


class PDFDetectionConfig(BaseModel):
    min_chars_per_page: int = 200
    min_alpha_ratio: float = 0.45
    max_replacement_char_ratio: float = 0.02


class OCRProviderConfig(BaseModel):
    command: str = "python ./scripts/run_local_ocr.py --input {input_image} --output {output_md}"
    timeout_seconds: int = 300
    model_name: str | None = "GLM-OCR"


class OCRConfig(BaseModel):
    default_provider: str = "shell"
    render_dpi: int = 300
    providers: dict[str, OCRProviderConfig] = Field(
        default_factory=lambda: {"shell": OCRProviderConfig()}
    )


class ChunkingConfig(BaseModel):
    target_chars: int = 600
    overlap_chars: int = 100
    min_chars: int = 150
    chunker_version: str = "v1"


class RetrievalConfig(BaseModel):
    alpha: float = 0.65
    top_k: int = 12
    candidate_k: int = 30


class LLMConfig(BaseModel):
    enabled: bool = False
    provider: str = "openai_compatible"
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-5.5-pro"
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.2


class AppConfig(BaseModel):
    data_dir: Path = Path(".study_local")
    weaviate: WeaviateConfig = Field(default_factory=WeaviateConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    pdf_detection: PDFDetectionConfig = Field(default_factory=PDFDetectionConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)

    @property
    def config_path(self) -> Path:
        return self.data_dir / "config.yaml"


def default_config() -> AppConfig:
    return AppConfig()


def load_config(config_path: Path | None = None) -> AppConfig:
    path = config_path or Path(".study_local/config.yaml")
    if not path.exists():
        return default_config()
    data = yaml.safe_load(path.read_text()) or {}
    return AppConfig.model_validate(data)


def config_to_yaml(config: AppConfig) -> str:
    data = config.model_dump(mode="json")
    return yaml.safe_dump(data, sort_keys=False)


def write_default_config(data_dir: Path, overwrite: bool = False) -> Path:
    config = default_config()
    config.data_dir = data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "config.yaml"
    if path.exists() and not overwrite:
        return path
    path.write_text(config_to_yaml(config))
    return path


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = deep_update(base[key], value)
        else:
            base[key] = value
    return base
