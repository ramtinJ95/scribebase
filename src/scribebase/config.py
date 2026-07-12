from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


CONFIG_ENV = "SCRIBEBASE_CONFIG"
DATA_DIR_ENV = "SCRIBEBASE_DATA_DIR"
HOST_ENV = "SCRIBEBASE_HOST"
PORT_ENV = "SCRIBEBASE_PORT"
API_TOKEN_ENV = "SCRIBEBASE_API_TOKEN"


class WeaviateConfig(BaseModel):
    url: str = "http://localhost:8081"
    collection: str = "Chunk"
    vector_name: str = "text_vector"


class EmbeddingConfig(BaseModel):
    provider: str = "llamacpp"
    base_url: str = "http://localhost:8080/v1"
    model: str = "Qwen3-Embedding-4B-Q4_K_M.gguf"
    timeout_seconds: int = 120
    batch_size: int = 8
    query_instruction: str = (
        "Instruct: Given a question, retrieve relevant source passages that answer it\nQuery: "
    )
    normalize: bool = True
    dimension: int | None = None


class PDFDetectionConfig(BaseModel):
    min_chars_per_page: int = 200
    min_alpha_ratio: float = 0.45
    max_replacement_char_ratio: float = 0.02


class OCRProviderConfig(BaseModel):
    command: str = "./scripts/run_local_ocr.py --input {input_image} --output {output_md}"
    timeout_seconds: int = 900
    model_name: str | None = "GLM-OCR"
    render_dpi: int | None = None


class OCRConfig(BaseModel):
    default_provider: str = "shell"
    render_dpi: int = 300
    providers: dict[str, OCRProviderConfig] = Field(
        default_factory=lambda: {
            "shell": OCRProviderConfig(),
            "apple_vision": OCRProviderConfig(
                command=(
                    "swift ./scripts/run_apple_vision_ocr.swift "
                    "--input {input_image} --output {output_md}"
                ),
                timeout_seconds=120,
                model_name="Apple Vision",
                render_dpi=200,
            ),
        }
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


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8765
    api_token_env: str = API_TOKEN_ENV
    max_upload_bytes: int = Field(default=250 * 1024 * 1024, gt=0)
    max_active_jobs: int = Field(default=20, gt=0)
    worker_poll_seconds: float = Field(default=2.0, gt=0)


class AppConfig(BaseModel):
    data_dir: Path = Path(".scribebase")
    weaviate: WeaviateConfig = Field(default_factory=WeaviateConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    pdf_detection: PDFDetectionConfig = Field(default_factory=PDFDetectionConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)

    @property
    def config_path(self) -> Path:
        return self.data_dir / "config.yaml"


def default_config() -> AppConfig:
    return AppConfig()


def load_environment() -> None:
    load_dotenv()


def resolve_data_dir(data_dir: Path | None = None) -> Path:
    load_environment()
    if data_dir is not None:
        return data_dir
    return Path(os.getenv(DATA_DIR_ENV, ".scribebase"))


def resolve_config_path(config_path: Path | None = None) -> Path:
    load_environment()
    if config_path is not None:
        return config_path
    env_path = os.getenv(CONFIG_ENV)
    if env_path:
        return Path(env_path)
    return resolve_data_dir() / "config.yaml"


def load_config(config_path: Path | None = None) -> AppConfig:
    load_environment()
    path = resolve_config_path(config_path)
    if not path.exists():
        data: dict[str, Any] = {}
    else:
        data = yaml.safe_load(path.read_text()) or {}
    data = deep_update(data, env_override_data())
    return AppConfig.model_validate(data)


def env_override_data() -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if data_dir := os.getenv(DATA_DIR_ENV):
        overrides["data_dir"] = data_dir
    server: dict[str, Any] = {}
    if host := os.getenv(HOST_ENV):
        server["host"] = host
    if port := os.getenv(PORT_ENV):
        server["port"] = port
    if server:
        overrides["server"] = server
    return overrides


def read_api_token(config: AppConfig) -> str | None:
    return os.getenv(config.server.api_token_env)


def config_to_yaml(config: AppConfig) -> str:
    data = config.model_dump(mode="json")
    return yaml.safe_dump(data, sort_keys=False)


def write_default_config(
    data_dir: Path,
    overwrite: bool = False,
    config_path: Path | None = None,
) -> Path:
    config = default_config()
    config.data_dir = data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    path = config_path or data_dir / "config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
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
