from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

from scribebase.durable_fs import atomic_write_text, durable_mkdir


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
    command: str
    timeout_seconds: int = 900
    model_name: str | None = None
    base_url: str | None = None
    require_multimodal: bool = False
    render_dpi: int | None = None


def glm_ocr_provider_config() -> OCRProviderConfig:
    return OCRProviderConfig(
        command=(
            "./scripts/run_local_ocr.py --input {input_image} --output {output_md} "
            "--base-url {base_url} --model {model_name}"
        ),
        model_name="GLM-OCR",
        base_url="http://localhost:8082/v1",
        require_multimodal=True,
    )


class OCRConfig(BaseModel):
    default_provider: str = "glm_ocr"
    render_dpi: int = 300
    providers: dict[str, OCRProviderConfig] = Field(
        default_factory=lambda: {
            "glm_ocr": glm_ocr_provider_config(),
            "apple_vision": OCRProviderConfig(
                command=(
                    "swift ./scripts/run_apple_vision_ocr.swift "
                    "--input {input_image} --output {output_md}"
                ),
                timeout_seconds=120,
                model_name="Apple Vision",
                base_url=None,
                require_multimodal=False,
                render_dpi=200,
            ),
        }
    )

    @model_validator(mode="after")
    def _validate_default_provider(self) -> "OCRConfig":
        if self.default_provider not in self.providers:
            raise ValueError(
                f"OCR default_provider is not configured: {self.default_provider}"
            )
        return self


class ChunkingConfig(BaseModel):
    target_chars: int = 1200
    overlap_chars: int = 150
    min_chars: int = 250
    chunker_version: str = "v2"


class RetrievalConfig(BaseModel):
    alpha: float = 0.65
    top_k: int = 12


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8765
    api_token_env: str = API_TOKEN_ENV
    max_upload_bytes: int = Field(default=250 * 1024 * 1024, gt=0)
    max_active_jobs: int = Field(default=20, gt=0)
    max_upload_storage_bytes: int = Field(default=1024 * 1024 * 1024, gt=0)
    worker_poll_seconds: float = Field(default=2.0, gt=0)
    worker_dependency_retry_seconds: float = Field(default=10.0, gt=0)
    worker_heartbeat_seconds: float = Field(default=2.0, gt=0)
    worker_stale_seconds: float = Field(default=15.0, gt=0)
    upload_reservation_timeout_seconds: int = Field(default=60 * 60, gt=0)
    identity_orphan_job_seconds: int = Field(default=5 * 60, ge=0)
    identity_direct_reservation_seconds: int = Field(default=24 * 60 * 60, gt=0)
    identity_reservation_heartbeat_seconds: float = Field(default=60.0, gt=0)
    failed_upload_retention_seconds: int = Field(default=7 * 24 * 60 * 60, ge=0)

    @model_validator(mode="after")
    def _validate_identity_heartbeat(self) -> "ServerConfig":
        if (
            self.identity_direct_reservation_seconds
            <= self.identity_reservation_heartbeat_seconds * 2
        ):
            raise ValueError(
                "identity_direct_reservation_seconds must exceed twice "
                "identity_reservation_heartbeat_seconds"
            )
        return self


class AppConfig(BaseModel):
    data_dir: Path = Path(".scribebase")
    weaviate: WeaviateConfig = Field(default_factory=WeaviateConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    pdf_detection: PDFDetectionConfig = Field(default_factory=PDFDetectionConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)

    @model_validator(mode="before")
    @classmethod
    def _reject_removed_llm_config(cls, data: Any) -> Any:
        if isinstance(data, dict) and "llm" in data:
            raise ValueError(
                "The llm configuration section is no longer supported; "
                "remove it and use a consuming agent for final generation"
            )
        return data

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
    data = migrate_legacy_ocr_config(data)
    data = deep_update(data, env_override_data())
    return AppConfig.model_validate(data)


def migrate_legacy_ocr_config(data: dict[str, Any]) -> dict[str, Any]:
    ocr = data.get("ocr")
    if not isinstance(ocr, dict) or ocr.get("default_provider") not in {
        "shell",
        "apple_vision",
    }:
        return data
    providers = ocr.get("providers")
    if not isinstance(providers, dict):
        providers = {}
        ocr["providers"] = providers
    if ocr["default_provider"] == "shell":
        legacy = providers.get("shell")
        if not _is_generated_legacy_glm_provider(legacy):
            raise ValueError(
                "Legacy OCR default_provider 'shell' is not a generated GLM-OCR "
                "configuration. Rename the custom provider and select it explicitly, or "
                "set default_provider to glm_ocr."
            )
        providers.pop("shell", None)
    providers.setdefault(
        "glm_ocr",
        glm_ocr_provider_config().model_dump(mode="json"),
    )
    ocr["default_provider"] = "glm_ocr"
    return data


def _is_generated_legacy_glm_provider(provider: Any) -> bool:
    if not isinstance(provider, dict):
        return False
    command = str(provider.get("command", ""))
    return (
        "scripts/run_local_ocr.py" in command
        and provider.get("model_name", "GLM-OCR") == "GLM-OCR"
    )


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
    durable_mkdir(data_dir)
    path = config_path or data_dir / "config.yaml"
    durable_mkdir(path.parent)
    if path.exists() and not overwrite:
        return path
    atomic_write_text(path, config_to_yaml(config))
    return path


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = deep_update(base[key], value)
        else:
            base[key] = value
    return base
