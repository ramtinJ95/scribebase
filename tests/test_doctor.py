from __future__ import annotations

import pytest
from typer.testing import CliRunner

from scribebase.cli import app
from scribebase.config import default_config


@pytest.mark.parametrize(("ocr_ok", "exit_code"), [(True, 0), (False, 1)])
def test_doctor_exit_status_reflects_ocr_readiness(
    tmp_path,
    monkeypatch,
    ocr_ok,
    exit_code,
) -> None:  # noqa: ANN001
    config = default_config()
    config.data_dir = tmp_path

    class ReadyStore:
        def __init__(self, _config) -> None:  # noqa: ANN001
            pass

        def is_ready(self) -> bool:
            return True

        def close(self) -> None:
            pass

    class ReadyEmbeddings:
        def __init__(self, _config) -> None:  # noqa: ANN001
            pass

        def check_health(self) -> tuple[bool, str]:
            return True, "ready"

    monkeypatch.setattr("scribebase.cli._config", lambda: config)
    monkeypatch.setattr("scribebase.cli.importlib.util.find_spec", lambda _name: object())
    monkeypatch.setattr(
        "scribebase.vectorstores.weaviate_store.WeaviateStore", ReadyStore
    )
    monkeypatch.setattr(
        "scribebase.embeddings.llamacpp_client.LlamaCppEmbeddingClient", ReadyEmbeddings
    )
    monkeypatch.setattr(
        "scribebase.cli.check_ocr_provider_health",
        lambda *_: (ocr_ok, "ready" if ocr_ok else "GLM-OCR unavailable; no fallback"),
    )

    result = CliRunner().invoke(app, ["doctor"])

    assert result.exit_code == exit_code
    assert f"[{'OK' if ocr_ok else 'FAIL'}] OCR provider: glm_ocr" in result.output
