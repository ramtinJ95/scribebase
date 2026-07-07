import pytest

from scribebase.config import (
    API_TOKEN_ENV,
    CONFIG_ENV,
    DATA_DIR_ENV,
    HOST_ENV,
    PORT_ENV,
    default_config,
    load_config,
    read_api_token,
    resolve_config_path,
    resolve_data_dir,
    write_default_config,
)


@pytest.fixture(autouse=True)
def clear_scribebase_env(monkeypatch) -> None:
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    for name in [CONFIG_ENV, DATA_DIR_ENV, HOST_ENV, PORT_ENV, API_TOKEN_ENV]:
        monkeypatch.delenv(name, raising=False)


def test_config_defaults_are_local_first() -> None:
    config = default_config()
    assert config.weaviate.url == "http://localhost:8081"
    assert config.embedding.base_url == "http://localhost:8080/v1"
    assert config.embedding.model == "Qwen3-Embedding-4B-Q4_K_M.gguf"
    assert config.embedding.batch_size == 8
    assert config.ocr.providers["apple_vision"].render_dpi == 200
    assert not config.llm.enabled
    assert config.server.host == "127.0.0.1"
    assert config.server.port == 8765
    assert config.server.api_token_env == API_TOKEN_ENV


def test_config_round_trip(tmp_path) -> None:
    path = write_default_config(tmp_path)
    loaded = load_config(path)
    assert loaded.data_dir == tmp_path
    assert loaded.ocr.default_provider == "shell"


def test_load_config_applies_env_overrides(tmp_path, monkeypatch) -> None:
    path = write_default_config(tmp_path)
    runtime_dir = tmp_path / "runtime"
    monkeypatch.setenv(DATA_DIR_ENV, str(runtime_dir))
    monkeypatch.setenv(HOST_ENV, "0.0.0.0")
    monkeypatch.setenv(PORT_ENV, "9876")

    loaded = load_config(path)

    assert loaded.data_dir == runtime_dir
    assert loaded.server.host == "0.0.0.0"
    assert loaded.server.port == 9876


def test_resolve_paths_honor_environment(tmp_path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    config_path = tmp_path / "config" / "scribebase.yaml"
    monkeypatch.setenv(DATA_DIR_ENV, str(data_dir))
    assert resolve_data_dir() == data_dir
    assert resolve_config_path() == data_dir / "config.yaml"

    monkeypatch.setenv(CONFIG_ENV, str(config_path))
    assert resolve_config_path() == config_path


def test_api_token_is_read_from_configured_env(monkeypatch) -> None:
    monkeypatch.setenv(API_TOKEN_ENV, "secret-token")
    assert read_api_token(default_config()) == "secret-token"
