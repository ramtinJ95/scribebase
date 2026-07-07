from scribebase.config import default_config, load_config, write_default_config


def test_config_defaults_are_local_first() -> None:
    config = default_config()
    assert config.weaviate.url == "http://localhost:8081"
    assert config.embedding.base_url == "http://localhost:8080/v1"
    assert config.embedding.model == "Qwen3-Embedding-4B-Q4_K_M.gguf"
    assert config.embedding.batch_size == 8
    assert config.ocr.providers["apple_vision"].render_dpi == 200
    assert not config.llm.enabled


def test_config_round_trip(tmp_path) -> None:
    path = write_default_config(tmp_path)
    loaded = load_config(path)
    assert loaded.data_dir == tmp_path
    assert loaded.ocr.default_provider == "shell"
