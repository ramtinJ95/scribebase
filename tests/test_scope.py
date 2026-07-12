from typer.testing import CliRunner

from scribebase.cli import app
from scribebase.config import default_config
from scribebase.paths import ensure_data_layout


def test_cli_exposes_retrieval_not_generation_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "search" in result.output
    assert "ask" not in result.output
    assert "quiz" not in result.output


def test_default_config_has_no_chat_model_provider() -> None:
    assert not hasattr(default_config(), "llm")


def test_data_layout_does_not_create_generation_output_directories(tmp_path) -> None:
    ensure_data_layout(tmp_path)

    assert not (tmp_path / "outputs").exists()
