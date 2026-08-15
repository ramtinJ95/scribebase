import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "verify_embedding_parity.py"
SPEC = importlib.util.spec_from_file_location("verify_embedding_parity", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
verify_embedding_parity = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verify_embedding_parity)


def test_parity_script_requires_2048_dimensions_by_default(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_embedding_parity.py",
            "--endpoint",
            "http://127.0.0.1:8090/v1",
            "--model-name",
            "Nemotron-3-Embed-1B-BF16",
            "--reference-model",
            "reference",
        ],
    )
    monkeypatch.setattr(
        verify_embedding_parity,
        "embed_server",
        lambda _endpoint, _model, texts: [[1.0, 0.0] for _ in texts],
    )
    monkeypatch.setattr(
        verify_embedding_parity,
        "embed_reference",
        lambda _model, texts: [[1.0, 0.0] for _ in texts],
    )

    assert verify_embedding_parity.main() == 1
