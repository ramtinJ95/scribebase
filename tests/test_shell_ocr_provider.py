from scribebase.config import OCRProviderConfig
from scribebase.ocr.shell_provider import ShellOCRProvider


def test_shell_ocr_provider_formats_command(tmp_path) -> None:
    provider = ShellOCRProvider(
        OCRProviderConfig(command="ocr --in {input_image} --out {output_md} --page {page_number}")
    )
    cmd = provider.format_command(
        tmp_path / "page.png", tmp_path / "page.md", {"page_number": 7, "source_id": "src"}
    )
    assert cmd == ["ocr", "--in", str(tmp_path / "page.png"), "--out", str(tmp_path / "page.md"), "--page", "7"]


def test_shell_ocr_provider_can_use_configured_name() -> None:
    provider = ShellOCRProvider(OCRProviderConfig(), name="apple_vision")
    assert provider.name == "apple_vision"


def test_glm_ocr_command_uses_configured_server_and_model(tmp_path) -> None:
    provider = ShellOCRProvider(OCRProviderConfig(), name="glm_ocr")

    cmd = provider.format_command(tmp_path / "page.png", tmp_path / "page.md", {})

    assert cmd[-4:] == ["--base-url", "http://localhost:8082/v1", "--model", "GLM-OCR"]
