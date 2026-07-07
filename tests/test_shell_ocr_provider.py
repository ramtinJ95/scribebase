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
