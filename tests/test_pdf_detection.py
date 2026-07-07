from scribebase.config import PDFDetectionConfig
from scribebase.pdf_router import evaluate_text_quality


def test_real_text_is_accepted() -> None:
    text = "This is a normal textbook paragraph about working memory. " * 20
    result = evaluate_text_quality(text, PDFDetectionConfig(min_chars_per_page=100))
    assert result.is_true_text
    assert result.word_count > 20


def test_empty_text_is_rejected() -> None:
    result = evaluate_text_quality("", PDFDetectionConfig(min_chars_per_page=10))
    assert not result.is_true_text
    assert "too_few_chars" in result.flags


def test_symbol_garbage_is_rejected() -> None:
    result = evaluate_text_quality("@@@@ #### !!!! **** " * 30, PDFDetectionConfig(min_chars_per_page=20))
    assert not result.is_true_text
    assert "low_alpha_ratio" in result.flags
