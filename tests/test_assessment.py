import pytest

from scribebase.config import TypeSafeConfig
from scribebase.models import Chunk, SearchResult
from scribebase.retrieval import assessment
from scribebase.typesafe import TypeSafeError


def result(index):
    return SearchResult(score=1 / (index + 1), chunk=Chunk(
        chunk_id=str(index), source_id="source", source_type="book", title="Title",
        chunk_index=index, text=f"Passage {index}", file_path="document.md", extraction_method="text",
    ))


def test_reranking_batches_preserves_scores_and_ties(monkeypatch):
    calls = []
    def fake(state, questions, config):
        calls.append(state)
        return {"model": "test", "answers": {key: {
            "score": 2, "confidence": 0.8,
        } for key in questions}}
    monkeypatch.setattr(assessment, "evaluate", fake)
    original = [result(i) for i in range(3)]
    ranked = assessment.assess_passages("question", original, TypeSafeConfig(reranking_enabled=True, batch_size=2))
    assert len(calls) == 2
    assert [r.chunk.chunk_id for r in ranked] == ["0", "1", "2"]
    assert [r.score for r in ranked] == [r.score for r in original]
    assert all(r.assessment is None for r in original)
    assert ranked[0].assessment.raw_answers["relevance"]["score"] == 2


def test_reranks_by_relevance(monkeypatch):
    monkeypatch.setattr(assessment, "evaluate", lambda *args: {
        "model": "test", "answers": {f"relevance_{i}": {"score": i, "confidence": 1} for i in range(3)}})
    ranked = assessment.assess_passages("question", [result(i) for i in range(3)], TypeSafeConfig(reranking_enabled=True))
    assert [r.chunk.chunk_id for r in ranked] == ["2", "1", "0"]


def test_disabled_and_empty_do_not_call(monkeypatch):
    def fail(*args):
        pytest.fail("unexpected call")
    monkeypatch.setattr(assessment, "evaluate", fail)
    original = [result(0)]
    assert assessment.assess_passages("q", original, TypeSafeConfig()) is original
    assert assessment.assess_passages("q", [], TypeSafeConfig(reranking_enabled=True)) == []


def test_service_failure_does_not_fallback(monkeypatch):
    def fail(*args):
        raise TypeSafeError("unavailable")
    monkeypatch.setattr(assessment, "evaluate", fail)
    with pytest.raises(TypeSafeError):
        assessment.assess_passages("q", [result(0)], TypeSafeConfig(reranking_enabled=True))


def test_truncation_visible_and_source_unchanged(monkeypatch):
    original = result(0)
    original.chunk.text = "x" * 700
    def fake(state, questions, config):
        assert len(state["passages"]["0"]["text"]) == 500
        return {"model": "test", "answers": {"relevance_0": {"score": 1, "confidence": 1}}}
    monkeypatch.setattr(assessment, "evaluate", fake)
    ranked = assessment.assess_passages("q", [original], TypeSafeConfig(reranking_enabled=True, passage_max_chars=500))
    assert ranked[0].assessment.text_truncated
    assert ranked[0].chunk.text == original.chunk.text
