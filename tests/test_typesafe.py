import json

import httpx
import pytest

from scribebase.config import TypeSafeConfig
from scribebase.typesafe import TypeSafeError, evaluate, validate_answers
from scribebase.chunking import structure
from scribebase.chunking.chunker import _units


def choice(value="chapter", confidence=0.95):
    return {"type": "choice", "choice": value, "confidence": confidence,
            "probabilities": {k: float(k == value) for k in structure.CRITERIA}}


def test_http_contract(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-secret")
    def post(self, url, **kwargs):
        assert url == "https://api.typesafe.ai/v1/systemone"
        assert kwargs["headers"]["Authorization"] == "Bearer test-secret"
        assert kwargs["json"]["model"] == "jev-latest"
        return httpx.Response(200, json={"model": "jev-1", "answers": {"role": choice()}})
    monkeypatch.setattr(httpx.Client, "post", post)
    assert evaluate("text", structure.QUESTIONS, TypeSafeConfig())["model"] == "jev-1"


def test_missing_key(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(TypeSafeError, match="missing"):
        evaluate("text", structure.QUESTIONS, TypeSafeConfig())


@pytest.mark.parametrize("answer", [
    {}, {"type": "score"}, choice("invalid"),
    {**choice(), "confidence": float("nan")},
    {**choice(), "probabilities": {"chapter": 0.5, "section": 0, "other": 0}},
    {**choice(), "choice": "other"},
])
def test_invalid_answers(answer):
    with pytest.raises(TypeSafeError):
        validate_answers({"role": answer}, structure.QUESTIONS)


def test_structure_persisted_and_content_addressed(tmp_path, monkeypatch):
    path = tmp_path / "document.md"
    original = "<!-- page: 1 -->\n\nThe control plane\n\n" + "Body text. " * 30
    path.write_text(original)
    calls = []
    def fake(state, questions, config):
        calls.append(state)
        return {"model": "test", "answers": {key: choice() for key in questions}}
    monkeypatch.setattr(structure, "evaluate", fake)
    config = TypeSafeConfig(structure_enabled=True)
    assert structure.recover_structure(path, config) == {1: "chapter"}
    assert structure.recover_structure(path, config) == {1: "chapter"}
    assert len(calls) == 1
    assert path.read_text() == original
    units = _units(original, {1: "chapter"})
    assert units[-1][3] == "The control plane"
    path.write_text(original + "changed")
    structure.recover_structure(path, config)
    assert len(calls) == 2


def test_uncertain_structure_and_corrupt_cache(tmp_path, monkeypatch):
    path = tmp_path / "document.md"
    path.write_text("Ambiguous title")
    monkeypatch.setattr(structure, "evaluate", lambda *args: {
        "model": "test", "answers": {"0": choice(confidence=0.3)}})
    assert structure.recover_structure(path, TypeSafeConfig()) == {}
    cache = next(tmp_path.glob("*.json"))
    record = json.loads(cache.read_text())
    record["answers"] = {}
    cache.write_text(json.dumps(record))
    with pytest.raises(TypeSafeError, match="annotations"):
        structure.recover_structure(path, TypeSafeConfig())


def test_explicit_chapter_precedes_recovered_role():
    assert _units("# Chapter 2: Control\n\nBody", {0: "section"})[-1][3] == "Chapter 2: Control"


def test_code_fences_not_candidates(tmp_path, monkeypatch):
    path = tmp_path / "document.md"
    path.write_text("```python\n\nFake heading\n\n```\n\nActual heading")
    def fake(state, questions, config):
        assert set(state) == {"3"}
        return {"model": "test", "answers": {"3": choice()}}
    monkeypatch.setattr(structure, "evaluate", fake)
    assert structure.recover_structure(path, TypeSafeConfig()) == {3: "chapter"}


@pytest.mark.parametrize("score, probabilities", [
    (2.61, {"0": 0.0, "1": 0.01, "2": 0.35, "3": 0.64}),
    (1.5, {"0": 0.26, "1": 0.25, "2": 0.25, "3": 0.26}),
])
def test_independently_rounded_score_distribution(score, probabilities):
    criteria = ["none", "background", "partial", "direct"]
    questions = {"relevance": {"type": "score", "criteria": criteria}}
    answers = {"relevance": {
        "type": "score", "score": score, "confidence": 0.61,
        "legend": dict(enumerate(criteria)), "probabilities": probabilities,
    }}
    answers["relevance"]["legend"] = {str(i): value for i, value in enumerate(criteria)}
    assert validate_answers(answers, questions) == answers
    answers["relevance"]["score"] = 0.5
    with pytest.raises(TypeSafeError, match="score does not match"):
        validate_answers(answers, questions)
