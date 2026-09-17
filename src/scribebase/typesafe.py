"""Small, validated HTTP boundary for hosted TypeSafe judgments."""
from __future__ import annotations

import math
import os
import time
from typing import Any

import httpx

from scribebase.config import TypeSafeConfig


class TypeSafeError(RuntimeError):
    pass


def _number(value: Any, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("expected a number")
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError("number outside permitted range")
    return float(value)


def validate_answers(answers: Any, questions: dict) -> dict:
    try:
        if not isinstance(answers, dict) or answers.keys() != questions.keys():
            raise ValueError("answer IDs do not match question IDs")
        for key, question in questions.items():
            answer = answers[key]
            kind = question["type"]
            if answer["type"] != kind:
                raise ValueError("answer type mismatch")
            if kind == "noul":
                _number(answer["noul"], 0, 1)
                continue
            criteria = question["criteria"]
            options = set(criteria) if kind == "choice" else {str(i) for i in range(len(criteria))}
            probabilities = answer["probabilities"]
            if set(probabilities) != options:
                raise ValueError("probability options mismatch")
            values = [_number(value, 0, 1) for value in probabilities.values()]
            if not math.isclose(sum(values), 1, abs_tol=0.01):
                raise ValueError("probabilities do not sum to one")
            _number(answer["confidence"], 0, 1)
            if kind == "choice":
                if answer["choice"] not in options:
                    raise ValueError("unknown choice")
                if probabilities[answer["choice"]] < max(values):
                    raise ValueError("choice is not the most probable option")
            elif kind == "score":
                score = _number(answer["score"], 0, len(criteria) - 1)
                if answer["legend"] != {str(i): value for i, value in enumerate(criteria)}:
                    raise ValueError("score legend mismatch")
                expected = sum(int(i) * p for i, p in probabilities.items())
                if not math.isclose(score, expected, abs_tol=0.02):
                    raise ValueError("score does not match probabilities")
            else:
                raise ValueError("unsupported question type")
    except (KeyError, TypeError, ValueError) as exc:
        raise TypeSafeError(f"Invalid TypeSafe response: {exc}") from exc
    return answers


def evaluate(state: Any, questions: dict, config: TypeSafeConfig) -> dict:
    key = os.getenv(config.api_key_env)
    if not key:
        raise TypeSafeError(f"TypeSafe is enabled but {config.api_key_env} is missing")
    with httpx.Client(timeout=config.timeout_seconds) as client:
        for attempt in range(3):
            try:
                response = client.post(
                    "https://api.typesafe.ai/v1/systemone",
                    headers={"Authorization": f"Bearer {key}"},
                    json={"model": config.model, "state": state, "questions": questions},
                )
            except httpx.TransportError as exc:
                raise TypeSafeError("TypeSafe transport failed") from exc
            if response.status_code in {429, 529} and attempt < 2:
                time.sleep(2 ** attempt)
                continue
            if response.is_error:
                # Never include response bodies (which may echo private state or secrets).
                raise TypeSafeError(f"TypeSafe request failed (HTTP {response.status_code})")
            try:
                payload = response.json()
                if not isinstance(payload.get("model"), str) or not payload["model"]:
                    raise ValueError("missing model")
                validate_answers(payload["answers"], questions)
            except (ValueError, KeyError, AttributeError) as exc:
                raise TypeSafeError("Invalid TypeSafe response envelope") from exc
            return payload
    raise TypeSafeError("TypeSafe retries exhausted")
