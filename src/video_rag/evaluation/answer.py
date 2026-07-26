from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import re

TOKEN_PATTERN = re.compile(r"[\w\u4e00-\u9fff]", re.UNICODE)


def answer_tokens(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def exact_match(prediction: str, reference: str) -> float:
    normalize = lambda value: "".join(answer_tokens(value))
    return float(normalize(prediction) == normalize(reference))


def token_f1(prediction: str, reference: str) -> float:
    predicted = Counter(answer_tokens(prediction))
    expected = Counter(answer_tokens(reference))
    overlap = sum((predicted & expected).values())
    if not predicted or not expected:
        return float(predicted == expected)
    precision = overlap / sum(predicted.values())
    recall = overlap / sum(expected.values())
    return 2 * precision * recall / (precision + recall) if overlap else 0.0


def evaluate_answers(
    predictions: Mapping[str, str],
    references: Mapping[str, str],
) -> dict[str, float]:
    if not references:
        raise ValueError("references must not be empty")
    identifiers = sorted(references)
    return {
        "exact_match": sum(
            exact_match(predictions.get(identifier, ""), references[identifier])
            for identifier in identifiers
        )
        / len(identifiers),
        "token_f1": sum(
            token_f1(predictions.get(identifier, ""), references[identifier])
            for identifier in identifiers
        )
        / len(identifiers),
    }

