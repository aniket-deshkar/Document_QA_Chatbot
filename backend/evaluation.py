"""Lightweight quality metrics for generated answers."""

import re
from typing import Iterable


def _tokenize(text: str) -> list[str]:
    """Split text into lowercase alphanumeric tokens."""
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _token_set(text: str) -> set[str]:
    """Build a de-duplicated token set while dropping very short tokens."""
    return {t for t in _tokenize(text) if len(t) > 2}


def _safe_div(n: float, d: float) -> float:
    """Safely divide two numbers and return 0 when denominator is not positive."""
    if d <= 0:
        return 0.0
    return n / d


def _clamp01(value: float) -> float:
    """Clamp a numeric score to the [0, 1] range."""
    return max(0.0, min(1.0, float(value)))


def answer_relevancy(query: str, response: str) -> float:
    """Estimate how well the response overlaps with query intent tokens."""
    q = _token_set(query)
    r = _token_set(response)
    if not q or not r:
        return 0.0
    overlap = len(q & r)
    return _clamp01(_safe_div(overlap, len(q)))


def faithfulness(response: str, context: str) -> float:
    """Estimate how much of the response is supported by retrieved context."""
    context_tokens = _token_set(context)
    if not context_tokens:
        return 0.0

    sentences = [s.strip() for s in re.split(r"[.!?\n]+", response or "") if s.strip()]
    if not sentences:
        return 0.0

    supported = 0
    for sent in sentences:
        sent_tokens = _token_set(sent)
        if not sent_tokens:
            continue
        overlap = len(sent_tokens & context_tokens)
        precision = _safe_div(overlap, len(sent_tokens))
        if precision >= 0.35:
            supported += 1
    return _clamp01(_safe_div(supported, len(sentences)))


def contextual_precision(response: str, context: str) -> float:
    """Estimate the fraction of response tokens grounded in context."""
    response_tokens = _token_set(response)
    if not response_tokens:
        return 0.0
    context_tokens = _token_set(context)
    if not context_tokens:
        return 0.0
    return _clamp01(_safe_div(len(response_tokens & context_tokens), len(response_tokens)))


def evaluate_metrics(
    query: str,
    response: str,
    context: str,
    error_text: str | None = None,
):
    """Compute answer quality metrics and return normalized scoring fields."""
    if error_text:
        return {
            "answer_relevancy": 0.0,
            "faithfulness": 0.0,
            "contextual_precision": 0.0,
            "overall": 0.0,
        }

    lower = (response or "").lower()
    if "i do not know based on the uploaded document" in lower:
        return {
            "answer_relevancy": 0.5,
            "faithfulness": 1.0,
            "contextual_precision": 1.0,
            "overall": 0.8,
        }

    ar = answer_relevancy(query, response)
    fa = faithfulness(response, context)
    cp = contextual_precision(response, context)
    overall = _clamp01((0.4 * ar) + (0.35 * fa) + (0.25 * cp))
    return {
        "answer_relevancy": ar,
        "faithfulness": fa,
        "contextual_precision": cp,
        "overall": overall,
    }
