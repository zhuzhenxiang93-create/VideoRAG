from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable

from video_rag.schemas import SearchHit, VideoSegment

TOKEN_PATTERN = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    text = text.lower()
    rough = TOKEN_PATTERN.findall(text)
    tokens: list[str] = []
    for token in rough:
        tokens.append(token)
        if any("\u4e00" <= char <= "\u9fff" for char in token):
            tokens.extend(token[index : index + 2] for index in range(len(token) - 1))
    return tokens


class InMemoryLexicalRetriever:
    """Small BM25-like baseline used for local development and tests."""

    name = "lexical"

    def __init__(self) -> None:
        self._segments: list[VideoSegment] = []
        self._documents: list[Counter[str]] = []
        self._document_frequency: Counter[str] = Counter()

    def build(self, segments: list[VideoSegment]) -> None:
        self._segments = list(segments)
        self._documents = [Counter(tokenize(item.searchable_text)) for item in segments]
        self._document_frequency.clear()
        for document in self._documents:
            self._document_frequency.update(document.keys())

    def search(self, query: str, top_k: int) -> list[SearchHit]:
        query_tokens = tokenize(query)
        if not query_tokens or not self._segments or top_k <= 0:
            return []

        total_documents = len(self._segments)
        scored: list[tuple[str, float]] = []
        for segment, document in zip(self._segments, self._documents, strict=True):
            score = 0.0
            for token in query_tokens:
                frequency = document.get(token, 0)
                if frequency == 0:
                    continue
                document_frequency = self._document_frequency[token]
                inverse_document_frequency = math.log(
                    1.0 + (total_documents - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
                score += inverse_document_frequency * frequency / (frequency + 1.2)
            if score > 0:
                scored.append((segment.segment_id, score))

        scored.sort(key=lambda item: (-item[1], item[0]))
        return [
            SearchHit(segment_id, score, self.name, rank)
            for rank, (segment_id, score) in enumerate(scored[:top_k], start=1)
        ]


class BM25Retriever:
    """In-memory Okapi BM25 retriever with document-length normalization."""

    name = "bm25"

    def __init__(
        self,
        *,
        k1: float = 1.2,
        b: float = 0.75,
        document_text: Callable[[VideoSegment], str] | None = None,
    ) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")
        self.k1 = k1
        self.b = b
        self._document_text = document_text or (lambda segment: segment.searchable_text)
        self._segments: list[VideoSegment] = []
        self._documents: list[Counter[str]] = []
        self._document_lengths: list[int] = []
        self._average_document_length = 0.0
        self._document_frequency: Counter[str] = Counter()

    def build(self, segments: list[VideoSegment]) -> None:
        self._segments = list(segments)
        self._documents = [Counter(tokenize(self._document_text(item))) for item in segments]
        self._document_lengths = [sum(document.values()) for document in self._documents]
        self._average_document_length = (
            sum(self._document_lengths) / len(self._document_lengths)
            if self._document_lengths
            else 0.0
        )
        self._document_frequency.clear()
        for document in self._documents:
            self._document_frequency.update(document.keys())

    def search(self, query: str, top_k: int) -> list[SearchHit]:
        query_tokens = tokenize(query)
        if not query_tokens or not self._segments or top_k <= 0:
            return []

        total_documents = len(self._segments)
        average_length = self._average_document_length or 1.0
        scored: list[tuple[str, float]] = []
        for segment, document, document_length in zip(
            self._segments,
            self._documents,
            self._document_lengths,
            strict=True,
        ):
            score = 0.0
            for token in query_tokens:
                frequency = document.get(token, 0)
                if frequency == 0:
                    continue
                document_frequency = self._document_frequency[token]
                inverse_document_frequency = math.log(
                    1.0
                    + (total_documents - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
                length_factor = 1.0 - self.b + self.b * document_length / average_length
                score += inverse_document_frequency * (
                    frequency * (self.k1 + 1.0)
                    / (frequency + self.k1 * length_factor)
                )
            if score > 0:
                scored.append((segment.segment_id, score))

        scored.sort(key=lambda item: (-item[1], item[0]))
        return [
            SearchHit(segment_id, score, self.name, rank)
            for rank, (segment_id, score) in enumerate(scored[:top_k], start=1)
        ]


class OCRBM25Retriever(BM25Retriever):
    """Independent sparse retrieval over timestamped on-screen text only."""

    name = "ocr_bm25"

    def __init__(self, *, k1: float = 1.2, b: float = 0.0) -> None:
        super().__init__(k1=k1, b=b, document_text=lambda segment: segment.ocr_text)
