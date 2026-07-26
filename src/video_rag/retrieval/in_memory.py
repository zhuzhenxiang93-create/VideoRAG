from __future__ import annotations

from collections import Counter
import math
import re

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

