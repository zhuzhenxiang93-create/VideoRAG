from __future__ import annotations

from typing import Protocol

from video_rag.schemas import SearchHit, VideoSegment


class Retriever(Protocol):
    name: str

    def build(self, segments: list[VideoSegment]) -> None: ...

    def search(self, query: str, top_k: int) -> list[SearchHit]: ...


class Reranker(Protocol):
    def score(self, query: str, segments: list[VideoSegment]) -> list[float]: ...


class Generator(Protocol):
    def generate(self, query: str, segments: list[VideoSegment]) -> str: ...

