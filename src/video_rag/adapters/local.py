from __future__ import annotations

from video_rag.retrieval.in_memory import tokenize
from video_rag.schemas import GeneratedAnswer, VideoSegment


class TokenOverlapReranker:
    """CPU-only development adapter; replace with Qwen3RerankerAdapter on GPU."""

    supports_confidence = True

    def score(self, query: str, segments: list[VideoSegment]) -> list[float]:
        query_tokens = set(tokenize(query))
        if not query_tokens:
            return [0.0 for _ in segments]
        scores: list[float] = []
        for segment in segments:
            document_tokens = set(tokenize(segment.searchable_text))
            scores.append(len(query_tokens & document_tokens) / len(query_tokens))
        return scores


class FusionOrderReranker:
    """No-model reranker that preserves the order produced by retrieval fusion."""

    supports_confidence = False

    def score(self, query: str, segments: list[VideoSegment]) -> list[float]:
        if not segments:
            return []
        denominator = max(1, len(segments) - 1)
        return [1.0 - index / denominator for index in range(len(segments))]


class EvidenceGenerator:
    """Deterministic local generator that makes the pipeline testable without a GPU."""

    def generate(self, query: str, segments: list[VideoSegment]) -> GeneratedAnswer:
        del query
        if not segments:
            return GeneratedAnswer("根据当前视频内容无法确定。", False)
        evidence = segments[0]
        content = (
            evidence.transcript.strip()
            or evidence.visual_caption.strip()
            or evidence.ocr_text.strip()
        )
        if not content:
            return GeneratedAnswer("根据当前视频内容无法确定。", False)
        return GeneratedAnswer(
            f"{content}（证据：{evidence.segment_id}，{evidence.start_time:.1f}–{evidence.end_time:.1f}秒）",
            True,
            citations=(evidence.segment_id,),
            confidence=1.0,
        )
