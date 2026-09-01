from __future__ import annotations

from collections.abc import Iterable, Mapping
from time import perf_counter

from video_rag.retrieval.base import Generator, Reranker, Retriever
from video_rag.retrieval.fusion import reciprocal_rank_fusion
from video_rag.retrieval.routing import AdaptiveFusionPolicy
from video_rag.schemas import Answer, Evidence, VideoSegment


class VideoRAGPipeline:
    def __init__(
        self,
        *,
        retrievers: list[Retriever],
        reranker: Reranker,
        generator: Generator,
        recall_top_k: int | Mapping[str, int] = 20,
        fusion_top_k: int = 20,
        rerank_top_k: int = 3,
        rrf_k: int = 60,
        minimum_rerank_score: float = 0.20,
        allow_abstention: bool = True,
        fusion_policy: AdaptiveFusionPolicy | None = None,
        reranker_weight: float = 1.0,
    ) -> None:
        if not retrievers:
            raise ValueError("At least one retriever is required")
        retriever_names = [retriever.name for retriever in retrievers]
        if len(retriever_names) != len(set(retriever_names)):
            raise ValueError("Retriever names must be unique")
        if isinstance(recall_top_k, int):
            recall_top_k_by_name = {name: recall_top_k for name in retriever_names}
        else:
            missing = set(retriever_names) - set(recall_top_k)
            if missing:
                raise ValueError(f"Missing recall_top_k values for: {sorted(missing)}")
            recall_top_k_by_name = {
                name: int(recall_top_k[name]) for name in retriever_names
            }
        if min(*recall_top_k_by_name.values(), fusion_top_k, rerank_top_k, rrf_k) <= 0:
            raise ValueError("Top-k and RRF k values must be positive")
        if not 0 <= reranker_weight <= 1:
            raise ValueError("reranker_weight must be between 0 and 1")
        self._retrievers = retrievers
        self._reranker = reranker
        self._generator = generator
        self._recall_top_k_by_name = recall_top_k_by_name
        self._fusion_top_k = fusion_top_k
        self._rerank_top_k = rerank_top_k
        self._rrf_k = rrf_k
        self._minimum_rerank_score = minimum_rerank_score
        self._allow_abstention = allow_abstention
        self._fusion_policy = fusion_policy
        self._reranker_weight = reranker_weight
        self._segments: dict[str, VideoSegment] = {}

    def build(self, segments: Iterable[VideoSegment]) -> None:
        materialized = list(segments)
        identifiers = [segment.segment_id for segment in materialized]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("segment_id must be unique")
        self._segments = {segment.segment_id: segment for segment in materialized}
        for retriever in self._retrievers:
            retriever.build(materialized)

    def warmup(self, query: str = "video content") -> None:
        """Load retrieval models and execute one minimal query before serving traffic."""
        if not self._segments:
            raise RuntimeError("Pipeline index is empty; call build() first")
        for retriever in self._retrievers:
            retriever.search(query, 1)

    def ask(self, query: str) -> Answer:
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        if not self._segments:
            raise RuntimeError("Pipeline index is empty; call build() first")

        started = perf_counter()
        result_lists = [
            retriever.search(query, self._recall_top_k_by_name[retriever.name])
            for retriever in self._retrievers
        ]
        recalled_at = perf_counter()
        fused = reciprocal_rank_fusion(
            result_lists,
            k=self._rrf_k,
            top_k=self._fusion_top_k,
            source_weights=(
                self._fusion_policy.source_weights(query)
                if self._fusion_policy is not None
                else None
            ),
            agreement_bonus=(
                self._fusion_policy.agreement_bonus
                if self._fusion_policy is not None
                else 0.0
            ),
        )
        candidates = [
            self._segments[hit.segment_id]
            for hit in fused
            if hit.segment_id in self._segments
        ]
        fused_score = {hit.segment_id: hit.score for hit in fused}
        fused_at = perf_counter()

        if not candidates:
            return self._abstention(started, recalled_at, fused_at)

        rerank_scores = self._reranker.score(query, candidates)
        if len(rerank_scores) != len(candidates):
            raise ValueError("Reranker must return one score per candidate")
        fused_rank_score = {
            segment.segment_id: 1.0 - rank / max(1, len(candidates) - 1)
            if len(candidates) > 1
            else 1.0
            for rank, segment in enumerate(candidates)
        }
        blended = [
            self._reranker_weight * score
            + (1.0 - self._reranker_weight) * fused_rank_score[segment.segment_id]
            for segment, score in zip(candidates, rerank_scores, strict=True)
        ]
        ranked = sorted(
            zip(candidates, rerank_scores, blended, strict=True),
            key=lambda item: (-item[2], -fused_score[item[0].segment_id]),
        )[: self._rerank_top_k]
        reranked_at = perf_counter()

        if not ranked or (
            self._allow_abstention
            and ranked[0][1] < self._minimum_rerank_score
        ):
            return self._abstention(started, recalled_at, fused_at, reranked_at)

        selected_segments = [item[0] for item in ranked]
        answer_text = self._generator.generate(query, selected_segments)
        finished = perf_counter()
        evidence = tuple(
            Evidence(
                segment=segment,
                fused_score=fused_score[segment.segment_id],
                rerank_score=score,
            )
            for segment, score, _ in ranked
        )
        return Answer(
            answer=answer_text,
            evidence=evidence,
            latency_ms={
                "recall": (recalled_at - started) * 1000,
                "fusion": (fused_at - recalled_at) * 1000,
                "rerank": (reranked_at - fused_at) * 1000,
                "generation": (finished - reranked_at) * 1000,
                "total": (finished - started) * 1000,
            },
        )

    def video_path(self, video_id: str) -> str | None:
        for segment in self._segments.values():
            if segment.video_id == video_id:
                return segment.source_path
        return None

    def _abstention(
        self,
        started: float,
        recalled_at: float,
        fused_at: float,
        reranked_at: float | None = None,
    ) -> Answer:
        finished = perf_counter()
        return Answer(
            answer="根据当前视频内容无法确定。",
            evidence=(),
            abstained=True,
            latency_ms={
                "recall": (recalled_at - started) * 1000,
                "fusion": (fused_at - recalled_at) * 1000,
                "rerank": 0.0 if reranked_at is None else (reranked_at - fused_at) * 1000,
                "generation": 0.0,
                "total": (finished - started) * 1000,
            },
        )
