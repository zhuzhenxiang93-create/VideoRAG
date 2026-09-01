from __future__ import annotations

from collections.abc import Iterable, Mapping
from time import perf_counter

from video_rag.retrieval.base import Generator, Reranker, Retriever
from video_rag.retrieval.fusion import reciprocal_rank_fusion
from video_rag.retrieval.routing import AdaptiveFusionPolicy
from video_rag.schemas import Answer, Evidence, GeneratedAnswer, VideoSegment


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
        minimum_generator_confidence: float = 0.20,
        allow_abstention: bool = True,
        require_citations: bool = True,
        fusion_policy: AdaptiveFusionPolicy | None = None,
        reranker_weight: float = 1.0,
        neighbor_hops: int = 1,
        temporal_neighbor_hops: int = 2,
        dedupe_overlap_ratio: float = 0.20,
        max_generation_segments: int = 8,
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
        if not 0 <= minimum_generator_confidence <= 1:
            raise ValueError("minimum_generator_confidence must be between 0 and 1")
        if min(neighbor_hops, temporal_neighbor_hops) < 0:
            raise ValueError("neighbor hops must be non-negative")
        if not 0 <= dedupe_overlap_ratio <= 1:
            raise ValueError("dedupe_overlap_ratio must be between 0 and 1")
        if max_generation_segments <= 0:
            raise ValueError("max_generation_segments must be positive")
        self._retrievers = retrievers
        self._reranker = reranker
        self._generator = generator
        self._recall_top_k_by_name = recall_top_k_by_name
        self._fusion_top_k = fusion_top_k
        self._rerank_top_k = rerank_top_k
        self._rrf_k = rrf_k
        self._minimum_rerank_score = minimum_rerank_score
        self._minimum_generator_confidence = minimum_generator_confidence
        self._allow_abstention = allow_abstention
        self._require_citations = require_citations
        self._fusion_policy = fusion_policy
        self._reranker_weight = reranker_weight
        self._neighbor_hops = neighbor_hops
        self._temporal_neighbor_hops = temporal_neighbor_hops
        self._dedupe_overlap_ratio = dedupe_overlap_ratio
        self._max_generation_segments = max_generation_segments
        self._segments: dict[str, VideoSegment] = {}
        self._segments_by_video: dict[str, list[VideoSegment]] = {}

    def build(self, segments: Iterable[VideoSegment]) -> None:
        materialized = list(segments)
        identifiers = [segment.segment_id for segment in materialized]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("segment_id must be unique")
        self._segments = {segment.segment_id: segment for segment in materialized}
        self._segments_by_video = {}
        for segment in materialized:
            self._segments_by_video.setdefault(segment.video_id, []).append(segment)
        for video_segments in self._segments_by_video.values():
            video_segments.sort(key=lambda item: (item.start_time, item.end_time, item.segment_id))
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
        decision = self._fusion_policy.decision(query) if self._fusion_policy else None
        source_weights = decision.source_weights if decision else None
        route_labels = decision.labels if decision else ("text",)
        result_lists = [
            retriever.search(query, self._recall_top_k_by_name[retriever.name])
            for retriever in self._retrievers
            if source_weights is None or source_weights.get(retriever.name, 1.0) > 0
        ]
        recalled_at = perf_counter()
        fused = reciprocal_rank_fusion(
            result_lists,
            k=self._rrf_k,
            top_k=self._fusion_top_k,
            source_weights=source_weights,
            agreement_bonus=(
                self._fusion_policy.agreement_bonus
                if self._fusion_policy is not None
                else 0.0
            ),
        )
        candidates = self._deduplicate_candidates([
            self._segments[hit.segment_id]
            for hit in fused
            if hit.segment_id in self._segments
        ])
        fused_score = {hit.segment_id: hit.score for hit in fused}
        fused_at = perf_counter()

        if not candidates:
            return self._abstention(
                started, recalled_at, fused_at, route_labels=route_labels
            )

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

        supports_confidence = bool(getattr(self._reranker, "supports_confidence", False))
        if not ranked or (
            self._allow_abstention
            and supports_confidence
            and ranked[0][1] < self._minimum_rerank_score
        ):
            return self._abstention(
                started, recalled_at, fused_at, reranked_at, route_labels=route_labels
            )

        selected_segments = [item[0] for item in ranked]
        generation_segments = self._expand_neighbors(selected_segments, route_labels)
        generated = self._coerce_generated_answer(
            self._generator.generate(query, generation_segments), generation_segments
        )
        finished = perf_counter()
        allowed_ids = {segment.segment_id for segment in generation_segments}
        unique_citations = tuple(dict.fromkeys(generated.citations))
        invalid_citations = set(unique_citations) - allowed_ids
        citations = tuple(value for value in unique_citations if value in allowed_ids)
        insufficient = (
            not generated.answerable
            or bool(invalid_citations)
            or (self._require_citations and not citations)
            or (
                generated.confidence is not None
                and generated.confidence < self._minimum_generator_confidence
            )
        )
        if self._allow_abstention and insufficient:
            return self._abstention(
                started,
                recalled_at,
                fused_at,
                reranked_at,
                route_labels=route_labels,
                confidence=generated.confidence,
                finished=finished,
            )

        rerank_score = {segment.segment_id: score for segment, score, _ in ranked}
        evidence_ids = citations or tuple(segment.segment_id for segment in selected_segments)
        evidence = tuple(
            Evidence(
                segment=self._segments[segment_id],
                fused_score=fused_score.get(segment_id, 0.0),
                rerank_score=rerank_score.get(segment_id, 0.0),
            )
            for segment_id in evidence_ids
            if segment_id in self._segments
        )
        return Answer(
            answer=generated.answer,
            evidence=evidence,
            citations=citations,
            confidence=generated.confidence,
            route_labels=route_labels,
            latency_ms={
                "recall": (recalled_at - started) * 1000,
                "fusion": (fused_at - recalled_at) * 1000,
                "rerank": (reranked_at - fused_at) * 1000,
                "generation": (finished - reranked_at) * 1000,
                "total": (finished - started) * 1000,
            },
        )

    @staticmethod
    def _temporal_overlap_ratio(first: VideoSegment, second: VideoSegment) -> float:
        if first.video_id != second.video_id:
            return 0.0
        overlap = max(
            0.0,
            min(first.end_time, second.end_time) - max(first.start_time, second.start_time),
        )
        shortest = min(
            first.end_time - first.start_time,
            second.end_time - second.start_time,
        )
        return overlap / shortest if shortest > 0 else 0.0

    def _deduplicate_candidates(
        self, candidates: list[VideoSegment]
    ) -> list[VideoSegment]:
        if self._dedupe_overlap_ratio <= 0:
            return candidates
        selected: list[VideoSegment] = []
        for candidate in candidates:
            if any(
                self._temporal_overlap_ratio(candidate, existing)
                >= self._dedupe_overlap_ratio
                for existing in selected
            ):
                continue
            selected.append(candidate)
        return selected

    def _expand_neighbors(
        self, anchors: list[VideoSegment], route_labels: tuple[str, ...]
    ) -> list[VideoSegment]:
        hops = (
            self._temporal_neighbor_hops
            if "temporal" in route_labels
            else self._neighbor_hops
        )
        selected: dict[str, VideoSegment] = {item.segment_id: item for item in anchors}
        for anchor in anchors:
            video_segments = self._segments_by_video.get(anchor.video_id, [])
            anchor_index = next(
                index
                for index, item in enumerate(video_segments)
                if item.segment_id == anchor.segment_id
            )
            lower = max(0, anchor_index - hops)
            upper = min(len(video_segments), anchor_index + hops + 1)
            for segment in video_segments[lower:upper]:
                selected.setdefault(segment.segment_id, segment)
        return sorted(
            selected.values(),
            key=lambda item: (item.video_id, item.start_time, item.end_time),
        )[: self._max_generation_segments]

    @staticmethod
    def _coerce_generated_answer(
        value: GeneratedAnswer | str, segments: list[VideoSegment]
    ) -> GeneratedAnswer:
        if isinstance(value, GeneratedAnswer):
            return value
        citations = tuple(
            segment.segment_id for segment in segments if segment.segment_id in value
        )
        return GeneratedAnswer(value, bool(value.strip()), citations=citations)

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
        *,
        route_labels: tuple[str, ...] = (),
        confidence: float | None = None,
        finished: float | None = None,
    ) -> Answer:
        finished = finished or perf_counter()
        return Answer(
            answer="根据当前视频内容无法确定。",
            evidence=(),
            abstained=True,
            confidence=confidence,
            route_labels=route_labels,
            latency_ms={
                "recall": (recalled_at - started) * 1000,
                "fusion": (fused_at - recalled_at) * 1000,
                "rerank": 0.0 if reranked_at is None else (reranked_at - fused_at) * 1000,
                "generation": (
                    0.0
                    if reranked_at is None
                    else max(0.0, (finished - reranked_at) * 1000)
                ),
                "total": (finished - started) * 1000,
            },
        )
