from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping

from video_rag.schemas import SearchHit


def reciprocal_rank_fusion(
    result_lists: list[list[SearchHit]],
    *,
    k: int = 60,
    top_k: int | None = None,
    source_weights: Mapping[str, float] | None = None,
    agreement_bonus: float = 0.0,
) -> list[SearchHit]:
    if k <= 0:
        raise ValueError("RRF k must be positive")
    if agreement_bonus < 0:
        raise ValueError("agreement_bonus must be non-negative")
    weights = dict(source_weights or {})
    if any(weight < 0 for weight in weights.values()):
        raise ValueError("RRF source weights must be non-negative")

    scores: dict[str, float] = defaultdict(float)
    sources: dict[str, set[str]] = defaultdict(set)

    for results in result_lists:
        seen: set[str] = set()
        for fallback_rank, hit in enumerate(results, start=1):
            if hit.segment_id in seen:
                continue
            seen.add(hit.segment_id)
            rank = hit.rank if hit.rank > 0 else fallback_rank
            scores[hit.segment_id] += weights.get(hit.source, 1.0) / (k + rank)
            sources[hit.segment_id].add(hit.source)

    if agreement_bonus:
        for segment_id, matched_sources in sources.items():
            scores[segment_id] *= 1.0 + agreement_bonus * (len(matched_sources) - 1)

    fused = [
        SearchHit(
            segment_id=segment_id,
            score=score,
            source="+".join(sorted(sources[segment_id])),
        )
        for segment_id, score in scores.items()
    ]
    fused.sort(key=lambda hit: (-hit.score, hit.segment_id))
    if top_k is not None:
        fused = fused[:top_k]
    return [
        SearchHit(hit.segment_id, hit.score, hit.source, rank)
        for rank, hit in enumerate(fused, start=1)
    ]
