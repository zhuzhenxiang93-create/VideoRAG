from __future__ import annotations

from collections import defaultdict

from video_rag.schemas import SearchHit


def reciprocal_rank_fusion(
    result_lists: list[list[SearchHit]],
    *,
    k: int = 60,
    top_k: int | None = None,
) -> list[SearchHit]:
    if k <= 0:
        raise ValueError("RRF k must be positive")

    scores: dict[str, float] = defaultdict(float)
    sources: dict[str, set[str]] = defaultdict(set)

    for results in result_lists:
        seen: set[str] = set()
        for fallback_rank, hit in enumerate(results, start=1):
            if hit.segment_id in seen:
                continue
            seen.add(hit.segment_id)
            rank = hit.rank if hit.rank > 0 else fallback_rank
            scores[hit.segment_id] += 1.0 / (k + rank)
            sources[hit.segment_id].add(hit.source)

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

