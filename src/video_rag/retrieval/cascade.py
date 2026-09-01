from __future__ import annotations

from collections.abc import Mapping, Sequence

from video_rag.schemas import SearchHit


def requires_fallback(
    primary_results: Sequence[SearchHit],
    *,
    source: str,
    minimum_scores: Mapping[str, float] | None = None,
    route_confidence: float = 1.0,
    minimum_route_confidence: float = 0.0,
    minimum_score_margin: float = 0.0,
) -> bool:
    """Return whether a single primary route needs its configured fallback."""
    if not primary_results:
        return True
    threshold = (minimum_scores or {}).get(source, float("-inf"))
    if primary_results[0].score < threshold:
        return True
    if route_confidence < minimum_route_confidence:
        return True
    if minimum_score_margin > 0 and len(primary_results) > 1:
        top_score = primary_results[0].score
        denominator = max(abs(top_score), 1e-12)
        relative_margin = (top_score - primary_results[1].score) / denominator
        if relative_margin < minimum_score_margin:
            return True
    return False


def ordered_candidate_union(
    result_lists: Sequence[Sequence[SearchHit]],
    *,
    top_k: int,
    interleave: bool,
) -> list[SearchHit]:
    """Deduplicate candidate lists without mixing incomparable retrieval scores."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    ordered: list[SearchHit] = []
    seen: set[str] = set()
    if interleave:
        max_length = max((len(results) for results in result_lists), default=0)
        stream = (
            results[index]
            for index in range(max_length)
            for results in result_lists
            if index < len(results)
        )
    else:
        stream = (hit for results in result_lists for hit in results)
    for hit in stream:
        if hit.segment_id in seen:
            continue
        seen.add(hit.segment_id)
        ordered.append(
            SearchHit(
                segment_id=hit.segment_id,
                score=1.0 - len(ordered) / max(1, top_k),
                source=hit.source,
                rank=len(ordered) + 1,
                metadata={**hit.metadata, "raw_score": hit.score},
            )
        )
        if len(ordered) >= top_k:
            break
    return ordered
