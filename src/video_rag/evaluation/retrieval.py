from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import math


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return float(bool(set(ranked[:k]) & relevant))


def reciprocal_rank(ranked: Sequence[str], relevant: set[str]) -> float:
    for rank, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, item in enumerate(ranked[:k], start=1)
        if item in relevant
    )
    ideal_hits = min(len(relevant), k)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / ideal if ideal else 0.0


def evaluate_retrieval(
    predictions: Mapping[str, Sequence[str]],
    ground_truth: Mapping[str, Iterable[str]],
    *,
    cutoffs: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    question_ids = sorted(ground_truth)
    if not question_ids:
        raise ValueError("ground_truth must not be empty")

    totals = {f"recall@{k}": 0.0 for k in cutoffs}
    totals.update({f"ndcg@{k}": 0.0 for k in cutoffs})
    totals["mrr"] = 0.0

    for question_id in question_ids:
        ranked = predictions.get(question_id, ())
        relevant = set(ground_truth[question_id])
        totals["mrr"] += reciprocal_rank(ranked, relevant)
        for k in cutoffs:
            totals[f"recall@{k}"] += recall_at_k(ranked, relevant, k)
            totals[f"ndcg@{k}"] += ndcg_at_k(ranked, relevant, k)

    return {name: value / len(question_ids) for name, value in totals.items()}

