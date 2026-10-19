"""Ranking metrics for retrieval evaluation.

ranked: ids in the order the system returned them (best first)
rel:    {id: grade} for the relevant ids: 2 = contains the answer, 1 = partially relevant
k:      only the top k results count (what a user, or the LLM's context, would actually see)
"""

import math


def _hits(ranked: list[str], rel: dict[str, int], k: int) -> list[bool]:
    return [rel.get(x, 0) > 0 for x in ranked[:k]]


def accuracy_at_k(ranked, rel, k) -> float:
    """1 if anything relevant is in the top k, else 0 (a.k.a. hit rate, success@k). "Did we find it at all?\""""
    return float(any(_hits(ranked, rel, k)))


def precision_at_k(ranked, rel, k) -> float:
    """Share of the top k that is relevant. "How much of what we returned is useful?\""""
    return sum(_hits(ranked, rel, k)) / k


def recall_at_k(ranked, rel, k) -> float:
    """Share of all relevant items that made it into the top k. "How much of what exists did we find?\""""
    n = sum(1 for g in rel.values() if g > 0)
    return sum(_hits(ranked, rel, k)) / n if n else 0.0


def mrr_at_k(ranked, rel, k) -> float:
    """1 / rank of the FIRST relevant item (1.0 at rank 1, 0.5 at rank 2, ..., 0 if none in top k).
    "How far down did we have to look?\""""
    for i, hit in enumerate(_hits(ranked, rel, k), 1):
        if hit:
            return 1 / i
    return 0.0


def ndcg_at_k(ranked, rel, k) -> float:
    """Graded and position-aware: each relevant item adds (2^grade - 1) / log2(rank + 1), so the answer
    (grade 2) counts more than a neighbour (grade 1), and both count more near the top. Divided by the
    score of the perfect ordering, so 1.0 = ideal ranking."""
    dcg = sum((2 ** rel.get(x, 0) - 1) / math.log2(i + 1) for i, x in enumerate(ranked[:k], 1))
    ideal = sorted((g for g in rel.values() if g > 0), reverse=True)[:k]
    idcg = sum((2**g - 1) / math.log2(i + 1) for i, g in enumerate(ideal, 1))
    return dcg / idcg if idcg else 0.0


METRICS = {"Acc": accuracy_at_k, "P": precision_at_k, "R": recall_at_k, "MRR": mrr_at_k, "nDCG": ndcg_at_k}
