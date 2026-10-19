import pytest

from rag.eval import paper_ranking, score
from rag.metrics import accuracy_at_k, mrr_at_k, ndcg_at_k, precision_at_k, recall_at_k

RANKED = ["a", "b", "c", "d", "e"]
REL = {"c": 2, "x": 1}  # the answer "c" is at rank 3; the partially relevant "x" was never retrieved


def test_metrics_by_hand():
    assert accuracy_at_k(RANKED, REL, 5) == 1.0  # something relevant is in the top 5
    assert accuracy_at_k(RANKED, REL, 2) == 0.0  # ...but not in the top 2
    assert precision_at_k(RANKED, REL, 5) == pytest.approx(1 / 5)  # 1 of 5 results is relevant
    assert recall_at_k(RANKED, REL, 5) == pytest.approx(1 / 2)  # found 1 of the 2 relevant items
    assert mrr_at_k(RANKED, REL, 5) == pytest.approx(1 / 3)  # first relevant at rank 3
    # DCG = (2^2-1)/log2(4) = 1.5; ideal = c then x at ranks 1, 2 = 3/1 + 1/log2(3)
    assert ndcg_at_k(RANKED, REL, 5) == pytest.approx(1.5 / (3 + 1 / 1.5849625), rel=1e-6)


def test_perfect_ranking_scores_one():
    rel = {"a": 2, "b": 1}
    for fn in (accuracy_at_k, recall_at_k, mrr_at_k, ndcg_at_k):
        assert fn(["a", "b", "z"], rel, 3) == pytest.approx(1.0)
    assert ndcg_at_k(["b", "a", "z"], rel, 3) < 1.0  # right items, wrong order: nDCG notices, recall doesn't


def test_paper_ranking_keeps_first_appearance():
    assert paper_ranking(["2610.1:0003", "2610.2:0001", "2610.1:0007"]) == ["2610.1", "2610.2"]


def test_score_reports_both_levels():
    questions = [
        {"id": "q1", "paper_id": "P", "relevant": {"P:0002": 2}},
        {"id": "q2", "paper_id": "Q", "relevant": {}},  # paper level only
    ]
    runs = {"cfg": {"q1": ["X:0001", "P:0002"], "q2": ["Q:0005"]}}
    report, summary = score(questions, runs, k=5)
    assert summary["chunk/cfg"]["MRR"] == pytest.approx(0.5)  # only q1 has chunk labels; answer at rank 2
    assert summary["paper/cfg"]["MRR"] == pytest.approx((0.5 + 1.0) / 2)  # q1: paper P at rank 2; q2: Q at rank 1
    assert "CHUNK level: 1 questions" in report and "PAPER level: 2 questions" in report
