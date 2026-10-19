import pytest

from rag.answer import rrf_fuse
from rag.context import build_context, strip_overlap
from rag.settings import Settings

SHARED = "this sentence is shared by both chunks."


def count(text: str) -> int:
    return len(text.split())


def P(pid, idx, text, pages=(3, 3)):
    return {
        "paper_id": pid, "chunk_id": f"{pid}:{idx:04d}", "chunk_index": idx, "title": f"Title {pid}",
        "section_path": ["1 Intro"], "breadcrumb": f"Title {pid} > 1 Intro", "page_start": pages[0],
        "page_end": pages[1], "text": text, "url": f"https://arxiv.org/abs/{pid}",
    }  # fmt: skip


@pytest.fixture
def s():
    return Settings(_env_file=None, context_max_tokens=1000, context_min_score=0.05)


def test_rrf_fuse_rewards_chunks_found_by_several_queries():
    fused = rrf_fuse([["a", "b", "c"], ["c", "a"]], key=lambda x: x)
    assert [item for item, _ in fused] == ["a", "c", "b"]


def test_strip_overlap():
    assert strip_overlap(f"alpha one. {SHARED}", f"{SHARED} alpha two.") == "alpha two."
    assert strip_overlap("no overlap here at all", "completely different text") == "completely different text"


def test_context_drops_weak_groups_by_paper_and_merges_neighbours(s):
    results = [
        (P("A", 5, f"alpha one. {SHARED}"), 0.9),
        (P("B", 1, "beta text"), 0.8),
        (P("A", 6, f"{SHARED} alpha two.", pages=(3, 4)), 0.7),
        (P("C", 0, "weak"), 0.01),  # below context_min_score
    ]
    sources = build_context(results, s, count)
    assert [src["paper_id"] for src in sources] == ["A", "B"]  # best paper first, C dropped
    assert sources[0]["text"] == f"alpha one. {SHARED}\n\nalpha two."  # merged, overlap removed
    assert (sources[0]["page_start"], sources[0]["page_end"]) == (3, 4)
    assert sources[0]["score"] == 0.9


def test_context_respects_token_budget(s):
    s = s.model_copy(update={"context_max_tokens": 5})
    results = [(P("A", 1, "one two three four"), 0.9), (P("B", 1, "five six seven"), 0.8), (P("C", 1, "eight"), 0.7)]
    assert [src["paper_id"] for src in build_context(results, s, count)] == ["A", "C"]  # B doesn't fit, C does
