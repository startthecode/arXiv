"""Context builder: turn reranked chunks into the numbered sources the LLM reads.

  drop weak chunks ─► fit a token budget ─► group by paper ─► merge neighbouring chunks ─► sources [1..n]

Why: the LLM's context window is limited and every irrelevant passage is a chance to go off-track.
Grouping and merging means one paper's argument reads as one coherent passage, not scattered pieces.
"""

from rag.settings import Settings


def strip_overlap(a: str, b: str, max_chars: int = 1500) -> str:
    """b minus the text it repeats from the end of a (consecutive chunks overlap by ~64 tokens)."""
    for n in range(min(len(a), len(b), max_chars), 20, -1):
        if a.endswith(b[:n]):
            return b[n:].lstrip()
    return b


def build_context(results: list[tuple[dict, float]], s: Settings, count, use_scores: bool = True) -> list[dict]:
    """results: [(payload, score)] best first. Returns sources (payload-shaped dicts, possibly merged)."""
    # 1) relevance + budget, greedily in score order: the best chunks always get in first
    picked, used = [], 0
    for p, score in results:
        if use_scores and score < s.context_min_score:
            continue
        n = count(p["text"])
        if used + n > s.context_max_tokens:
            continue
        picked.append((p, score))
        used += n

    # 2) group by paper (best paper first), document order inside each paper; 3) merge adjacent chunks
    sources: list[dict] = []
    for pid in dict.fromkeys(p["paper_id"] for p, _ in picked):
        best = max(score for p, score in picked if p["paper_id"] == pid)
        last_index = None
        for p in sorted((p for p, _ in picked if p["paper_id"] == pid), key=lambda p: p["chunk_index"]):
            if last_index is not None and p["chunk_index"] == last_index + 1:
                prev = sources[-1]
                prev["text"] += "\n\n" + strip_overlap(prev["text"], p["text"])
                prev["page_end"] = max(prev["page_end"], p["page_end"])
            else:
                sources.append({**p, "score": best})
            last_index = p["chunk_index"]
    return sources
