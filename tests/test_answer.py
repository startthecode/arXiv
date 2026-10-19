from rag.answer import build_messages, format_sources

P = {
    "paper_id": "2610.13563", "title": "Not All Fixes", "section_path": ["4 Results", "4.1 Tests"],
    "page_start": 12, "page_end": 13, "text": "F2P tests discriminate patches.",
    "breadcrumb": "Not All Fixes > 4 Results > 4.1 Tests", "url": "https://arxiv.org/abs/2610.13563",
}  # fmt: skip


def test_build_messages_numbers_sources_and_ends_with_question():
    msgs = build_messages("What do F2P tests do?", [P, P | {"paper_id": "2610.99999"}])
    assert [m["role"] for m in msgs] == ["system", "user"]
    user = msgs[1]["content"]
    assert user.startswith(
        "Sources:\n\n[1] Not All Fixes — 4 Results > 4.1 Tests (arXiv 2610.13563, pp. 12-13)\nF2P tests"
    )
    assert "[2] " in user and "2610.99999" in user
    assert user.endswith("Question: What do F2P tests do?")


def test_format_sources_maps_numbers_to_papers():
    out = format_sources([P])
    assert out.startswith("[1] 2610.13563 pp. 12-13") and "https://arxiv.org/abs/2610.13563" in out
