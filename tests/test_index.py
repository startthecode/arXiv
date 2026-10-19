from rag.index import BM25_MODEL, bm25_doc, payload, point_id

REC = {
    "chunk_id": "2601.00001:0003", "paper_id": "2601.00001", "chunk_index": 3, "title": "T", "authors": ["A"],
    "section_path": ["1 Intro"], "breadcrumb": "T > 1 Intro", "section_level": 1, "section_type": "body",
    "section_top": "intro", "page_start": 2, "page_end": 2, "text": "raw text", "embed_text": "T\n1 Intro\n\nraw text",
    "n_tokens": 7,
}  # fmt: skip


def test_point_id_is_deterministic_uuid():
    assert point_id("2601.00001:0003") == point_id("2601.00001:0003")  # re-indexing overwrites, never duplicates
    assert point_id("2601.00001:0003") != point_id("2601.00001:0004")
    assert len(point_id("x")) == 36  # Qdrant ids must be unsigned ints or UUIDs


def test_payload_keeps_display_text_drops_embed_text():
    p = payload(REC)
    assert p["text"] == "raw text" and "embed_text" not in p and "n_tokens" not in p
    assert p["url"] == "https://arxiv.org/abs/2601.00001"


def test_bm25_doc_targets_server_side_bm25():
    d = bm25_doc("hello world")
    assert d.model == BM25_MODEL and d.text == "hello world"
