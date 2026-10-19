import json
import re

import pymupdf
import pytest

from rag import parse
from rag.settings import Settings


def make_pdf(path, pages=3):
    doc = pymupdf.open()
    for i in range(1, pages + 1):
        page = doc.new_page()
        page.insert_text((72, 72), f"Section {i}", fontsize=16)
        page.insert_text((72, 110), f"Body text on page {i}. " * 3, fontsize=10)
    doc.set_metadata({"title": "Test Paper", "author": "Ada Lovelace; Alan Turing", "keywords": "rag, chunking"})
    doc.set_toc([[1, "Section 1", 1], [1, "Section 2", 2], [2, "Section 2.1", 3]])
    doc.save(path)


@pytest.fixture
def s(tmp_path):
    (tmp_path / "pdf").mkdir()
    make_pdf(tmp_path / "pdf" / "2601.00001.pdf")
    return Settings(pdf_dir=tmp_path / "pdf", md_dir=tmp_path / "md", _env_file=None)


def test_writes_page_tagged_md_and_sidecar(s):
    r = parse.run(s, workers=1)
    assert (r["parsed"], r["failed"], r["pages"]) == (1, 0, 3)

    md = (s.md_dir / "2601.00001.md").read_text(encoding="utf-8")
    assert re.findall(r"<!-- page: (\d+) -->", md) == ["1", "2", "3"]
    page2 = md.split("<!-- page: 2 -->")[1].split("<!-- page: 3 -->")[0]
    assert "Body text on page 2" in page2

    side = json.loads((s.md_dir / "2601.00001.json").read_text(encoding="utf-8"))
    assert side["title"] == "Test Paper"
    assert side["authors"] == ["Ada Lovelace", "Alan Turing"]
    assert side["keywords"] == ["rag", "chunking"]
    assert side["toc"][2] == [2, "Section 2.1", 3]
    assert side["page_count"] == 3
    assert side["parser"]["opts_hash"] == parse.opts_hash(True)


def test_cached_then_force(s):
    parse.run(s, workers=1)
    assert parse.run(s, workers=1)["skipped"] == 1
    assert parse.run(s, workers=1, force=True)["parsed"] == 1


def test_corrupt_pdf_is_logged_and_retried(s):
    (s.pdf_dir / "bad.pdf").write_bytes(b"not a pdf")
    r = parse.run(s, workers=1)
    assert (r["parsed"], r["failed"]) == (1, 1)
    assert not (s.md_dir / "bad.md").exists()
    fail = json.loads((s.md_dir / "_failures.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert fail["id"] == "bad" and not fail["ok"]
    assert parse.run(s, workers=1)["failed"] == 1  # failures are not cached


def test_process_pool_legacy_mode(s):
    make_pdf(s.pdf_dir / "2601.00002.pdf")
    s = s.model_copy(update={"parse_layout": False})
    r = parse.run(s, workers=2)
    assert (r["parsed"], r["failed"], r["workers"]) == (2, 0, 2)
    side = json.loads((s.md_dir / "2601.00002.json").read_text(encoding="utf-8"))
    assert side["parser"]["layout"] is False


def test_unknown_id_raises(s):
    with pytest.raises(FileNotFoundError):
        parse.discover(s.pdf_dir, ids=["nope"])
