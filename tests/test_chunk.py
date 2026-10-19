import pytest

from rag.chunk import chunk_paper, make_header, merge_tiny, split_table
from rag.sections import Block, Section, assign_levels, build_sections, drop_bibliography_runs, read_blocks
from rag.settings import Settings


def count(text: str) -> int:  # whitespace "tokenizer": keeps the tests offline and easy to reason about
    return len(text.split())


@pytest.fixture
def s():
    return Settings(_env_file=None, chunk_max_tokens=60, chunk_overlap_tokens=10, chunk_min_tokens=15, chunk_drop_tokens=3)


# ---------------------------------------------------------------- blocks


def test_read_blocks_kinds_pages_and_noise():
    md = (
        "<!-- page: 1 -->\n\n## **Intro**\n\nHello world\n\n"
        "<!-- Start of picture text -->noise<br>more<!-- End of picture text -->\n\n"
        "<!-- page: 2 -->\n\n```\ncode line\n\nmore code\n```\n\n"
        "|a|b|\n|---|---|\n|1|2|\n\n> footnote here\n"
    )
    blocks = read_blocks(md)
    assert [(b.kind, b.page_start) for b in blocks] == [("heading", 1), ("text", 1), ("code", 2), ("table", 2), ("text", 2)]
    assert blocks[0].text == "Intro" and blocks[0].md_level == 2
    assert blocks[2].text == "```\ncode line\n\nmore code\n```"  # blank line inside code doesn't split it
    assert blocks[4].text == "footnote here"
    assert "noise" not in " ".join(b.text for b in blocks)


def test_bibliography_runs_dropped_but_single_citation_kept():
    refs = [Block("text", f"- [{i}] A. Author. 2020. Title. In Proceedings of X.", 1, 1) for i in range(3)]
    body = Block("text", "Smith et al. (2020) showed this.", 1, 1)  # looks like a citation, but alone
    plain = Block("text", "We then train the model.", 1, 1)
    assert drop_bibliography_runs([body, plain, *refs]) == [body, plain]


# ---------------------------------------------------------------- heading levels


def _levels(md, toc=(), title="T"):
    return {b.text: b.level for b in assign_levels(read_blocks(md), list(toc), title) if b.kind == "heading"}


def test_levels_from_outline_and_unlisted_subheadings():
    md = "\n\n".join(
        ["## **Abstract**", "x", "## **1 Introduction**", "x", "## **1.1 Scope**", "x", "#### **User Prompt**", "x", "## **2 Results**", "x"]
    )
    toc = [[1, "Abstract", 1], [1, "Introduction", 1], [2, "Scope", 1], [1, "Results", 1]]
    assert _levels(md, toc) == {"Abstract": 1, "1 Introduction": 1, "1.1 Scope": 2, "User Prompt": 3, "2 Results": 1}


def test_levels_from_ieee_roman_numbering():
    md = "\n\n".join(["## I. INTRODUCTION", "x", "## II. METHOD", "x", "## _A. Datasets_", "x", "## III. RESULTS", "x"])
    assert _levels(md) == {"I. INTRODUCTION": 1, "II. METHOD": 1, "A. Datasets": 2, "III. RESULTS": 1}


def test_fake_headings_and_front_matter_become_text():
    md = "\n\n".join(
        ["# Jane Doe", "Some University", "## **Abstract**", "x", "## **1 Intro**", "x", "##### **Table 2**", "x", "## distinct solutions", "x"]
    )
    assert _levels(md) == {"Abstract": 1, "1 Intro": 1}  # author name, caption, lowercase fragment demoted


def test_abstract_from_front_matter_prefix():
    md = "Some Author\n\nAbstract—" + "We propose a thing. " * 15 + "\n\nKeywords: x, y\n\n## **1 Introduction**\n\nIntro text."
    secs = build_sections(assign_levels(read_blocks(md), [], "T"))
    assert secs[0].kind == "abstract" and secs[0].text().startswith("We propose")
    assert "Keywords" not in secs[0].text() and "Some Author" not in secs[0].text()


# ---------------------------------------------------------------- end to end

P1 = " ".join(f"Sentence number {i} talks about things." for i in range(30))  # 180 words
ROWS = "\n".join(f"|row{i} item|value {i}|" for i in range(40))
MD = f"""<!-- page: 1 -->

# **A Study of Things**

Ada Lovelace, Alan Turing

## **Abstract**

We study things. {"word " * 20}

## **1 Introduction**

{P1}

This paragraph continues on the

<!-- page: 2 -->

next page with lowercase start.

## **2 Method**

Method intro has five words.

## **2.1 Data**

The data section has exactly ten words in it ok.

## **2.2 Model**

The model section is short too.

<!-- page: 3 -->

## **3 Results**

Results are shown below.

|Model|Score|
|---|---|
|A|1|
|B|2|

|Name|Value|
|---|---|
{ROWS}

## **Acknowledgments**

We thank everyone for their help.

## **References**

- [1] A. Author. 2020. Title one. In Proceedings of X.

- [2] B. Author. 2021. Title two. In Proceedings of Y.

- [3] C. Author. 2022. Title three. In Proceedings of Z.

## **A Extra Details**

Appendix content goes here with enough words.
"""
META = {"paper_id": "2601.00001", "title": "A Study of Things", "authors": ["Ada Lovelace", "Alan Turing"], "toc": []}


def test_chunk_paper_end_to_end(s):
    recs = chunk_paper(MD, META, s, count)
    by_top = lambda top: [r for r in recs if r["section_top"] == top]  # noqa: E731

    # abstract first, with authors in the embedded header
    assert recs[0]["section_type"] == "abstract" and recs[0]["chunk_index"] == 0
    assert "Authors: Ada Lovelace, Alan Turing" in recs[0]["embed_text"]
    assert recs[0]["chunk_id"] == "2601.00001:0000"

    # every chunk fits the budget, header included
    assert all(count(r["embed_text"]) <= s.chunk_max_tokens for r in recs)
    assert all(r["embed_text"].startswith("A Study of Things\n") for r in recs)

    # long section split by sentences, consecutive chunks overlap by a sentence
    intro = [r for r in by_top("introduction") if r["text"].startswith("Sentence")]
    assert len(intro) >= 3
    for a, b in zip(intro, intro[1:]):
        assert a["text"].split(". ")[-1] in b["text"]

    # paragraph re-joined across the page break, with the right page span
    cont = next(r for r in recs if r["text"].startswith("This paragraph continues"))
    assert cont["text"] == "This paragraph continues on the next page with lowercase start."
    assert (cont["page_start"], cont["page_end"]) == (1, 2)

    # short parent intro + tiny siblings merged into one chunk
    (method,) = by_top("method")
    assert method["section_path"] == ["2 Method", "2.1 Data / 2.2 Model"]
    assert "Method intro" in method["text"] and "2.2 Model" in method["text"] and "model section" in method["text"]

    # small table intact, big table split by rows with the header repeated
    results = by_top("results")
    assert any("|Model|Score|\n|---|---|\n|A|1|\n|B|2|" in r["text"] for r in results)
    table_parts = [r for r in results if "|row" in r["text"]]
    assert len(table_parts) >= 3 and all("|Name|Value|\n|---|---|" in r["text"] for r in table_parts)
    assert all(r["page_start"] == 3 for r in results)

    # admin sections and references gone, appendix kept
    everything = " ".join(r["text"] for r in recs)
    assert "We thank" not in everything and "Proceedings" not in everything and "Ada Lovelace, Alan" not in everything
    (appendix,) = [r for r in recs if r["section_type"] == "appendix"]
    assert appendix["section_path"] == ["A Extra Details"]

    # deterministic
    assert chunk_paper(MD, META, s, count) == recs


def test_bibliography_packed_into_one_block_is_dropped():
    refs = "\n".join(f"- [{i}] A. Author et al. Title. In Proceedings of X, 2021." for i in range(4))
    packed = Block("text", refs, 5, 5)  # entries on consecutive lines -> one block
    mixed = Block("text", "See [1] for details.\nWe follow Smith et al. (2020).", 5, 5)  # 1 of 2 lines bib-like
    assert drop_bibliography_runs([packed, mixed]) == [mixed]


def test_abstract_header_caps_author_list():
    authors = [f"Author {i}" for i in range(300)]
    header = make_header("T", Section(["Abstract"], 1, "abstract", 1), authors)
    assert header.rstrip().endswith("Author 8, Author 9 et al.") and "Author 10" not in header


def test_empty_heading_does_not_leak_into_breadcrumb(s):
    secs = [
        Section(["Appendix"], 1, "appendix", 21),  # heading with no text of its own
        Section(["A Details"], 1, "appendix", 21, [Block("text", "Some appendix text here.", 21, 21)]),
    ]
    (merged,) = merge_tiny(secs, "T", s, count)
    assert merged.path == ["A Details"]


def test_split_table_repeats_header():
    table = "|h1|h2|\n|---|---|\n" + "\n".join(f"|a{i}|b{i}|" for i in range(10))
    parts = split_table(table, budget=6, count=count)
    assert len(parts) > 1 and all(p.startswith("|h1|h2|\n|---|---|\n") for p in parts)
