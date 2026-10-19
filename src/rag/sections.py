"""Page-tagged markdown -> blocks -> leveled headings -> section tree. Pure functions, no I/O."""

import re
from dataclasses import dataclass, field

from rag.clean import PICTURE_TEXT, clean_inline, is_bib_like

PAGE = re.compile(r"^<!-- page: (\d+) -->$")
HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
END_PUNCT = (".", "!", "?", ":", ";")
MAX_DEPTH = 4

NUMBERED = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+\S")  # 3 / 3.2 / 3.2.1.
LETTERED = re.compile(r"^([A-Z])((?:\.\d+)*)\.?\s+\S")  # A / B.3 (appendix) or A. (IEEE subsection)
ROMAN = re.compile(r"^([IVX]+)\.\s+\S")  # IEEE: II. METHOD
ROMAN_SECTION = re.compile(r"^(?:II|III|IV|V|VI|VII|VIII|IX|X)\.\s")
CAPTION = re.compile(r"^(?:figure|fig|table|tab|algorithm|listing|prompt|box|example)\.?\s*\d", re.I)
ABSTRACT_PREFIX = re.compile(r"^\W*abstract\b[\W_]*", re.I)

REFERENCES = {
    "references", "reference", "bibliography", "references and notes", "references cited",
    "literature cited", "works cited", "cited works", "citations", "literature",
}  # fmt: skip
# admin sections with no retrieval value
DROP = re.compile(
    r"^(?:acknowledge?ments?|funding|author contributions?|contributions? statement|competing interests?"
    r"|conflicts? of interests?|declaration|(?:neurips |iclr |icml )?(?:paper )?checklist|reproducibility checklist"
    r"|(?:table of )?contents)\b"
)
TOP_LEVEL = {
    "abstract", "introduction", "related work", "related works", "background", "preliminaries",
    "method", "methods", "methodology", "approach", "experiments", "experimental setup", "results",
    "evaluation", "discussion", "conclusion", "conclusions", "conclusion and future work", "limitations",
    "future work", "ethics statement", "broader impact", "impact statement", "appendix", "appendices",
    "acknowledgments", "acknowledgements", "acknowledgment", "acknowledgement", *REFERENCES,
}  # fmt: skip


@dataclass
class Block:
    kind: str  # heading | text | table | code
    text: str
    page_start: int
    page_end: int
    md_level: int = 0  # '#' count from the parser (headings only)
    level: int = 0  # section level we assign (headings only)


@dataclass
class Section:
    path: list[str]  # heading texts, top level first
    level: int
    kind: str  # abstract | body | appendix
    page: int
    blocks: list[Block] = field(default_factory=list)

    def text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks)


def norm(text: str) -> str:
    """Comparable form of a heading: lowercase, no numbering/punctuation. '2.1 Data Sets' -> 'data sets'."""
    s = clean_inline(text).lower()
    if re.fullmatch(r"(?:\w ){3,}\w", s):  # letter-spaced: "a b s t r a c t"
        s = s.replace(" ", "")
    s = re.sub(r"^(?:\d+(?:\.\d+)*|[a-z](?:\.\d+)*|[ivx]+)[.:)]?\s+", "", s)
    return " ".join(re.sub(r"[^\w\s]", " ", s).split())


# ---------------------------------------------------------------- blocks


def read_blocks(md: str) -> list[Block]:
    """Split markdown into heading/text/table/code blocks, each tagged with its page(s)."""
    blocks: list[Block] = []
    buf: list[str] = []
    kind, start, page, in_code = "text", 1, 1, False

    def flush() -> None:
        if buf:
            text = "\n".join(buf) if kind == "code" else "\n".join(filter(None, map(clean_inline, buf)))
            if text.strip():
                blocks.append(Block(kind, text, start, page))
            buf.clear()

    for line in PICTURE_TEXT.sub("", md).split("\n"):
        s = line.strip()
        if m := PAGE.match(s):
            if not in_code:
                flush()
            page = int(m[1])
            continue
        if in_code:
            buf.append(line)
            if s.startswith("```"):
                flush()
                in_code = False
            continue
        if s.startswith("```"):
            flush()
            kind, start, in_code = "code", page, True
            buf.append(line)
            continue
        if not s:
            flush()
            continue
        if m := HEADING.match(s):
            flush()
            if text := clean_inline(m[2]):
                blocks.append(Block("heading", text, page, page, md_level=len(m[1])))
            continue
        line_kind = "table" if s.startswith("|") else "text"
        if buf and line_kind != kind:
            flush()
        if not buf:
            kind, start = line_kind, page
        buf.append(s[2:] if s.startswith("> ") else s)  # footnotes arrive as blockquotes
    flush()
    return blocks


def _bib_entries(b: Block) -> int:
    """How many reference-list entries a text block holds (0 unless most of its lines look like references)."""
    if b.kind != "text":
        return 0
    lines = b.text.split("\n")
    hits = sum(map(is_bib_like, lines))
    return hits if hits >= 0.6 * len(lines) else 0


def drop_bibliography_runs(blocks: list[Block], min_entries: int = 3) -> list[Block]:
    """Remove runs of reference-like blocks holding 3+ entries in total. Catches bibliographies whose
    heading wasn't detected, whether entries arrive one per block or many lines in one block."""
    entries = [_bib_entries(b) for b in blocks]
    keep = [True] * len(blocks)
    i = 0
    while i < len(blocks):
        j = i
        while j < len(blocks) and entries[j]:
            j += 1
        if sum(entries[i:j]) >= min_entries:
            keep[i:j] = [False] * (j - i)
        i = max(j, i + 1)
    return [b for b, k in zip(blocks, keep) if k]


def merge_continuations(blocks: list[Block]) -> list[Block]:
    """Re-join a paragraph split by a page or column break (no end punctuation + lowercase next)."""
    out: list[Block] = []
    for b in blocks:
        p = out[-1] if out else None
        if p and p.kind == b.kind == "text" and not p.text.endswith(END_PUNCT) and b.text[:1].islower():
            out[-1] = Block("text", f"{p.text} {b.text}", p.page_start, b.page_end)
        else:
            out.append(b)
    return out


def guess_title(blocks: list[Block]) -> str:
    return next((b.text for b in blocks if b.kind == "heading" and b.page_start == 1 and len(b.text.split()) >= 3), "")


# ---------------------------------------------------------------- heading levels


def numbering_depth(text: str, roman: bool, appendix: bool) -> int | None:
    """Depth implied by a heading's numbering, or None if it isn't numbered."""
    if m := NUMBERED.match(text):
        return m[1].count(".") + 1
    if roman and ROMAN.match(text):
        return 1
    if m := LETTERED.match(text):
        if roman and text[1:2] == ".":
            return 2 + m[2].count(".")  # IEEE: II. Method > A. Datasets
        if appendix or m[2]:
            return 1 + m[2].count(".")  # Appendix A / A.1 (a bare "A" before references is likely "A ...")
    return None


def _toc_level(n: str, toc_levels: dict[str, int]) -> int | None:
    if not n or not toc_levels:
        return None
    if n in toc_levels:
        return toc_levels[n]
    if len(n) >= 12:
        for k, v in toc_levels.items():
            if len(k) >= 12 and (n.startswith(k) or k.startswith(n)):
                return v
    return None


def _fake_heading(text: str, n: str, toc_levels: dict[str, int]) -> bool:
    """Parser 'headings' that are really captions, citations, run-in labels or stray fragments."""
    if _toc_level(n, toc_levels) is not None:
        return False
    return (
        not re.search(r"[a-z]", n)
        or len(text.split()) > 15
        or bool(CAPTION.match(text))
        or text.startswith("[")
        or text.endswith(":")
        or text[:1].islower()
    )


def assign_levels(blocks: list[Block], toc: list, title: str) -> list[Block]:
    """Give every heading a level, trusting (in order) the PDF outline, numbering, well-known
    section names, then the parser's '#' depth. Fake headings become text; title repeats and
    front-matter headings (author names, affiliations) before the first real section are dropped."""
    heads = [b for b in blocks if b.kind == "heading"]

    toc_levels: dict[str, int] = {}
    if len(toc) >= 3 and max(t[0] for t in toc) <= 6:
        for lvl, text, *_ in toc:
            toc_levels.setdefault(norm(text), lvl)
        if sum(_toc_level(norm(h.text), toc_levels) is not None for h in heads) < len(toc_levels) / 2:
            toc_levels = {}  # outline doesn't match the text; don't trust it
    use_toc = bool(toc_levels)
    roman = sum(bool(ROMAN_SECTION.match(h.text)) for h in heads) >= 2
    numbered = sum(bool(NUMBERED.match(h.text)) for h in heads) >= 3
    md_min = min((h.md_level for h in heads), default=1)
    title_n = norm(title)

    def strong(h: Block) -> bool:
        n = norm(h.text)
        return (
            _toc_level(n, toc_levels) is not None
            or n in TOP_LEVEL
            or (not use_toc and numbering_depth(h.text, roman, False) is not None)
        )

    first = next((i for i, b in enumerate(blocks) if b.kind == "heading" and strong(b)), 0)

    out: list[Block] = []
    anchor, prev, appendix = 0, 0, False  # anchor = level of the last trusted heading
    for i, b in enumerate(blocks):
        if b.kind != "heading":
            out.append(b)
            continue
        n = norm(b.text)
        if b.page_start <= 2 and n not in TOP_LEVEL and (n == title_n or (len(n) >= 20 and n in title_n)):
            continue  # the title (or a fragment of it) repeated as a heading
        if i < first or _fake_heading(b.text, n, toc_levels):
            out.append(Block("text", b.text, b.page_start, b.page_end))
            continue
        trusted = True
        if (lvl := _toc_level(n, toc_levels)) is not None:
            level = lvl
        elif not use_toc and (d := numbering_depth(b.text, roman, appendix)) is not None:
            level = d
        elif n in TOP_LEVEL or n.startswith("appendi"):
            level = 1
        elif use_toc or numbered:
            level, trusted = anchor + 1, False  # unlisted sub-heading (box title, prompt part): child of current
        else:
            level = min(b.md_level - md_min + 1, prev + 1)  # no outline, no numbering: parser's '#' depth
        level = max(1, min(level, MAX_DEPTH))
        if trusted:
            anchor = level
        prev = level
        appendix = appendix or n in REFERENCES or n.startswith("appendi")
        out.append(Block("heading", b.text, b.page_start, b.page_end, b.md_level, level))
    return out


# ---------------------------------------------------------------- sections


def _front_abstract(front: list[Block]) -> Section | None:
    """Abstract without its own heading: an 'Abstract—...' paragraph, else the first run of long paragraphs."""
    texts = [b for b in front if b.kind == "text"]
    start = next((i for i, b in enumerate(texts) if ABSTRACT_PREFIX.match(b.text)), None)
    body: list[Block] = []
    if start is not None:
        first = texts[start]
        if rest := ABSTRACT_PREFIX.sub("", first.text, count=1):
            body.append(Block("text", rest, first.page_start, first.page_end))
        for b in texts[start + 1 :]:
            if len(b.text.split()) < 40:  # keywords, footnotes, affiliations
                break
            body.append(b)
    else:
        for b in texts:
            if len(b.text.split()) >= 50:
                body.append(b)
            elif body:
                break
    return Section(["Abstract"], 1, "abstract", body[0].page_start, body) if body else None


def build_sections(blocks: list[Block]) -> list[Section]:
    """Walk leveled blocks into sections with heading paths. References and admin sections
    (with their sub-sections) are dropped; everything after References is appendix."""
    front: list[Block] = []
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    cur: Section | None = None
    skip_level: int | None = None
    appendix = False
    for b in blocks:
        if b.kind != "heading":
            if skip_level is None:
                (cur.blocks if cur else front).append(b)
            continue
        n = norm(b.text)
        if skip_level is not None and b.level > skip_level:
            continue  # sub-heading of a dropped section
        skip_level = None
        if n in REFERENCES or DROP.match(n):
            appendix = appendix or n in REFERENCES
            skip_level = b.level
            continue
        appendix = appendix or n.startswith("appendi") or n.startswith("supplementary")
        while stack and stack[-1][0] >= b.level:
            stack.pop()
        stack.append((b.level, b.text))
        kind = "abstract" if n == "abstract" else "appendix" if appendix else "body"
        cur = Section([t for _, t in stack], b.level, kind, b.page_start)
        sections.append(cur)
    if not any(s.kind == "abstract" for s in sections) and (abstract := _front_abstract(front)):
        sections.insert(0, abstract)
    return sections
