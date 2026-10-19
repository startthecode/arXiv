"""Text cleanup for parsed markdown. Pure functions, no I/O."""

import re
import unicodedata

# text that pymupdf4llm extracts from inside figures: axis labels, legends, ... -> noise
PICTURE_TEXT = re.compile(r"<!--\s*Start of picture text\s*-->.*?<!--\s*End of picture text\s*-->", re.S)

_BR = re.compile(r"<br\s*/?>", re.I)
_TAG = re.compile(r"</?(?:sup|sub|mark|strong|span|small|em|u|b|i|s)\b[^>]*>", re.I)
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_STRIKE = re.compile(r"~~(.+?)~~")
_ITALIC = re.compile(r"(?<![\w\\])_(?=\S)(.+?)(?<=\S)_(?!\w)")  # _x_ but not snake_case
_SPACES = re.compile(r"[ \t]+")


def clean_inline(text: str) -> str:
    """Normalize one line: unicode (math italics, ligatures), HTML tags, markdown emphasis."""
    text = unicodedata.normalize("NFKC", text).replace("�", "")  # 𝑝 -> p, ﬁ -> fi; drop broken glyphs
    text = _BR.sub(" ", text)
    text = _TAG.sub("", text)
    text = _BOLD.sub(r"\1", text)
    text = _STRIKE.sub(r"\1", text)
    text = _ITALIC.sub(r"\1", text)
    return _SPACES.sub(" ", text).strip()


_BIB_PATTERNS = [
    re.compile(p)
    for p in (
        r"^(?:- )?\[\d+\]",  # [12] Author ...
        r"\b(?:19|20)\d{2}[a-z]?\b[.,)]",  # year followed by punctuation
        r"\bet al\.",
        r"arXiv|preprint",
        r"\bdoi\b|https?://",
        r"Proceedings|Conference|Journal|Transactions|Workshop|Symposium",
    )
]


def is_bib_like(text: str) -> bool:
    """A short block that looks like a reference-list entry (2+ citation signals)."""
    return len(text.split()) <= 80 and sum(bool(p.search(text)) for p in _BIB_PATTERNS) >= 2
