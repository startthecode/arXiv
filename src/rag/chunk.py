"""Stage 2: page-tagged markdown -> structure-aware, recursively split chunks (JSONL).

Pipeline per paper:
  read_blocks -> drop bibliography runs -> assign heading levels -> re-join split paragraphs
  -> section tree -> merge tiny sections -> recursive split of big ones -> records
"""

import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from multiprocessing import get_context

from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from rag.files import atomic_write
from rag.sections import (
    Block,
    Section,
    assign_levels,
    build_sections,
    drop_bibliography_runs,
    guess_title,
    merge_continuations,
    norm,
    read_blocks,
)
from rag.settings import Settings

CHUNKER_VERSION = 3  # bump when the chunking rules change, to invalidate cached chunks
MAX_AUTHORS = 10
# paragraph -> line -> sentence -> clause -> word -> character
SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]


@lru_cache
def token_counter(model: str):
    """Token counter using the embedding model's own tokenizer (only tokenizer.json is downloaded)."""
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(hf_hub_download(model, "tokenizer.json"))
    return lambda text: len(tok.encode(text, add_special_tokens=False).ids)


def make_header(title: str, sec: Section, authors: list[str]) -> str:
    """Context prepended to every chunk before embedding/BM25: paper title + section breadcrumb."""
    lines = [title]
    if sec.kind == "abstract":
        lines.append("Abstract")
        if authors:  # capped: some papers list hundreds of authors, which would eat the token budget
            lines.append("Authors: " + ", ".join(authors[:MAX_AUTHORS]) + (" et al." if len(authors) > MAX_AUTHORS else ""))
    else:
        lines.append(" > ".join(sec.path))
    return "\n".join(line for line in lines if line) + "\n\n"


def merge_tiny(secs: list[Section], title: str, s: Settings, count) -> list[Section]:
    """Fold small sections into a neighbour so no chunk is just a heading and a sentence:
    tiny siblings merge (headings kept inline); a short parent intro joins its first child."""
    out: list[Section] = []
    for sec in secs:
        prev = out[-1] if out else None
        if prev and not prev.blocks and sec.path[:-1] == prev.path[:-1]:
            out[-1] = sec  # a heading with no text of its own (e.g. "Appendix") adds nothing
            continue
        if prev and prev.kind == sec.kind != "abstract":
            a, b = count(prev.text()), count(sec.text())
            room = s.chunk_max_tokens - count(make_header(title, sec, [])) - 8
            if a + b <= room and min(a, b) < s.chunk_min_tokens:
                if sec.path[:-1] == prev.path[:-1]:  # siblings
                    prev.blocks += [Block("text", sec.path[-1], sec.page, sec.page), *sec.blocks]
                    prev.path = [*prev.path[:-1], f"{prev.path[-1]} / {sec.path[-1]}"]
                    continue
                if sec.path[:-1] == prev.path and a < s.chunk_min_tokens:  # parent intro + first child
                    sec.blocks = prev.blocks + sec.blocks
                    out[-1] = sec
                    continue
        out.append(sec)
    return out


def split_table(table: str, budget: int, count) -> list[str]:
    """Split an oversized markdown table by rows, repeating the header row in every piece."""
    lines = table.split("\n")
    has_head = len(lines) > 2 and set(lines[1]) <= set("|-: ")
    head, rows = (lines[:2], lines[2:]) if has_head else ([], lines)
    base = count("\n".join(head)) if head else 0
    parts, cur, size = [], [], base
    for row in rows:
        n = count(row) + 1
        if cur and size + n > budget:
            parts.append("\n".join(head + cur))
            cur, size = [], base
        cur.append(row)
        size += n
    if cur:
        parts.append("\n".join(head + cur))
    return parts


def split_section(sec: Section, header: str, s: Settings, count) -> list[tuple[str, int, int]]:
    """Section -> [(chunk text, page_start, page_end)]. Small sections stay whole; big ones are
    split recursively (paragraph -> line -> sentence -> word) with token overlap."""
    budget = max(s.chunk_max_tokens - count(header) - 4, 2 * s.chunk_overlap_tokens)
    pieces: list[tuple[str, int, int]] = []
    for b in sec.blocks:
        if b.kind == "table" and count(b.text) > budget:
            pieces += [(t, b.page_start, b.page_end) for t in split_table(b.text, budget, count)]
        else:
            pieces.append((b.text, b.page_start, b.page_end))

    text = "\n\n".join(t for t, _, _ in pieces)
    spans, pos = [], 0  # char offsets of each piece, to map chunks back to pages
    for t, p0, p1 in pieces:
        spans.append((pos, pos + len(t), p0, p1))
        pos += len(t) + 2

    if count(text) <= budget:
        chunks = [text.strip()]
    else:
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=budget,
            chunk_overlap=s.chunk_overlap_tokens,
            length_function=count,
            separators=SEPARATORS,
            keep_separator="end",
            strip_whitespace=True,
        ).split_text(text)

    out, cursor = [], 0
    for c in chunks:
        start = text.find(c, cursor)
        if start < 0:
            start = max(text.find(c), 0)
        end = start + len(c)
        pages = [(p0, p1) for a, b, p0, p1 in spans if a < end and b > start] or [(spans[0][2], spans[-1][3])]
        out.append((c, min(p for p, _ in pages), max(p for _, p in pages)))
        cursor = start + 1  # chunks overlap, so the next one starts after this one's start
    return out


def chunk_paper(md: str, meta: dict, s: Settings, count) -> list[dict]:
    pid = meta["paper_id"]
    authors = meta.get("authors") or []
    blocks = read_blocks(md)
    title = meta.get("title") or guess_title(blocks) or pid
    blocks = drop_bibliography_runs(blocks)
    blocks = assign_levels(blocks, meta.get("toc") or [], title)
    blocks = merge_continuations(blocks)
    secs = build_sections(blocks)
    secs.sort(key=lambda sec: sec.kind != "abstract")  # abstract is always chunk 0
    secs = merge_tiny(secs, title, s, count)

    records, seen = [], set()
    for sec in secs:
        if not sec.blocks:
            continue
        header = make_header(title, sec, authors)
        for text, p0, p1 in split_section(sec, header, s, count):
            if text in seen or (sec.kind != "abstract" and count(text) < s.chunk_drop_tokens):
                continue
            seen.add(text)
            records.append(
                {
                    "chunk_id": f"{pid}:{len(records):04d}",
                    "paper_id": pid,
                    "chunk_index": len(records),
                    "title": title,
                    "authors": authors,
                    "section_path": sec.path,
                    "breadcrumb": " > ".join([title, *sec.path]),
                    "section_level": sec.level,
                    "section_type": sec.kind,
                    "section_top": norm(sec.path[0]),
                    "page_start": p0,
                    "page_end": p1,
                    "text": text,
                    "embed_text": header + text,
                    "n_tokens": count(header + text),
                }
            )
    return records


# ---------------------------------------------------------------- batch run


def settings_hash(s: Settings) -> str:
    key = [CHUNKER_VERSION, s.embed_model, s.chunk_max_tokens, s.chunk_overlap_tokens, s.chunk_min_tokens, s.chunk_drop_tokens]
    return hashlib.sha256(json.dumps(key).encode()).hexdigest()[:16]


def chunk_file(pid: str, s: Settings) -> dict:
    try:
        md = (s.md_dir / f"{pid}.md").read_text(encoding="utf-8")
        meta = json.loads((s.md_dir / f"{pid}.json").read_text(encoding="utf-8"))
        records = chunk_paper(md, meta, s, token_counter(s.embed_model))
        atomic_write(s.chunks_dir / f"{pid}.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
        return {"id": pid, "ok": True, "chunks": len(records), "tokens": sum(r["n_tokens"] for r in records)}
    except Exception as e:
        return {"id": pid, "ok": False, "error": f"{type(e).__name__}: {e}"}


def _results(todo: list[str], s: Settings, workers: int):
    if workers <= 1:
        for pid in todo:
            yield chunk_file(pid, s)
        return
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"))
    try:
        futures = {pool.submit(chunk_file, pid, s): pid for pid in todo}
        for f in as_completed(futures):
            try:
                yield f.result()
            except Exception as e:
                yield {"id": futures[f], "ok": False, "error": f"{type(e).__name__}: {e}"}
    finally:
        pool.shutdown(cancel_futures=True)


def run(s: Settings, ids=None, limit=None, force=False, workers=None) -> dict:
    s.chunks_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = s.chunks_dir / "_manifest.json"  # paper id -> settings hash it was chunked with
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    h = settings_hash(s)

    pids = sorted(p.stem for p in s.md_dir.glob("*.json") if not p.stem.startswith("_"))
    if ids:
        missing = set(ids) - set(pids)
        if missing:
            raise FileNotFoundError(f"not parsed yet: {sorted(missing)}")
        pids = [p for p in pids if p in set(ids)]
    pids = pids[:limit] if limit else pids

    def fresh(pid: str) -> bool:
        out = s.chunks_dir / f"{pid}.jsonl"
        return manifest.get(pid) == h and out.exists() and out.stat().st_mtime >= (s.md_dir / f"{pid}.md").stat().st_mtime

    todo = [p for p in pids if force or not fresh(p)]
    workers = max(1, min(workers or s.chunk_workers, len(todo)))
    if todo:
        token_counter(s.embed_model)  # download the tokenizer once, before workers start...
        os.environ["HF_HUB_OFFLINE"] = "1"  # ...so workers (which inherit env) load it from cache, no network

    stats = {"chunked": 0, "skipped": len(pids) - len(todo), "failed": 0, "chunks": 0, "tokens": 0, "workers": workers}
    t0 = time.perf_counter()
    try:
        with tqdm(total=len(todo), unit="paper", disable=not todo) as bar:
            for r in _results(todo, s, workers):
                if r["ok"]:
                    stats["chunked"] += 1
                    stats["chunks"] += r["chunks"]
                    stats["tokens"] += r["tokens"]
                    manifest[r["id"]] = h
                else:
                    stats["failed"] += 1
                    tqdm.write(f"FAILED {r['id']}: {r['error']}")
                bar.update()
                bar.set_postfix(chunks=stats["chunks"], failed=stats["failed"])
    finally:
        atomic_write(manifest_path, json.dumps(manifest, indent=0, sort_keys=True))
    stats["seconds"] = round(time.perf_counter() - t0, 1)
    return stats
