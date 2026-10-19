"""Stage 1: PDF -> page-tagged markdown + JSON sidecar.

No cleanup heuristics here on purpose: dehyphenation, heading re-leveling and
reference removal live in the chunk stage, so they can change without re-parsing.
"""

import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path

import onnxruntime as ort
import pymupdf

from rag.files import atomic_write
from rag.settings import Settings, get_settings

# pymupdf-layout builds its ONNX sessions at import time with ORT's default of one
# thread per core, so N worker processes thrash N x cores threads. Cap them before
# pymupdf4llm is imported (it activates the layout model on import).
_ORTSessionOptions = ort.SessionOptions


def _capped_session_options():
    so = _ORTSessionOptions()
    so.intra_op_num_threads = get_settings().parse_ort_threads
    so.inter_op_num_threads = 1
    return so


ort.SessionOptions = _capped_session_options

import pymupdf4llm  # noqa: E402
from tqdm import tqdm  # noqa: E402

MD_KWARGS = dict(page_chunks=True, header=False, footer=False, use_ocr=False, force_text=True, show_progress=False)


def opts_hash(layout: bool) -> str:
    key = json.dumps([pymupdf4llm.VERSION, layout, MD_KWARGS], sort_keys=True)
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def discover(pdf_dir: Path, ids: list[str] | None = None, limit: int | None = None) -> list[Path]:
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if ids:
        missing = set(ids) - {p.stem for p in pdfs}
        if missing:
            raise FileNotFoundError(f"no PDF for ids: {sorted(missing)}")
        pdfs = [p for p in pdfs if p.stem in set(ids)]
    return pdfs[:limit] if limit else pdfs


def is_cached(pdf: Path, md_dir: Path, h: str) -> bool:
    sidecar = md_dir / f"{pdf.stem}.json"
    if not (sidecar.exists() and (md_dir / f"{pdf.stem}.md").exists()):
        return False
    try:
        return json.loads(sidecar.read_text(encoding="utf-8"))["parser"]["opts_hash"] == h
    except (ValueError, KeyError):
        return False


def _init_worker(layout: bool) -> None:
    pymupdf.TOOLS.mupdf_display_errors(False)
    pymupdf.TOOLS.mupdf_display_warnings(False)
    if not layout:
        pymupdf4llm.use_layout(False)


def parse_one(pdf: Path, md_dir: Path, layout: bool) -> dict:
    t0 = time.perf_counter()
    try:
        pymupdf.TOOLS.reset_mupdf_warnings()
        with pymupdf.open(pdf) as doc:
            toc = doc.get_toc()
            kwargs = dict(MD_KWARGS)
            if not layout:  # legacy mode can take heading levels from the PDF outline
                usable = len(toc) >= 3 and max(t[0] for t in toc) <= 4
                kwargs["hdr_info"] = (
                    pymupdf4llm.TocHeaders(doc) if usable else pymupdf4llm.IdentifyHeaders(doc, max_levels=4)
                )
            pages = pymupdf4llm.to_markdown(doc, **kwargs)
            meta = doc.metadata or {}
            page_count = doc.page_count

        md = "".join(f"<!-- page: {i} -->\n\n{p['text'].strip()}\n\n" for i, p in enumerate(pages, 1))
        seconds = round(time.perf_counter() - t0, 2)
        sidecar = {
            "paper_id": pdf.stem,
            "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
            "title": (meta.get("title") or "").strip(),
            "authors": [a.strip() for a in (meta.get("author") or "").split(";") if a.strip()],
            "keywords": [k.strip() for k in (meta.get("keywords") or "").replace(";", ",").split(",") if k.strip()],
            "subject": (meta.get("subject") or "").strip(),
            "page_count": page_count,
            "toc": toc,
            "pages": [{"page": i, "md_chars": len(p["text"])} for i, p in enumerate(pages, 1)],
            "parser": {"pymupdf4llm": pymupdf4llm.VERSION, "layout": layout, "opts_hash": opts_hash(layout)},
            "seconds": seconds,
            "mupdf_warnings": pymupdf.TOOLS.mupdf_warnings()[:2000],
        }
        atomic_write(md_dir / f"{pdf.stem}.md", md)
        atomic_write(md_dir / f"{pdf.stem}.json", json.dumps(sidecar, ensure_ascii=False, indent=1))  # json last = done marker
        return {"id": pdf.stem, "ok": True, "pages": page_count, "seconds": seconds}
    except Exception as e:
        return {"id": pdf.stem, "ok": False, "error": f"{type(e).__name__}: {e}", "seconds": round(time.perf_counter() - t0, 2)}


def _results(todo: list[Path], s: Settings, workers: int):
    if workers <= 1:
        _init_worker(s.parse_layout)
        for pdf in todo:
            yield parse_one(pdf, s.md_dir, s.parse_layout)
        return

    os.environ.setdefault("OMP_NUM_THREADS", "1")  # inherited by workers; avoids thread oversubscription
    pool = ProcessPoolExecutor(
        max_workers=workers,
        mp_context=get_context("spawn"),
        initializer=_init_worker,
        initargs=(s.parse_layout,),
        # no max_tasks_per_child: on Python 3.12 the pool hangs once every worker hits the
        # limit (replacements never spawn). Worker RSS stays ~0.5 GB, so recycling isn't needed.
    )
    try:
        futures = {pool.submit(parse_one, pdf, s.md_dir, s.parse_layout): pdf for pdf in todo}
        for f in as_completed(futures):
            try:
                yield f.result()
            except Exception as e:  # e.g. BrokenProcessPool if a worker dies inside MuPDF
                yield {"id": futures[f].stem, "ok": False, "error": f"{type(e).__name__}: {e}"}
    finally:
        pool.shutdown(cancel_futures=True)  # on Ctrl+C, don't wait for the whole queue


def run(s: Settings, ids=None, limit=None, force=False, workers=None) -> dict:
    s.md_dir.mkdir(parents=True, exist_ok=True)
    h = opts_hash(s.parse_layout)
    pdfs = discover(s.pdf_dir, ids, limit)
    todo = [p for p in pdfs if force or not is_cached(p, s.md_dir, h)]
    todo.sort(key=lambda p: p.stat().st_size, reverse=True)  # big files first for better load balance
    workers = max(1, min(workers or s.parse_workers, len(todo)))

    stats = {"parsed": 0, "skipped": len(pdfs) - len(todo), "failed": 0, "pages": 0, "workers": workers}
    run_ts = datetime.now().isoformat(timespec="seconds")
    t0 = time.perf_counter()
    with tqdm(total=len(todo), unit="pdf", disable=not todo) as bar:
        for r in _results(todo, s, workers):
            if r["ok"]:
                stats["parsed"] += 1
                stats["pages"] += r["pages"]
            else:
                stats["failed"] += 1
                with open(s.md_dir / "_failures.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps({"run": run_ts, **r}, ensure_ascii=False) + "\n")
            bar.update()
            bar.set_postfix(pages=stats["pages"], failed=stats["failed"])
    stats["seconds"] = round(time.perf_counter() - t0, 1)
    stats["pages_per_s"] = round(stats["pages"] / stats["seconds"], 2) if stats["seconds"] else 0.0
    return stats
