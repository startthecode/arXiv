"""Web app: a FastAPI server that keeps Python, CUDA and the models warm between questions, plus a chat page.

  GET /                 static/index.html: the chat UI
  GET /api/health       are Qdrant and Ollama reachable?
  GET /api/ask?q=...    Server-Sent Events stream:  plan -> sources -> token, token, ... -> done  (or fail)

Server-Sent Events (SSE): one HTTP response that stays open; the server writes small text events into it
as they happen ("event: <name>\\ndata: <json>\\n\\n"), and the browser's EventSource receives them live.
"""

import json
import threading
import time
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, StreamingResponse

from rag.answer import ask_events
from rag.settings import get_settings

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="arXiv RAG")
_gpu = threading.Lock()  # one question at a time: a single 8 GB GPU can't run two pipelines at once


def sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def public_source(n: int, p: dict) -> dict:
    """What the browser needs to show one source card."""
    return {
        "n": n,
        "paper_id": p["paper_id"],
        "title": p["title"],
        "section": " > ".join(p["section_path"]),
        "pages": f"{p['page_start']}-{p['page_end']}",
        "url": p["url"],
        "score": round(p.get("score", 0.0), 3),
        "text": p["text"],
    }


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    from ollama import Client

    from rag.index import qdrant_problem

    s = get_settings()
    try:
        Client(host=s.ollama_url).list()
        ollama_ok = True
    except Exception:
        ollama_ok = False
    return {"qdrant": qdrant_problem(s) is None, "ollama": ollama_ok, "model": s.ollama_model}


@app.get("/api/ask")
def ask(q: str = Query(min_length=3, max_length=500), rewrite: bool = True, rerank: bool = True):
    s = get_settings()

    def stream():
        if not _gpu.acquire(blocking=False):  # say so instead of silently waiting for the lock
            yield sse("status", "Waiting for the previous question to finish...")
            _gpu.acquire()
        start = time.perf_counter()
        log = lambda msg: print(f"[ask {time.perf_counter() - start:6.1f}s] {msg}", flush=True)  # noqa: E731
        log(f"question: {q!r}")
        first_token = True
        try:
            for event, data in ask_events(s, q, rewrite=rewrite, rerank=rerank):
                if event == "status":
                    log(data)
                    yield sse("status", data)
                elif event == "plan":
                    log(f"plan: {len(data.queries)} queries, filter papers={data.paper_ids or 'none'}")
                    yield sse("plan", {"queries": data.queries, "paper_ids": data.paper_ids,
                                       "section": data.section, "notes": data.notes})  # fmt: skip
                elif event == "sources":
                    log(f"context: {len(data)} sources")
                    yield sse("sources", [public_source(i, p) for i, p in enumerate(data, 1)])
                    if not data:
                        yield sse("fail", f"No passage scored above {s.context_min_score} relevance; try rephrasing.")
                elif event == "token":
                    if first_token:
                        log("first answer token")
                        first_token = False
                    yield sse("token", data)
                # "retrieved" and "messages" are only for the CLI
            log("done")
            yield sse("done", {})
        except Exception as e:  # Qdrant/Ollama down, model missing, ...: tell the browser instead of hanging
            log(f"FAILED: {type(e).__name__}: {e}")
            yield sse("fail", f"{type(e).__name__}: {e}")
        finally:  # runs on success, error, AND when the browser disconnects mid-answer
            _gpu.release()

    # a plain (sync) generator: Starlette runs it in a worker thread, so slow GPU work doesn't block the server
    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
