from fastapi.testclient import TestClient

from rag import server
from rag.query import QueryPlan

SRC = {
    "paper_id": "2610.13573", "title": "MatQuery", "section_path": ["2 Data"], "page_start": 2, "page_end": 3,
    "url": "https://arxiv.org/abs/2610.13573", "text": "Snapshots are frozen.", "score": 0.91,
}  # fmt: skip


def fake_events(s, q, rewrite=True, rerank=True):  # stands in for the real pipeline: no GPU, Qdrant or Ollama
    yield "status", "Searching..."
    yield "plan", QueryPlan(question=q, queries=[q, "rewritten"])
    yield "retrieved", []
    yield "sources", [SRC]
    yield "messages", []
    yield "token", "Frozen copies [1]."


def test_index_serves_the_chat_page():
    r = TestClient(server.app).get("/")
    assert r.status_code == 200 and "EventSource" in r.text


def test_ask_streams_sse_events_in_order(monkeypatch):
    monkeypatch.setattr(server, "ask_events", fake_events)
    r = TestClient(server.app).get("/api/ask", params={"q": "how are dbs frozen?"})
    assert r.headers["content-type"].startswith("text/event-stream")
    events = [line.removeprefix("event: ") for line in r.text.splitlines() if line.startswith("event: ")]
    assert events == ["status", "plan", "sources", "token", "done"]  # CLI-only events are not forwarded
    assert '"n": 1' in r.text and '"pages": "2-3"' in r.text
    assert 'data: "Frozen copies [1]."' in r.text


def test_ask_reports_failures_as_events(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("Ollama is down")
        yield  # makes this a generator, like the real ask_events

    monkeypatch.setattr(server, "ask_events", broken)
    r = TestClient(server.app).get("/api/ask", params={"q": "anything at all"})
    assert "event: fail" in r.text and "Ollama is down" in r.text
    assert server._gpu.acquire(blocking=False)  # the lock was released despite the error
    server._gpu.release()


def test_ask_rejects_too_short_question():
    assert TestClient(server.app).get("/api/ask", params={"q": "x"}).status_code == 422  # validated by FastAPI
