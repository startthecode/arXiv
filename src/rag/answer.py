"""Stage 4: query plan -> multi-query hybrid retrieval (filtered) -> rerank -> local LLM answer with citations.

Query rewrite + metadata live in query.py, the context builder in context.py; this module wires them up.
"""

from collections.abc import Callable, Iterator

from rag.query import QueryPlan, to_filter
from rag.settings import Settings

SYSTEM_PROMPT = (
    "You are a research assistant answering questions about arXiv papers.\n"
    "Use ONLY the numbered sources provided. After every claim, cite its source number in square brackets, "
    "e.g. [2] or [1][3].\n"
    "If the sources do not contain the answer, say that you could not find it in the indexed papers; "
    "do not use outside knowledge.\n"
    "Be concise and precise."
)


def rrf_fuse(rankings: list[list[dict]], key: Callable[[dict], str], k: int = 60) -> list[tuple[dict, float]]:
    """Reciprocal Rank Fusion across rankings: score = sum of 1/(k + rank). Uses ranks only, so rankings
    with incomparable scores (different queries) can be combined. A chunk found by several queries wins."""
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, 1):
            kk = key(item)
            scores[kk] = scores.get(kk, 0.0) + 1 / (k + rank)
            items.setdefault(kk, item)
    return sorted(((items[kk], sc) for kk, sc in scores.items()), key=lambda x: x[1], reverse=True)


def retrieve(s: Settings, plan: QueryPlan, rerank: bool = True, top_k: int | None = None) -> list[tuple[dict, float]]:
    """Best chunks as [(payload, score)]: every planned query searched with the metadata filter,
    fused with RRF, then re-scored by the reranker against the ORIGINAL question (the user's real intent)."""
    from rag.search import search

    top_k = top_k or s.answer_top_k

    def run(flt):
        return [[h.payload for h in search(s, q, mode="hybrid", k=s.rerank_candidates, flt=flt)] for q in plan.queries]

    flt = to_filter(plan)
    rankings = run(flt)
    if flt is not None and not any(rankings):
        plan.notes.append("filter matched nothing; searched all papers instead")
        rankings = run(None)
    fused = rrf_fuse(rankings, key=lambda p: p["chunk_id"])[: s.rerank_candidates]
    if not rerank or not fused:
        return fused[:top_k]

    from rag.rerank import rerank as rerank_docs

    docs = [f"{p['breadcrumb']}\n\n{p['text']}" for p, _ in fused]  # give the reranker the section context too
    return [(fused[i][0], score) for i, score in rerank_docs(s, plan.question, docs, top_k)]


def build_messages(question: str, sources: list[dict]) -> list[dict]:
    """Chat messages for the LLM: rules in the system message, numbered sources + question in the user message."""
    blocks = [
        f"[{i}] {p['title']} — {' > '.join(p['section_path'])} "
        f"(arXiv {p['paper_id']}, pp. {p['page_start']}-{p['page_end']})\n{p['text']}"
        for i, p in enumerate(sources, 1)
    ]
    user = "Sources:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def format_sources(sources: list[dict]) -> str:
    return "\n".join(
        f"[{i}] {p['paper_id']} pp. {p['page_start']}-{p['page_end']}  {p['breadcrumb'][:100]}\n    {p['url']}"
        for i, p in enumerate(sources, 1)
    )


def format_retrieved(results: list[tuple[dict, float]]) -> str:
    return "\n".join(
        f"{i}. [{score:.3f}] {p['paper_id']} pp. {p['page_start']}-{p['page_end']}  {p['breadcrumb'][:100]}\n"
        f"   {' '.join(p['text'].split())[:220]}"
        for i, (p, score) in enumerate(results, 1)
    )


def release_gpu() -> None:
    """Free our embedder/reranker VRAM before Ollama loads the LLM: all three don't fit in 8 GB at once."""
    import gc

    import torch

    from rag.embed import load_model
    from rag.rerank import load_reranker

    load_model.cache_clear()
    load_reranker.cache_clear()
    gc.collect()
    torch.cuda.empty_cache()


def generate(s: Settings, messages: list[dict]) -> Iterator[str]:
    """Stream the answer from the local Ollama server, piece by piece."""
    from ollama import Client

    stream = Client(host=s.ollama_url).chat(
        model=s.ollama_model,
        messages=messages,
        stream=True,
        think=s.ollama_think,
        keep_alive=s.ollama_keep_alive,
        options={"num_ctx": s.ollama_num_ctx, "temperature": s.ollama_temperature},
    )
    for part in stream:
        yield part.message.content or ""


def ask_events(s: Settings, question: str, rewrite: bool = True, rerank: bool = True) -> Iterator[tuple[str, object]]:
    """The whole question pipeline as a stream of (event, data). The CLI prints these; the web server
    forwards them to the browser. One implementation, two interfaces.

      status     str                    progress ("Searching ...") before each slow step
      plan       QueryPlan              rewrite + metadata extraction + filter
      retrieved  [(payload, score)]     reranked chunks
      sources    [source dict]          after the context builder ([] = nothing relevant enough)
      messages   [chat message]         the exact prompt sent to the LLM
      token      str                    answer pieces, as Ollama streams them
    """
    from rag import context, query
    from rag.chunk import token_counter

    yield "status", "Rewriting the question with the LLM..." if rewrite else "Planning the search..."
    plan = query.plan_query(s, question, rewrite=rewrite)
    yield "status", f"Searching {len(plan.queries)} queries (dense + BM25)" + (", then reranking..." if rerank else "...")
    results = retrieve(s, plan, rerank=rerank)
    yield "plan", plan  # after retrieve: it may add a "filter matched nothing" note
    yield "retrieved", results
    sources = context.build_context(results, s, token_counter(s.embed_model), use_scores=rerank)
    yield "sources", sources
    if not sources:
        return  # nothing passed the relevance gate: don't let the LLM improvise
    messages = build_messages(question, sources)
    yield "messages", messages
    yield "status", f"Loading {s.ollama_model} and writing the answer..."
    if s.release_gpu_before_llm:
        release_gpu()
    for piece in generate(s, messages):
        yield "token", piece
