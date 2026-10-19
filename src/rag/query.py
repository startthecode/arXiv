"""Query understanding: rewrite the question for search, extract metadata, turn it into a Qdrant filter.

question ─► arXiv ids (regex) ─────────────────────────────┐
         └► LLM, JSON output: search_query, alt_queries,    ├─► resolve against the paper catalog ─► filter
                              paper_titles, authors, section┘

The LLM only *proposes* mentions. Deterministic code checks them against the real metadata from Stage 1,
so a hallucinated title or author can never become a filter.
"""

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from qdrant_client import models

from rag.settings import Settings

ARXIV_ID = re.compile(r"\b(\d{4}\.\d{4,5})(?:v\d+)?\b")
MAX_TITLE_MATCHES = 5  # a "title" matching more papers than this is a topic, not a paper
MAX_AUTHOR_MATCHES = 20  # an author mention matching more papers than this is too vague to filter on

REWRITE_PROMPT = """You turn questions about a library of arXiv AI papers into search instructions. Reply with JSON only.
- search_query: the question rewritten as one standalone, specific search query. Keep technical terms, names, numbers and acronyms exactly; add an acronym's expansion only if you are sure of it.
- alt_queries: up to {n} alternative phrasings that use different words for the same information need.
- paper_titles: titles or distinctive title words of specific papers the question names (e.g. "in the MatQuery paper" -> "MatQuery"); [] if none.
- authors: person names the question mentions as authors; [] if none.
- section: "appendix" if the question asks about appendix material, "abstract" if it asks for a paper's abstract, otherwise "any".
Do not answer the question."""


class QueryAnalysis(BaseModel):
    """What the LLM must return (its JSON schema is sent to Ollama as the required output format)."""

    search_query: str
    alt_queries: list[str] = Field(default_factory=list)
    paper_titles: list[str] = Field(default_factory=list)
    authors: list[str] = Field(default_factory=list)
    section: Literal["any", "abstract", "appendix"] = "any"


class QueryPlan(BaseModel):
    """Everything retrieval needs: which queries to run and which filter to apply."""

    question: str
    queries: list[str]  # original question first, then rewrites (deduplicated)
    paper_ids: list[str] = Field(default_factory=list)  # empty = search all papers
    section: Literal["any", "abstract", "appendix"] = "any"
    notes: list[str] = Field(default_factory=list)  # what was extracted or ignored, shown to the user


def analyze_with_llm(s: Settings, question: str) -> QueryAnalysis:
    from ollama import Client

    reply = Client(host=s.ollama_url).chat(
        model=s.rewrite_model,
        messages=[
            {"role": "system", "content": REWRITE_PROMPT.format(n=s.rewrite_alt_queries)},
            {"role": "user", "content": question},
        ],
        format=QueryAnalysis.model_json_schema(),  # structured output: the reply must match this schema
        think=False,
        keep_alive=s.ollama_keep_alive,
        options={"temperature": 0, "num_ctx": 2048},
    )
    return QueryAnalysis.model_validate_json(reply.message.content)


@lru_cache
def load_catalog(md_dir: str) -> list[dict]:
    """Paper-level metadata extracted in Stage 1 (PDF metadata in the .json sidecars)."""
    rows = []
    for f in sorted(Path(md_dir).glob("*.json")):
        if not f.stem.startswith("_"):
            m = json.loads(f.read_text(encoding="utf-8"))
            rows.append({"paper_id": m["paper_id"], "title": m.get("title") or "", "authors": m.get("authors") or []})
    return rows


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def resolve_papers(
    titles: list[str], authors: list[str], ids: list[str], catalog: list[dict]
) -> tuple[list[str], list[str]]:
    """Map mentions to real paper ids. Returns (paper_ids, notes); unmatched or vague mentions are ignored."""
    known = {p["paper_id"] for p in catalog}
    found, notes = [], []
    for pid in ids:
        if pid in known:
            found.append(pid)
            notes.append(f"arXiv id {pid} -> filter")
        else:
            notes.append(f"arXiv id {pid} is not in the library, ignored")

    for t in titles:
        nt = _norm(t)
        hits = [p["paper_id"] for p in catalog if len(nt) >= 4 and f" {nt} " in f" {_norm(p['title'])} "]
        if not hits:
            notes.append(f"title '{t}' not found, ignored")
        elif len(hits) > MAX_TITLE_MATCHES:
            notes.append(f"title '{t}' matches {len(hits)} papers (a topic, not a paper), ignored")
        else:
            found += hits
            notes.append(f"title '{t}' -> {', '.join(hits)}")

    for a in authors:
        tokens = _norm(a).split()
        hits = [
            p["paper_id"]
            for p in catalog
            if tokens and any(all(t in _norm(name).split() for t in tokens) for name in p["authors"])
        ]
        if not hits:
            notes.append(f"author '{a}' not found, ignored")
        elif len(hits) > MAX_AUTHOR_MATCHES:
            notes.append(f"author '{a}' matches {len(hits)} papers (too vague), ignored")
        else:
            found += hits
            notes.append(f"author '{a}' -> {len(hits)} paper(s)")
    return list(dict.fromkeys(found)), notes


def plan_query(s: Settings, question: str, rewrite: bool = True) -> QueryPlan:
    queries, titles, authors, section, notes = [question], [], [], "any", []
    if rewrite:
        try:
            a = analyze_with_llm(s, question)
            queries += [a.search_query, *a.alt_queries[: s.rewrite_alt_queries]]
            titles, authors, section = a.paper_titles, a.authors, a.section
        except Exception as e:  # Ollama down, bad JSON, ...: fall back to the plain question
            notes.append(f"query rewrite skipped ({type(e).__name__}: {e})")
    paper_ids, more = resolve_papers(titles, authors, ARXIV_ID.findall(question), load_catalog(str(s.md_dir)))
    seen, unique = set(), []
    for q in queries:  # dedupe on a normalized form: "...frozen?" and "...frozen" are the same search
        if (key := _norm(q)) and key not in seen:
            seen.add(key)
            unique.append(q.strip())
    queries = unique
    return QueryPlan(question=question, queries=queries, paper_ids=paper_ids, section=section, notes=notes + more)


def to_filter(plan: QueryPlan) -> models.Filter | None:
    """Metadata filter applied inside both retrievers (uses the payload indexes on paper_id, section_type)."""
    must = []
    if plan.paper_ids:
        must.append(models.FieldCondition(key="paper_id", match=models.MatchAny(any=plan.paper_ids)))
    if plan.section != "any":
        must.append(models.FieldCondition(key="section_type", match=models.MatchValue(value=plan.section)))
    return models.Filter(must=must) if must else None


def describe(plan: QueryPlan) -> str:
    lines = ["Queries:"] + [f"  {i}) {q}" for i, q in enumerate(plan.queries, 1)]
    filters = []
    if plan.paper_ids:
        filters.append(f"papers {', '.join(plan.paper_ids)}")
    if plan.section != "any":
        filters.append(f"section = {plan.section}")
    lines.append("Filter:  " + ("; ".join(filters) if filters else "none (all papers)"))
    lines += [f"  - {n}" for n in plan.notes]
    return "\n".join(lines)
