from rag.query import ARXIV_ID, QueryAnalysis, QueryPlan, resolve_papers, to_filter

CATALOG = [
    {"paper_id": "2610.13573", "title": "MatQuery: Execution-Grounded Natural-Language Querying", "authors": ["Ada Lovelace"]},
    {"paper_id": "2610.13563", "title": "Not All Correct Fixes Run Alike", "authors": ["Han Peng", "Xing Hu"]},
    # 25 papers sharing an author name: too vague to filter on
    *[{"paper_id": f"2610.9{i:04d}", "title": f"Paper {i}", "authors": [f"Wei Zhang {i}"]} for i in range(25)],
]  # fmt: skip


def test_arxiv_ids_found_with_or_without_version():
    assert ARXIV_ID.findall("compare 2610.13563 with 2610.13573v2 please") == ["2610.13563", "2610.13573"]


def test_resolve_keeps_real_mentions_and_ignores_bad_ones():
    ids, notes = resolve_papers(
        titles=["MatQuery", "A Paper That Does Not Exist"],
        authors=["Xing Hu", "Wei Zhang"],
        ids=["2610.13563", "2601.00000"],
        catalog=CATALOG,
    )
    assert ids == ["2610.13563", "2610.13573"]  # id, then title; the author hit duplicates the id
    joined = "\n".join(notes)
    assert "2601.00000 is not in the library" in joined
    assert "'A Paper That Does Not Exist' not found" in joined
    assert "'Wei Zhang' matches 25 papers (too vague)" in joined


def test_title_match_is_whole_words():
    ids, _ = resolve_papers(titles=["Query"], authors=[], ids=[], catalog=CATALOG)
    assert ids == []  # "query" must not match inside "MatQuery" or "Querying"


def test_to_filter():
    f = to_filter(QueryPlan(question="q", queries=["q"], paper_ids=["a", "b"], section="appendix"))
    assert f.must[0].key == "paper_id" and f.must[0].match.any == ["a", "b"]
    assert f.must[1].key == "section_type" and f.must[1].match.value == "appendix"
    assert to_filter(QueryPlan(question="q", queries=["q"])) is None


def test_plan_dedupes_near_identical_queries_keeping_order(monkeypatch):
    from rag import query
    from rag.settings import Settings

    fake = QueryAnalysis(search_query="How are DBs frozen", alt_queries=["how are dbs frozen?", "Snapshot design"])
    monkeypatch.setattr(query, "analyze_with_llm", lambda *_: fake)  # no Ollama needed
    monkeypatch.setattr(query, "load_catalog", lambda *_: [])
    plan = query.plan_query(Settings(_env_file=None), "How are DBs frozen?")
    assert plan.queries == ["How are DBs frozen?", "Snapshot design"]  # original first, duplicates dropped


def test_llm_json_parses_with_defaults():
    a = QueryAnalysis.model_validate_json('{"search_query": "f2p tests patch performance"}')
    assert a.alt_queries == [] and a.paper_titles == [] and a.section == "any"
