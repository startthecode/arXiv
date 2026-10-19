import argparse
import os
import sys

from rag.settings import get_settings


def _batch_args(p: argparse.ArgumentParser, what: str) -> None:
    p.add_argument("--ids", nargs="+", help="arXiv ids (file stems) to process")
    p.add_argument("--limit", type=int, help="process only the first N papers (by name)")
    p.add_argument("--force", action="store_true", help=f"re-{what} even if cached")
    p.add_argument("--workers", type=int, help="worker processes (1 = in-process, for debugging)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rag", description="RAG over arXiv papers: ingest, search, ask")
    sub = ap.add_subparsers(dest="cmd", required=True)
    _batch_args(sub.add_parser("parse", help="PDF -> page-tagged markdown + JSON sidecar (data/raw/md)"), "parse")
    _batch_args(sub.add_parser("chunk", help="markdown -> structure-aware chunks (data/processed/chunks)"), "chunk")
    sub.add_parser("stats", help="sanity report over the chunks")

    p = sub.add_parser("index", help="chunks -> embeddings (GPU) + BM25 -> Qdrant")
    p.add_argument("--ids", nargs="+", help="arXiv ids (file stems) to index")
    p.add_argument("--limit", type=int, help="index only the first N papers (by name)")
    p.add_argument("--force", action="store_true", help="re-upload even if already indexed")
    p.add_argument("--recreate", action="store_true", help="delete and recreate the Qdrant collection first")

    p = sub.add_parser("search", help="query the index")
    p.add_argument("query")
    p.add_argument("--mode", choices=["hybrid", "dense", "bm25"], default="hybrid")
    p.add_argument("-k", type=int, default=5, help="number of results")
    p.add_argument("--paper", help="only search inside this arXiv id")

    p = sub.add_parser("smoke", help="known-item check of the index in all three modes")
    p.add_argument("-n", type=int, default=50, help="number of probe sentences")

    p = sub.add_parser("ask", help="answer a question: rewrite -> filter -> retrieve -> rerank -> context -> LLM")
    p.add_argument("question")
    p.add_argument("--no-rewrite", action="store_true", help="skip LLM query rewrite/metadata extraction")
    p.add_argument("--no-rerank", action="store_true", help="skip the reranker (to compare)")
    p.add_argument("--retrieve-only", action="store_true", help="show the reranked chunks; don't generate")
    p.add_argument("--show-context", action="store_true", help="print the exact sources the LLM receives")

    from rag.eval import CONFIGS

    p = sub.add_parser("eval-gen", help="write synthetic test questions from random chunks (local LLM)")
    p.add_argument("-n", type=int, default=100, help="number of questions")
    p.add_argument("--seed", type=int, default=0, help="which chunks get sampled (same seed = same test set)")

    p = sub.add_parser("eval", help="score retrieval configs: accuracy, precision, recall, MRR, nDCG @k")
    p.add_argument("-k", type=int, default=5, help="cut-off: only the top k results count")
    p.add_argument("--configs", nargs="+", choices=CONFIGS, default=list(CONFIGS))
    p.add_argument("--limit", type=int, help="only the first N questions (quick check)")

    p = sub.add_parser("serve", help="web app: API + chat page (default http://127.0.0.1:8000)")
    p.add_argument("--host", default="127.0.0.1", help="127.0.0.1 = this computer only")
    p.add_argument("--port", type=int, default=8000)

    args = ap.parse_args(argv)
    sys.stdout.reconfigure(errors="replace")  # paper text can hold characters a Windows console can't show
    s = get_settings()
    if s.hf_home:  # must be set before huggingface_hub is imported; worker processes inherit it
        os.environ.setdefault("HF_HOME", str(s.hf_home))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # Windows w/o Developer Mode: cache copies files, harmless

    if args.cmd in ("index", "search", "smoke", "ask", "eval", "serve"):  # need the Qdrant server: fail fast, clearly
        from rag.index import qdrant_problem

        if problem := qdrant_problem(s):
            print(problem)
            return 1

    if args.cmd == "parse":
        from rag import parse

        r = parse.run(s, ids=args.ids, limit=args.limit, force=args.force, workers=args.workers)
        print(
            f"parsed {r['parsed']}  skipped {r['skipped']}  failed {r['failed']}  "
            f"pages {r['pages']}  {r['pages_per_s']} pages/s  {r['seconds']}s  workers {r['workers']}"
            f"  (layout={s.parse_layout}, out={s.md_dir})"
        )
        if r["failed"]:
            print(f"failures logged to {s.md_dir / '_failures.jsonl'}")
        return 1 if r["failed"] else 0

    if args.cmd == "chunk":
        from rag import chunk

        r = chunk.run(s, ids=args.ids, limit=args.limit, force=args.force, workers=args.workers)
        print(
            f"chunked {r['chunked']}  skipped {r['skipped']}  failed {r['failed']}  "
            f"chunks {r['chunks']}  tokens {r['tokens']}  {r['seconds']}s  workers {r['workers']}  (out={s.chunks_dir})"
        )
        return 1 if r["failed"] else 0

    if args.cmd == "stats":
        from rag import stats

        print(stats.report(s))

    if args.cmd == "index":
        from rag import index

        r = index.run(s, ids=args.ids, limit=args.limit, force=args.force, recreate=args.recreate)
        print(
            f"indexed {r['indexed']}  skipped {r['skipped']}  points {r['points']}  embedded {r['embedded']}  "
            f"{r['seconds']}s  (collection {s.qdrant_collection!r} now holds {r['collection_points']} points)"
        )

    if args.cmd == "search":
        from rag import search

        print(search.format_hits(search.search(s, args.query, mode=args.mode, k=args.k, paper_id=args.paper)))

    if args.cmd == "smoke":
        from rag import search

        print(search.smoke(s, n=args.n))

    if args.cmd == "ask":
        from rag import answer, query

        sources = []
        try:
            for event, data in answer.ask_events(s, args.question, rewrite=not args.no_rewrite, rerank=not args.no_rerank):
                if event == "status":
                    print(f"... {data}", file=sys.stderr, flush=True)  # progress on stderr: stdout stays clean
                elif event == "plan":
                    print(query.describe(data) + "\n")
                elif event == "retrieved" and args.retrieve_only:
                    print(answer.format_retrieved(data) or "nothing retrieved")
                    return 0
                elif event == "sources":
                    sources = data
                    if not sources:
                        print(f"No passage scored above {s.context_min_score} relevance; not asking the LLM.")
                        return 0
                elif event == "messages" and args.show_context:
                    print(data[1]["content"] + "\n" + "-" * 80)
                elif event == "token":
                    print(data, end="", flush=True)
        except Exception as e:  # Ollama not running, model not pulled, ...
            print(f"\n[error] {type(e).__name__}: {e}\nIs Ollama running, and did you `ollama pull {s.ollama_model}`?")
            return 1
        print("\n\nSources:\n" + answer.format_sources(sources))

    if args.cmd == "eval-gen":
        from rag import eval as evaluation

        path, rejected = evaluation.generate_questions(s, n=args.n, seed=args.seed)
        print(f"wrote {args.n} questions to {path} ({rejected} rejected as malformed)")

    if args.cmd == "eval":
        from rag import eval as evaluation

        print(evaluation.run(s, k=args.k, configs=args.configs, limit=args.limit))

    if args.cmd == "serve":
        # models are already downloaded: load them from the local cache without asking the Hub for updates
        # on every (re)load. Run `rag ask` once first if a model has never been downloaded.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import uvicorn

        from rag.server import app

        print(f"Chat UI: http://{args.host}:{args.port}   (Ctrl+C to stop)")
        uvicorn.run(app, host=args.host, port=args.port)
    return 0
