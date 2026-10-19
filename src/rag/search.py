"""Query the index: dense, BM25, or hybrid (both fused with RRF). Used to verify ingestion."""

import json
import random
import re

from qdrant_client import models

from rag.embed import embed_query
from rag.index import DENSE, SPARSE, bm25_doc, get_client
from rag.settings import Settings

MODES = ("hybrid", "dense", "bm25")


def search(
    s: Settings,
    query: str,
    mode: str = "hybrid",
    k: int = 5,
    paper_id: str | None = None,
    flt: models.Filter | None = None,  # metadata filter, applied inside each retriever (not after)
):
    qc = get_client(s)
    if paper_id:
        flt = models.Filter(must=[models.FieldCondition(key="paper_id", match=models.MatchValue(value=paper_id))])
    common = dict(limit=k, with_payload=True)
    if mode == "bm25":
        return qc.query_points(s.qdrant_collection, query=bm25_doc(query), using=SPARSE, query_filter=flt, **common).points
    qv = embed_query(query, s).tolist()
    if mode == "dense":
        return qc.query_points(s.qdrant_collection, query=qv, using=DENSE, query_filter=flt, **common).points
    # hybrid: each retriever proposes its top candidates, then Reciprocal Rank Fusion merges the two rankings
    return qc.query_points(
        s.qdrant_collection,
        prefetch=[
            models.Prefetch(query=qv, using=DENSE, limit=s.prefetch_limit, filter=flt),
            models.Prefetch(query=bm25_doc(query), using=SPARSE, limit=s.prefetch_limit, filter=flt),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        **common,
    ).points


def format_hits(hits) -> str:
    lines = []
    for rank, h in enumerate(hits, 1):
        p = h.payload
        snippet = " ".join(p["text"].split())[:220]
        lines.append(
            f"{rank}. [{h.score:.3f}] {p['paper_id']} p{p['page_start']}-{p['page_end']}  {p['breadcrumb'][:110]}\n   {snippet}"
        )
    return "\n".join(lines)


def smoke(s: Settings, n: int = 50, k: int = 5, seed: int = 0) -> str:
    """Known-item check: take a sentence from a random chunk, search for it, and see whether that
    chunk (or its neighbour) comes back in the top k. Every mode should score high on this easy task;
    a low score means the index, not the retriever, is broken."""
    rng = random.Random(seed)
    files = sorted(s.chunks_dir.glob("*.jsonl"))
    probes = []
    while len(probes) < n:
        recs = [json.loads(line) for line in rng.choice(files).read_text(encoding="utf-8").splitlines()]
        r = rng.choice(recs)
        sentences = [x for x in re.split(r"(?<=[.!?])\s+", r["text"]) if 12 <= len(x.split()) <= 40]
        if sentences:
            probes.append((rng.choice(sentences), r))

    hits = {m: 0 for m in MODES}
    for sentence, r in probes:
        for m in MODES:
            found = search(s, sentence, mode=m, k=k)
            if any(
                h.payload["paper_id"] == r["paper_id"] and abs(h.payload["chunk_index"] - r["chunk_index"]) <= 1
                for h in found
            ):
                hits[m] += 1
    return "  ".join(f"{m} hit@{k}: {hits[m] / n:.0%}" for m in MODES) + f"   ({n} probes)"
