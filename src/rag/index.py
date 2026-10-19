"""Stage 3: chunks -> dense vectors (GPU) + BM25 sparse vectors -> one Qdrant collection.

Per chunk, Qdrant stores one "point":
  vector "dense": 1024 floats from Qwen3-Embedding (semantic search)
  vector "bm25":  sparse {term: weight}, computed by the Qdrant server from the text (keyword search)
  payload:        text + metadata (paper, section, pages) returned with search results
"""

import hashlib
import json
import time
import uuid
from itertools import batched

import numpy as np
from qdrant_client import QdrantClient, models
from tqdm import tqdm

from rag.files import atomic_write
from rag.settings import Settings

DENSE, SPARSE = "dense", "bm25"
BM25_MODEL = "Qdrant/bm25"
POINT_NAMESPACE = uuid.UUID("6f1f6a3e-2b7c-4d1a-9a52-3c1f0e7b9d10")  # fixed: same chunk_id -> same point id
PAYLOAD_FIELDS = (
    "chunk_id", "paper_id", "chunk_index", "title", "authors", "section_path", "breadcrumb",
    "section_level", "section_type", "section_top", "page_start", "page_end", "text",
)  # fmt: skip


def get_client(s: Settings) -> QdrantClient:
    return QdrantClient(url=s.qdrant_url, timeout=120)


def qdrant_problem(s: Settings) -> str | None:
    """None if the Qdrant server answers; otherwise a human-readable fix (instead of a long traceback)."""
    try:
        get_client(s).get_collections()
        return None
    except Exception as e:
        return (
            f"Cannot reach Qdrant at {s.qdrant_url} ({type(e).__name__}).\n"
            "Is Docker Desktop running? Then start the server with:  docker compose up -d"
        )


def bm25_doc(text: str) -> models.Document:
    """Raw text tagged for Qdrant's built-in BM25: the server tokenizes, stems and weights it.
    Must be built the same way at index and query time."""
    return models.Document(text=text, model=BM25_MODEL)


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(POINT_NAMESPACE, chunk_id))


def payload(r: dict) -> dict:
    return {k: r[k] for k in PAYLOAD_FIELDS} | {"url": f"https://arxiv.org/abs/{r['paper_id']}"}


def ensure_collection(qc: QdrantClient, s: Settings, dim: int, recreate: bool) -> bool:
    """Create the collection if needed. Returns True when it was (re)created empty."""
    name = s.qdrant_collection
    if recreate and qc.collection_exists(name):
        qc.delete_collection(name)
    if qc.collection_exists(name):
        size = qc.get_collection(name).config.params.vectors[DENSE].size
        if size != dim:
            raise RuntimeError(f"collection {name!r} has {size}-d vectors, model gives {dim}-d: use --recreate")
        return False
    qc.create_collection(
        name,
        vectors_config={DENSE: models.VectorParams(size=dim, distance=models.Distance.COSINE)},
        sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},  # IDF from corpus stats
    )
    for field in ("paper_id", "section_type"):  # fast filtering: "only this paper", "skip appendix"
        qc.create_payload_index(name, field, models.PayloadSchemaType.KEYWORD)
    return True


def _paper_filter(pid: str) -> models.Filter:
    return models.Filter(must=[models.FieldCondition(key="paper_id", match=models.MatchValue(value=pid))])


def _vectors(pid: str, sha: str, records: list[dict], s: Settings) -> np.ndarray | None:
    """Cached embeddings for this exact version of the paper's chunks, if any."""
    f = s.embeddings_dir / s.embed_model.replace("/", "__") / f"{pid}-{sha[:12]}.npy"
    if f.exists():
        vecs = np.load(f)
        if len(vecs) == len(records):
            return vecs.astype(np.float32)
    return None


def _save_vectors(pid: str, sha: str, vecs: np.ndarray, s: Settings) -> None:
    folder = s.embeddings_dir / s.embed_model.replace("/", "__")
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob(f"{pid}-*.npy"):  # vectors of an older chunking of this paper
        old.unlink()
    np.save(folder / f"{pid}-{sha[:12]}.npy", vecs.astype(np.float16))  # float16 halves disk; plenty precise


def run(s: Settings, ids=None, limit=None, force=False, recreate=False) -> dict:
    from rag.embed import embed_documents, load_model

    pids = sorted(p.stem for p in s.chunks_dir.glob("*.jsonl"))
    if ids:
        missing = set(ids) - set(pids)
        if missing:
            raise FileNotFoundError(f"not chunked yet: {sorted(missing)}")
        pids = [p for p in pids if p in set(ids)]
    pids = pids[:limit] if limit else pids

    qc = get_client(s)
    dim = load_model(s.embed_model).get_embedding_dimension()
    created = ensure_collection(qc, s, dim, recreate)

    # paper id -> sha of the chunk file it was indexed from (a fresh collection starts empty)
    state_path = s.chunks_dir.parent / f"index_{s.qdrant_collection}.json"
    state = {} if created or not state_path.exists() else json.loads(state_path.read_text(encoding="utf-8"))
    shas = {pid: hashlib.sha256((s.chunks_dir / f"{pid}.jsonl").read_bytes()).hexdigest() for pid in pids}
    todo = [pid for pid in pids if force or state.get(pid) != shas[pid]]

    def load(pid: str) -> list[dict]:
        lines = (s.chunks_dir / f"{pid}.jsonl").read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines]

    stats = {"indexed": 0, "skipped": len(pids) - len(todo), "points": 0, "embedded": 0}
    t0 = time.perf_counter()
    total = sum((s.chunks_dir / f"{pid}.jsonl").read_bytes().count(b"\n") for pid in todo)  # one line per chunk
    try:
        with tqdm(total=total, unit="chunk", disable=not todo) as bar:
            for group in batched(todo, s.index_papers_per_batch):
                records = {pid: load(pid) for pid in group}
                vectors = {pid: _vectors(pid, shas[pid], records[pid], s) for pid in group}

                # embed every not-yet-cached chunk of the group in one call: the model sorts by length,
                # so similar-sized chunks share a GPU batch and padding is minimal
                need = [pid for pid in group if vectors[pid] is None]
                texts = [r["embed_text"] for pid in need for r in records[pid]]
                if texts:
                    embs = embed_documents(texts, s)
                    offset = 0
                    for pid in need:
                        n = len(records[pid])
                        vectors[pid] = embs[offset : offset + n]
                        _save_vectors(pid, shas[pid], vectors[pid], s)
                        offset += n
                    stats["embedded"] += len(texts)

                for pid in group:
                    # delete-then-upsert: a re-chunked paper with fewer chunks leaves no stale points
                    qc.delete(s.qdrant_collection, points_selector=models.FilterSelector(filter=_paper_filter(pid)), wait=True)
                    points = [
                        models.PointStruct(
                            id=point_id(r["chunk_id"]),
                            vector={DENSE: vec.tolist(), SPARSE: bm25_doc(r["embed_text"])},
                            payload=payload(r),
                        )
                        for r, vec in zip(records[pid], vectors[pid])
                    ]
                    for batch in batched(points, s.upsert_batch):
                        qc.upsert(s.qdrant_collection, points=list(batch), wait=True)
                    state[pid] = shas[pid]
                    stats["indexed"] += 1
                    stats["points"] += len(points)
                    bar.update(len(points))
                atomic_write(state_path, json.dumps(state, indent=0, sort_keys=True))  # resumable after each group
    finally:
        atomic_write(state_path, json.dumps(state, indent=0, sort_keys=True))

    stats["seconds"] = round(time.perf_counter() - t0, 1)
    stats["collection_points"] = qc.count(s.qdrant_collection, exact=True).count
    return stats
