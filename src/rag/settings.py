import os
from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=ROOT / ".env", extra="ignore")

    pdf_dir: Path = Path("data/raw/pdf")
    md_dir: Path = Path("data/raw/md")
    chunks_dir: Path = Path("data/processed/chunks")
    hf_home: Path | None = None  # Hugging Face download cache (tokenizer/model); C: is nearly full, set in .env

    parse_layout: bool = True
    parse_workers: int = max(1, (os.cpu_count() or 3) - 2)  # benchmarked: more workers = faster on 28 threads
    parse_ort_threads: int = 1  # ONNX threads per worker process (layout model)

    embed_model: str = "Qwen/Qwen3-Embedding-0.6B"  # its tokenizer also measures chunk sizes
    chunk_max_tokens: int = 512  # hard cap per chunk, header included
    chunk_overlap_tokens: int = 64  # only applied when a section is split
    chunk_min_tokens: int = 128  # sections smaller than this merge with a sibling
    chunk_drop_tokens: int = 24  # leftover fragments below this are dropped
    chunk_workers: int = max(1, (os.cpu_count() or 3) - 2)

    embeddings_dir: Path = Path("data/processed/embeddings")  # cached vectors, so re-indexing skips the GPU
    embed_batch_size: int = 32  # chunks per GPU forward pass; lower it if you hit out-of-memory
    # Qwen3-Embedding is asymmetric: queries get an instruction, documents don't.
    embed_query_prompt: str = (
        "Instruct: Given a question about AI research, retrieve passages from arXiv papers that answer it\nQuery:"
    )

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "arxiv_chunks"
    index_papers_per_batch: int = 16  # papers embedded together (bigger GPU batches, less padding)
    upsert_batch: int = 128  # points per Qdrant request
    prefetch_limit: int = 50  # candidates each retriever (dense, BM25) contributes before fusion

    # Stage 4: answer questions
    rewrite_model: str = "qwen3:8b"  # LLM for query rewrite + metadata extraction (qwen3:4b is faster)
    rewrite_alt_queries: int = 2  # extra phrasings searched besides the question and its rewrite
    rerank_model: str = "Qwen/Qwen3-Reranker-0.6B"
    rerank_instruction: str = "Given a question about AI research, judge whether this passage from an arXiv paper helps answer it"
    rerank_candidates: int = 50  # fused hybrid results the reranker re-scores
    rerank_batch_size: int = 16
    answer_top_k: int = 8  # best reranked chunks offered to the context builder
    context_max_tokens: int = 3500  # token budget for sources in the LLM prompt
    context_min_score: float = 0.05  # reranker relevance (0..1) below which a chunk never reaches the LLM
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:8b"
    ollama_num_ctx: int = 8192  # context window: 6 chunks (~2.5k tokens) + instructions + answer
    ollama_temperature: float = 0.2  # low: stick to the sources
    ollama_think: bool = False  # Qwen3 "thinking" mode: better reasoning, much slower
    ollama_keep_alive: str = "0"  # unload the LLM after answering: it and our models don't fit in 8 GB together
    # free the embedder/reranker before the LLM runs. Set false (+ a small LLM like qwen3:4b and keep_alive=30m)
    # to keep everything loaded in the server: much faster, if it all fits in VRAM
    release_gpu_before_llm: bool = True

    eval_dir: Path = Path("data/eval")  # test questions (synthetic.jsonl, manual.jsonl) and results/

    @field_validator("pdf_dir", "md_dir", "chunks_dir", "hf_home", "embeddings_dir", "eval_dir")
    @classmethod
    def _resolve(cls, p: Path | None) -> Path | None:
        return p if p is None or p.is_absolute() else ROOT / p


@lru_cache
def get_settings() -> Settings:
    return Settings()
