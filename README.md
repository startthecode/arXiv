# arXiv RAG

A fully local Retrieval-Augmented Generation (RAG) system over **1,050 arXiv AI papers**. Ask a question in a
chat page and get an answer that cites the exact paper, section and pages it came from. No API keys: every
model runs on your own machine.

**Pipeline highlights**

- **Structure-aware + recursive chunking.** Section trees are rebuilt from the PDF outline and heading numbering, and References and acknowledgments are dropped. Oversized sections are split paragraph → sentence → word, at ≤ 512 tokens with overlap.
- **Hybrid search** in Qdrant: dense vectors (Qwen3-Embedding-0.6B) plus BM25, fused with Reciprocal Rank Fusion.
- **Query understanding:** an LLM rewrites the query and extracts metadata (paper titles, authors, arXiv ids). Mentions are verified against the paper catalog before they become metadata filters.
- **Reranking** with Qwen3-Reranker-0.6B, then a **context builder** applies a relevance gate and a token budget, groups chunks by paper and merges neighbouring chunks.
- **Local LLM answers** via Ollama (qwen3:8b), with `[n]` citations.
- **Evaluation:** Accuracy, Precision, Recall, MRR and nDCG for every retrieval configuration.
- **Web app:** FastAPI + Server-Sent Events + a plain HTML/JS chat page that streams the plan, sources and answer.

---

## Dataset

The corpus is **1,050 arXiv PDFs** (arXiv ids `2610.*`, AI and computer-science papers; about 2.1 GB and 23.9k pages). The PDFs are not stored in this repository.

**Download:** https://mega.nz/folder/Gk4ESQIT#DRoUJsZimiAvnA5cOlNdrg

Put the PDFs in `data/raw/pdf/` (create the folder), so you end up with `data/raw/pdf/2610.13563.pdf` and so on.
Each paper remains under the license chosen by its authors on arXiv. Any other folder of PDFs works too: the file name (without `.pdf`) becomes the paper id.

---

## How it works

```
INGESTION (offline)
  PDFs ──parse──► page-tagged markdown + metadata ──chunk──► 60,477 chunks ──index──► Qdrant
        pymupdf4llm                               section tree + recursive     dense (1024-d, HNSW)
        (ML layout model)                         splitting (≤ 512 tokens)     + BM25 (inverted index)

QUERY (per question)
  question ─► rewrite + metadata extraction (LLM, JSON) ─► verified filter (paper_id / section)
           ─► hybrid search for each query variant ─► RRF fusion ─► rerank top 50 (cross-encoder)
           ─► context builder (relevance gate, token budget, group + merge) ─► LLM answer with [n] citations
```

| Layer | Tool |
|---|---|
| PDF parsing | [pymupdf4llm](https://github.com/pymupdf/pymupdf4llm) (layout mode) |
| Chunking | Own structure-aware code + `langchain-text-splitters` (recursive), token counts from `tokenizers` |
| Embeddings | `Qwen/Qwen3-Embedding-0.6B` via sentence-transformers (GPU, bf16) |
| Vector DB + BM25 | [Qdrant](https://qdrant.tech) in Docker (server-side BM25 with IDF) |
| Reranker | `Qwen/Qwen3-Reranker-0.6B` via sentence-transformers `CrossEncoder` |
| LLM | [Ollama](https://ollama.com) with `qwen3:8b` (query rewrite + answers) |
| Web app | FastAPI + Server-Sent Events + vanilla HTML/JS |
| Tooling | [uv](https://docs.astral.sh/uv/), pydantic-settings, pytest |

---

## Requirements

- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- An NVIDIA GPU with about 8 GB of VRAM. torch is installed from the **CUDA 13** wheel index; see `pyproject.toml`.
- [Docker](https://www.docker.com/) for Qdrant
- [Ollama](https://ollama.com)
- About 20 GB of free disk space: dataset, the CUDA build of torch, models, vectors and caches

Developed and tested on Windows 11 with an RTX 5060 Laptop GPU (8 GB) and 32 GB of RAM.

---

## Setup

```powershell
git clone <this-repo-url>
cd RAG
uv sync                                   # creates .venv (torch with CUDA)
Copy-Item .env.example .env               # optional: settings overrides (cp on Linux/macOS)

docker compose up -d                      # Qdrant on http://localhost:6333
ollama pull qwen3:8b                      # local LLM (~5 GB)
```

On the first run, the embedding model and the reranker (about 1.2 GB each) download from Hugging Face. Set
`RAG_HF_HOME` in `.env` to choose where they're stored.

Check that the GPU works:

```powershell
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Build the index

```powershell
uv run rag parse     # PDF → markdown (all CPU cores; ~20 min for 1,050 papers)
uv run rag chunk     # markdown → chunks
uv run rag stats     # chunk quality report
uv run rag index     # embeddings (GPU) + BM25 → Qdrant
uv run rag smoke     # sanity check: hit@5 per search mode
```

Every stage caches its output and skips finished work. To add papers, drop new PDFs into `data/raw/pdf/` and run the same three commands.

## Ask questions

```powershell
uv run rag serve                                            # chat UI at http://127.0.0.1:8000
uv run rag ask "In the MatQuery paper, how are the databases kept frozen?"   # or in the terminal
uv run rag ask "..." --retrieve-only                        # see the retrieved chunks only
uv run rag search "frozen snapshots" --mode bm25            # raw search: hybrid | dense | bm25
```

All commands and options, settings and troubleshooting: **[COMMANDS.md](COMMANDS.md)**.

---

## Evaluation

```powershell
uv run rag eval-gen -n 100     # LLM writes test questions from random chunks → data/eval/synthetic.jsonl
uv run rag eval                # scores every retrieval configuration
```

Hand-written questions live in `data/eval/manual.jsonl`, one JSON object per line:
`{"question": "...", "paper_id": "2610.13617", "pages": [3, 4]}`.

Here are the results of one run: 60 questions (synthetic + hand-written), **chunk level** (did the exact passage come back?), k = 5.

| Config | Acc@5 | P@5 | R@5 | MRR@5 | nDCG@5 |
|---|---|---|---|---|---|
| dense | 0.767 | 0.243 | 0.361 | 0.676 | 0.513 |
| bm25 | 0.950 | 0.340 | 0.486 | 0.883 | 0.692 |
| hybrid | 0.950 | 0.320 | 0.466 | 0.878 | 0.669 |
| **hybrid + rerank** | **0.967** | **0.337** | **0.489** | **0.921** | **0.711** |
| full (rewrite + filter + rerank) | 0.933 | 0.330 | 0.475 | 0.904 | 0.687 |

At **paper level** (was the right paper retrieved?), hybrid + rerank reaches 1.0 on every metric except precision, which caps at 0.2 because each question has a single relevant paper.
Most questions are synthetic, written from the passage itself, so they favour BM25's exact-word matching. Treat these numbers as a comparison between configurations, not as absolute quality.

---

## Project structure

```
src/rag/
  cli.py        `rag` command: parse | chunk | stats | index | search | smoke | ask | eval-gen | eval | serve
  settings.py   all settings (env vars RAG_*, .env)
  parse.py      Stage 1: PDF → page-tagged markdown + metadata
  clean.py      text normalization, bibliography detection
  sections.py   heading levels, section tree
  chunk.py      tiny-section merging, recursive splitting, chunk records
  stats.py      chunk quality report
  embed.py      embeddings (Qwen3-Embedding)
  index.py      Qdrant collection + upload (dense + BM25)
  search.py     dense / BM25 / hybrid search, smoke check
  query.py      query rewrite, metadata extraction, filters
  rerank.py     Qwen3-Reranker
  context.py    context builder
  answer.py     retrieval pipeline + Ollama generation (ask_events)
  eval.py       test-set generation + evaluation
  metrics.py    Accuracy, Precision, Recall, MRR, nDCG
  server.py     FastAPI app (SSE)
  static/       chat page
tests/          pytest suite (offline: no GPU, Qdrant or Ollama needed)
data/
  raw/pdf/      the dataset (download separately, see above)
  eval/         test questions
docker-compose.yml   Qdrant
COMMANDS.md          command cheat sheet
```

Run the tests with `uv run pytest -q`.

---

## Notes

- `pymupdf`, `pymupdf-layout` and `pymupdf4llm` are licensed under **AGPL-3.0** (or a commercial license). Keep that in mind before distributing a product built on this pipeline.
- Each question loads the embedder, the reranker and the LLM. With 8 GB of VRAM they are swapped in and out of GPU memory. On a larger GPU, set `RAG_RELEASE_GPU_BEFORE_LLM=false` and `RAG_OLLAMA_KEEP_ALIVE=30m` for much faster answers.
