# Commands: from tests to a running app

Run everything in **PowerShell** from `Z:\Study\AI\RAG`.
`rag` and `rag-ingest` are the same CLI. `uv run rag --help` and `uv run rag <command> --help` list every option.

## Quick start (when everything is already set up)

```powershell
docker compose up -d     # 1. start Qdrant (Docker Desktop must be running)
ollama list              # 2. Ollama is running and qwen3:8b is listed
uv run pytest -q         # 3. tests: expect "43 passed"
uv run rag smoke         # 4. index is healthy: hit@5 around 85% or higher in each mode
uv run rag serve         # 5. open http://127.0.0.1:8000  (Ctrl+C to stop)
```

---

## 0. One-time setup

| Step | Command | Why |
|---|---|---|
| Install Python dependencies | `uv sync` | Creates `.venv` from `uv.lock` (torch comes from the CUDA 13 index) |
| uv cache on Z: | `uv.toml` with `cache-dir = "Z:/.cache/uv"` | Machine-specific, so it lives in `uv.toml` (not committed) instead of `pyproject.toml` |
| Check the GPU | `uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` | Expect `2.14.1+cu130 True` |
| Docker storage on Z: | Docker Desktop → Settings → Resources → Advanced → Disk image location = `Z:\DockerDesktop` | C: is nearly full |
| Start Docker with Windows | Docker Desktop → Settings → General → "Start Docker Desktop when you sign in" | Qdrant then comes back after a reboot |
| Install Ollama | `winget install Ollama.Ollama` | Local LLM server |
| Ollama models on Z: | `setx OLLAMA_MODELS "Z:\.cache\ollama"`, then quit and restart Ollama | Before pulling, so models don't land on C: |
| Download the LLM | `ollama pull qwen3:8b` | ~5 GB; `qwen3:4b` is the faster alternative |
| Local settings | `.env` (already has `RAG_HF_HOME=Z:/.cache/huggingface`) | See `.env.example` for every option |

---

## 1. Start the services

```powershell
docker compose up -d     # start Qdrant in the background
docker ps                # rag-qdrant should show "Up"
ollama list              # proves Ollama is running and shows downloaded models
```

Qdrant dashboard: http://localhost:6333/dashboard (collection `arxiv_chunks`, ~60,477 points).

---

## 2. Test

```powershell
uv run pytest -q                     # all tests (offline: no GPU, Qdrant or Ollama needed)
uv run pytest tests/test_chunk.py -q # one file
uv run pytest -q -k rerank           # only tests whose name matches "rerank"
```

---

## 3. Ingestion: PDF to Qdrant (already done; re-runs skip finished work)

| # | Command | What it does | Output |
|---|---|---|---|
| 1 | `uv run rag parse` | PDF → page-tagged markdown + metadata (26 workers; ~20 min for all 1,050) | `data/raw/md/` |
| 2 | `uv run rag chunk` | Structure-aware + recursive chunks (≤ 512 tokens) | `data/processed/chunks/` |
| 3 | `uv run rag stats` | Chunk quality report | printed |
| 4 | `uv run rag index` | Embeddings (GPU) + BM25 → Qdrant | Qdrant + `data/processed/embeddings/` |
| 5 | `uv run rag smoke` | Each mode must find its own sentences (hit@5) | printed |

Useful flags for `parse`, `chunk` and `index`:

```powershell
uv run rag chunk --limit 10                # only the first 10 papers (quick trial)
uv run rag chunk --ids 2610.13563          # specific papers
uv run rag chunk --force                   # redo even if cached
uv run rag parse --workers 1               # in-process, for debugging
uv run rag index --recreate                # delete and rebuild the Qdrant collection (vectors come from the .npy cache)
```

**Adding new papers:** copy the PDFs into `data/raw/pdf/`, then run `parse` → `chunk` → `index`. Only the new papers get processed.

---

## 4. Search and ask (CLI)

```powershell
# raw search: compare the three retrievers
uv run rag search "runtime performance of patches that pass the tests"
uv run rag search "runtime performance of patches that pass the tests" --mode bm25
uv run rag search "runtime performance of patches that pass the tests" --mode dense -k 10
uv run rag search "frozen snapshots" --paper 2610.13573       # only inside one paper

# full RAG: rewrite → filter → retrieve → rerank → context → LLM answer with citations
uv run rag ask "In the MatQuery paper, how are the databases kept frozen?"
uv run rag ask "..." --retrieve-only     # plan + reranked chunks, no LLM answer
uv run rag ask "..." --show-context      # also print the exact sources the LLM receives
uv run rag ask "..." --no-rewrite        # skip query rewrite / metadata extraction
uv run rag ask "..." --no-rerank         # skip the reranker (to compare)
```

---

## 5. Evaluation (retrieval quality)

```powershell
uv run rag eval-gen -n 100                       # LLM writes 100 test questions → data/eval/synthetic.jsonl
# add your own questions to data/eval/manual.jsonl, ONE JSON object per line:
# {"question": "...", "paper_id": "2610.13617", "pages": [3, 4]}      ("pages" optional)

uv run rag eval --limit 10 --configs dense bm25 hybrid    # quick trial
uv run rag eval                                           # all configs, k = 5
uv run rag eval -k 10                                     # looser cut-off
```

Metrics: Accuracy, Precision, Recall, MRR and nDCG at chunk level and paper level, for each of `dense`, `bm25`, `hybrid`, `hybrid+rerank` and `full`.
Per-question rankings are saved to `data/eval/results/<timestamp>.json`.

---

## 6. Web app

```powershell
uv run rag serve                  # http://127.0.0.1:8000
uv run rag serve --port 8080      # another port
```

- The browser page shows: the progress line with a timer → the search plan → source cards → the answer streaming in. Click `[n]` to jump to that source.
- The server terminal logs each stage with timings (`[ask   6.1s] Searching 3 queries ...`).
- After changing `index.html`, press **Ctrl+F5** in the browser to reload it.

API endpoints: `GET /api/health`, `GET /api/ask?q=...&rewrite=true&rerank=true` (Server-Sent Events).

---

## 7. Settings (no code changes needed)

Set a value for **this terminal only**:

```powershell
$env:RAG_OLLAMA_MODEL = "qwen3:4b"       # set
Remove-Item Env:RAG_OLLAMA_MODEL          # unset
```

To make a value permanent, put the same line without `$env:` in `.env`, e.g. `RAG_OLLAMA_MODEL=qwen3:4b`.

| Setting | Default | Use it to |
|---|---|---|
| `RAG_OLLAMA_MODEL` / `RAG_REWRITE_MODEL` | `qwen3:8b` | Answer / rewrite with a smaller, faster model |
| `RAG_RELEASE_GPU_BEFORE_LLM` | `true` | `false` keeps the embedder and reranker loaded (faster, needs VRAM) |
| `RAG_OLLAMA_KEEP_ALIVE` | `0` | e.g. `30m` keeps the LLM loaded between questions |
| `RAG_ANSWER_TOP_K` | `8` | Chunks passed from the reranker to the context builder |
| `RAG_CONTEXT_MAX_TOKENS` | `3500` | Token budget for sources in the prompt |
| `RAG_CONTEXT_MIN_SCORE` | `0.05` | Reranker relevance (0–1) a chunk needs to reach the LLM |
| `RAG_EMBED_BATCH_SIZE` | `32` | Lower it if you get CUDA out of memory |
| `RAG_QDRANT_COLLECTION` | `arxiv_chunks` | e.g. `arxiv_dev` for experiments |

**Fast mode** (everything stays loaded; needs `ollama pull qwen3:4b`):

```powershell
$env:RAG_OLLAMA_MODEL = "qwen3:4b"; $env:RAG_REWRITE_MODEL = "qwen3:4b"
$env:RAG_RELEASE_GPU_BEFORE_LLM = "false"; $env:RAG_OLLAMA_KEEP_ALIVE = "30m"
uv run rag serve
```

---

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Cannot reach Qdrant at http://localhost:6333` / `WinError 10061` | Qdrant isn't running | Start Docker Desktop, then `docker compose up -d` |
| `[error] ... Is Ollama running` | Ollama stopped, or the model isn't pulled | Start the Ollama app; `ollama pull qwen3:8b` |
| `CUDA out of memory`, or the GPU suddenly gets very slow | VRAM full | `$env:RAG_EMBED_BATCH_SIZE = "16"`, or use qwen3:4b |
| Web page waits a long time | First question loads all models | Watch the timer and the server log; the last log line shows the stuck stage |
| "Waiting for the previous question to finish..." | Another tab is still answering | Wait, or close the other tab |
| `manual.jsonl line N is not a complete JSON object` | JSON spread over several lines | One `{...}` per line, no commas between lines |
| `paper_id ... has no chunks (typo?)` | Wrong paper id in `manual.jsonl` | Check the id in `data/raw/md/<id>.json` |
| `uv` complains about the lock file | `pyproject.toml` changed | `uv sync` |

---

## 9. Stop

```powershell
# web server: Ctrl+C in its terminal
docker compose down      # stop Qdrant (data stays in the Docker volume)
ollama stop qwen3:8b     # free the LLM's VRAM right away
```
