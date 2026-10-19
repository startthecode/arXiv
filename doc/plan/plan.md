# RAG Data Ingestion Plan: structure-aware + recursive chunking, hybrid BM25 + dense in Qdrant

## Context
This is a new RAG app at `Z:\Study\AI\RAG`. The only existing content is the corpus: 1,050 arXiv PDFs in `data/raw/pdf/` (2.1 GB, about 23.9k pages, averaging 22 pages per paper). Every PDF has a text layer. 945 of them (90%) have a PDF outline, mostly 2 to 3 levels deep. Only 286 outlines list "References". The raw text has line-break hyphenation (`ma-\nterials`), stray U+FFFD characters and math-italic glyphs. `data/raw/md/` and `data/processed/` are empty.

This stage builds the ingestion pipeline: **PDF → markdown → structure-aware + recursive chunks → dense vectors and BM25 sparse vectors in one Qdrant collection**. The next stage, hybrid retrieval, can then run over it. A small hybrid query is included only to check that ingestion worked.

**Decisions made with the user:**
- pymupdf4llm for parsing.
- A local GPU embedding model.
- Qdrant in Docker.
- Plain Python (langchain-text-splitters only for the recursive splitter).

**Findings checked against current docs (2026-10-17) that shape the design:**
- pymupdf4llm 1.28.2 runs in *layout mode* by default, and that mode silently ignores `TocHeaders`/`hdr_info`. So heading levels are reassigned from the PDF outline in the chunk stage, not by the parser.
- Qdrant 1.19.2 computes BM25 on the server: a sparse vector with `modifier=IDF`, filled from `models.Document(text, model="Qdrant/bm25")`. fastembed is not needed, and Qdrant local mode should not be used.
- RTX 5060 (Blackwell, sm_120) → torch 2.14.1 from the **cu130** index (cu128 stops at 2.11). The global torch is the CPU-only build, so the project needs its own venv.
- Embedding model: **Qwen/Qwen3-Embedding-0.6B** (1024 dimensions, Apache-2.0, fits in 8 GB). Queries need an instruction prompt; documents don't. `BAAI/bge-m3` can be swapped in through config.

## Prerequisites (one-time setup)
- C: has only **6.5 GB free**, so move the large caches to Z: first:
  - `setx UV_CACHE_DIR Z:\.cache\uv`
  - `setx HF_HOME Z:\.cache\huggingface`
  - `setx PYTHONUTF8 1` (the cp1252 console crashed on Unicode during exploration)
  - Docker Desktop → Settings → Resources → Disk image location → `Z:\DockerDesktop`.
- **Start Docker Desktop** (the daemon is not running now).
- NVIDIA Control Panel → CUDA Sysmem Fallback Policy → "Prefer No Sysmem Fallback" for python.exe. This stops VRAM from silently spilling into system RAM.

## Project layout
```
pyproject.toml          # uv project; cu130 torch index; console script `rag-ingest`
docker-compose.yml      # qdrant/qdrant:v1.19.2, named volume (Qdrant warns against Windows bind mounts)
.env.example, .gitignore
src/rag/
  settings.py   # pydantic-settings (RAG_ prefix): paths, parse/chunk/embed/qdrant params, per-stage settings hash
  cli.py        # argparse: parse | chunk | index | all | stats | smoke   (--ids --limit --force --workers --recreate)
  parse.py      # PDF -> data/raw/md/{id}.md (+ "<!-- page: N -->" markers) + {id}.json sidecar; spawn ProcessPool
  clean.py      # pure text fixes: NFKC, U+FFFD ligature repair, dehyphenation, noise/boilerplate removal
  sections.py   # md -> page-tagged blocks -> heading re-leveling (TOC/numbering/font) -> section tree + classification
  chunk.py      # tiny-section merge -> recursive split -> chunk JSONL; tokenizer-based counting
  embed.py      # SentenceTransformer (cuda, bf16), encode_document with .npy cache
  index.py      # ensure collection + payload indexes, select changed papers, delete-then-upsert per paper
  stats.py      # sanity metrics over chunks and Qdrant counts -> data/processed/stats.json
  smoke.py      # hybrid query (dense + bm25 prefetch, RRF) + known-item self-retrieval check
tests/test_clean.py, test_sections.py, test_chunk.py
```

**Dependencies:**
- `pymupdf4llm==1.28.2` (pins pymupdf and pymupdf-layout; AGPL, which is fine for study use)
- `torch==2.14.1` (from the explicit cu130 index via `[tool.uv.sources]`)
- `sentence-transformers>=6.1,<7`, `transformers>=5.20,<6`
- `langchain-text-splitters>=1.1.3,<1.2`
- `qdrant-client>=1.19.1,<1.20` (no fastembed extra)
- `pydantic-settings`, `numpy`, `tqdm`
- dev: `pytest`

## Stage 1: `parse` (CPU, cached, about 10–60 min)
- Call `pymupdf4llm.to_markdown(doc, page_chunks=True, header=False, footer=False, use_ocr=False, force_text=True)`.
- Join the pages with `<!-- page: N -->` marker lines and write the result to `data/raw/md/{id}.md`.
- This stage applies no heuristics on purpose, so cleanup rules can change without re-parsing.
- Sidecar `{id}.json` fields: `paper_id`, `pdf_sha256`, `title`, `authors` (split on `;`), `keywords`, `page_count`, `toc` (`doc.get_toc()`), `parser` version and options hash, `seconds`, and captured MuPDF warnings.
- Workers:
  - `ProcessPoolExecutor(spawn, ~12 workers, max_tasks_per_child=25)`. Processes, because pymupdf is not thread-safe.
  - The worker init silences MuPDF stderr with `TOOLS.mupdf_display_errors(False)` and sets `OMP_NUM_THREADS=1`.
- Writes are atomic (`.tmp` then `os.replace`). Failures go to `_failures.jsonl` and the run continues.
- Skip a paper if it is already cached with the same parser options, unless `--force`.
- `RAG_PARSE_LAYOUT=false` falls back to legacy mode with `TocHeaders`/`IdentifyHeaders`.

## Stage 2: `chunk` (CPU, cached, about 1–3 min): structure-aware + recursive
1. **Blocks.** Split on blank lines. Tables, fenced code and picture-text spans stay whole. Page markers set each block's `page_start`/`page_end` and are then dropped.
2. **Clean** (`clean.py`; uses a vocabulary of the paper's own words):
   - Apply NFKC. Repair U+FFFD by trying the ligatures ff/fi/fl/ffi/ffl against the vocabulary.
   - Dehyphenate `word-\nword` when the joined word appears in the paper; keep the hyphen in compounds like `state-of-the-art` and `pre- and post-`.
   - Merge a paragraph that runs across a page or column break.
   - Attach `Table N:` captions to their table.
   - Drop picture text, the arXiv margin stamp, lines that are only equation or page numbers, and ACM/IEEE boilerplate.
   - Keep figure and table captions; they carry a lot of retrieval signal.
3. **Re-level headings** (`sections.py`). `norm()` strips numbering, markdown markup and case. Sources in priority order:
   - (a) Align headings with the PDF outline, using the outline's levels when it has 1–4 levels and at least 3 entries. Match in order within a window of the next 8 outline entries, on equal `norm`, a difflib ratio of at least 0.9, or a prefix match, with the page within ±1. Outline entries missing from the markdown promote the matching bold line to a heading.
   - (b) Numbering depth (`3.2.1` → level 3; IEEE `II.` / `A.`).
   - (c) Well-known top-level names (abstract, introduction, conclusion, references, …).
   - (d) The layout `#` count, capped at the previous level + 1.
   - Headings longer than 15 words, ending in `.`, or starting with `Figure/Table N` are demoted to text.
4. **Section tree and classification.**
   - Build each section's path with a stack.
   - **Drop** References/Bibliography up to the next heading at the same or higher level (usually Appendix A).
   - **Drop** acknowledgments, funding, author contributions, competing interests and checklists.
   - Mark appendix sections `section_type="appendix"` and keep them.
   - Find the abstract from a heading, an inline `**Abstract.**`, or the front matter; it becomes its own chunk.
5. **Tiny-section merge.** Greedily merge siblings under the same parent while the combined size fits and either side is under `min_tokens`. Merged sections keep inline `**2.1 Foo**` lines and get the breadcrumb `parent > 2.1 Foo / 2.2 Bar`. The abstract is never merged.
6. **Recursive split** of sections over budget.
   - `header = f"{title}\n{' > '.join(section_path)}\n\n"`; `budget = max_tokens - tokens(header)`.
   - Oversize tables are split by row, repeating the header row in each piece.
   - Then `RecursiveCharacterTextSplitter(chunk_size=budget, chunk_overlap=64, length_function=<embedding model tokenizer>, separators=["\n\n","\n",". ","? ","! ","; ",", "," ",""], keep_separator="end")`, which splits by paragraph, then line, then sentence, then word.
   - Each chunk is located back in the section text to get its exact page span.
7. **Final pass.**
   - `embed_text = header + text`, used for both dense and BM25. Raw `text` is kept for display.
   - Drop chunks under 24 tokens.
   - Drop bibliography-like leftovers: chunks where at least 60% of blocks match at least 2 citation patterns. This catches papers whose References heading was missed.
   - Remove duplicates within a paper.
   - Number chunks in document order; the abstract is `chunk_index` 0.
   - Assert every chunk is ≤ 512 tokens.

**Defaults** (all set in `settings.py`):

| Setting | Value | Why |
|---|---|---|
| `max_tokens` | 512 | About one sub-argument of a paper. Keeps BM25 length normalization meaningful. |
| `overlap` | 64 | Applied only inside sections that get split. |
| `min_tokens` | 128 | Below this, the chunk's vector is dominated by its header. |
| `drop_tokens` | 24 | Removes leftover fragments. |

Expected volume: about 30–45k chunks.

**Output:** `data/processed/chunks/{id}.jsonl`, one record per chunk with these fields:
- `chunk_id` (`{paper_id}:{idx:04d}`), `paper_id`, `chunk_index`
- `title`, `authors`
- `section_path`, `breadcrumb`, `section_level`, `section_type` (abstract/body/appendix), `section_top`
- `page_start`, `page_end`
- `text`, `embed_text`, `n_tokens`

The chunk settings hash (which includes the embedding model, because counts depend on its tokenizer) triggers a re-chunk when settings change.

## Stage 3: `index` (GPU + Qdrant, about 25–40 min cold)
- **Collection** `arxiv_chunks`:
  - dense vector `"dense"` (1024 dimensions, COSINE)
  - sparse vector `"bm25"` (`modifier=IDF`)
  - KEYWORD payload indexes on `paper_id`, `section_type`, `section_top`; INTEGER index on `section_level`
- **Change detection:** `data/processed/index_state.json` stores the sha256 of each paper's chunk JSONL. Skip a paper when the sha matches and Qdrant's per-paper count (`client.facet(..., key="paper_id", exact=True)`) equals its chunk count. A mismatch between the stored model or collection and the current one refuses to run unless `--recreate`.
- **Embedding:**
  - Load `SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cuda", model_kwargs={"dtype": torch.bfloat16}, processor_kwargs={"padding_side": "left"})`.
  - Call `encode_document` with batch size 64 over groups of about 4k chunks, so length sorting keeps padding low. Halve the batch on OOM.
  - Cache vectors as `data/processed/emb/<model>/{id}-{sha12}.npy` (float16), so rebuilding Qdrant doesn't re-embed.
- **Upsert per paper:**
  - Delete the paper's existing points with a `paper_id` filter, then upsert in batches of 256. This leaves no stale points after a re-chunk.
  - Point id = `uuid5(NS, f"{paper_id}:{chunk_index}")`.
  - Vectors: `{"dense": vec, "bm25": models.Document(text=embed_text, model="Qdrant/bm25", options=Bm25Config(avg_len=256))}`.
  - Payload: the chunk fields minus `embed_text`, plus `arxiv_url`.
  - Update the state file only after the paper succeeds.

## Verification
0. **Parser trial run (about 1 hour, before writing the stage modules).** Parse `2610.13563` (has an outline), `2610.13663` (no outline, numbered headings), `2610.13678` (no outline, unnumbered headings) and `2610.15567` (outline 28 levels deep). Run each in both layout and legacy+TocHeaders modes and compare:
   - heading output
   - two-column reading order
   - header/footer removal
   - which hyphenation form appears
   - seconds per page

   This fixes the `parse_layout` default and the noise regexes.
1. **Unit tests** (`uv run pytest -q`, run on synthetic markdown with a whitespace token counter):
   - dehyphenation cases
   - NFKC and U+FFFD repair
   - page spans across markers
   - outline re-leveling with unnumbered outline titles
   - numbering and IEEE levels
   - References dropped while the appendix is kept
   - leftover bibliography with no heading
   - abstract detection (heading and inline)
   - tiny-sibling merge
   - a 1,500-token section splits into chunks ≤ max, with overlap
   - tables kept whole; oversize tables split with the header repeated
   - fenced code kept whole
   - deterministic ids
2. **CUDA check:** `uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability())"` should print `2.14.1+cu130 True (12, 0)`.
3. **Qdrant:** run `docker compose up -d`; the dashboard is at http://localhost:6333/dashboard.
4. **10-paper subset** into a dev collection:
   ```
   $env:RAG_QDRANT_COLLECTION="arxiv_dev"; uv run rag-ingest all --limit 10
   uv run rag-ingest stats
   uv run rag-ingest smoke
   ```
   Hand-read the chunks for 3 papers.
5. **Idempotency:**
   - A re-run skips every stage, and the point count is unchanged.
   - `index --force --ids 2610.13563` leaves no duplicates.
   - Changing `chunk_max_tokens` re-chunks and leaves no stale points.
6. **`--limit 100` timing run.** Measure pages/s and tokens/s, and tune the worker count (try 8, 12 and 16).
7. **Full run, then `stats` and `smoke`.**

   `stats` must meet these thresholds:
   - parse failures < 0.5%
   - 0 papers with no chunks
   - abstract chunk present in ≥ 95% of papers
   - 0 chunks over 512 tokens
   - ≤ 5% of chunks under 64 tokens
   - section path present on ≥ 95% of chunks
   - References detected in ≥ 90% of papers
   - bibliography-like chunks ≤ 0.5%
   - Qdrant point count = chunk count

   `smoke` must meet these known-item thresholds (50 sentences sampled from chunks and used as queries, hit@5):
   - BM25 ≥ 0.9
   - dense ≥ 0.85
   - RRF ≥ 0.9

   It also checks that a `paper_id` filter returns only that paper.

## Risks to watch
- Layout mode silently drops unsupported keyword arguments, so tests must assert that heading levels come out as expected.
- Display equations produce no text in 1.28.2.
- Windows spawn means worker functions must be top-level.
- Qdrant's RRF uses `k=2` by default (most implementations use about 60); tune it in the retrieval stage.
- Changing the embedding model means re-chunking and a new collection.
- The BM25 options must be the same at ingest time and query time.





# Easy Explanation

Your project is a research-paper search engine powered by AI. You have 1,050 arXiv research papers in PDF format, and you want to turn them into a system where you can ask questions and retrieve the most relevant parts of those papers.

You don't need to understand every library or configuration immediately. First, understand what each stage does, why you need it, and how the data moves through your application.

## 1. What are you actually building?

Imagine you have 1,050 research papers on AI, machine learning, and other technical topics.

You ask your application:

> "How does hybrid search improve a RAG system?"

Your application should find the relevant passages from your papers and eventually use those passages to generate an answer with citations.

1,050 research PDFs

Original documents · 2.1 GB

PDF parsing + cleaning

Extract readable text and preserve document structure

Structure-aware chunking

Divide papers into useful, searchable passages

Two search representations

Dense vectors for meaning + BM25 for keywords

Qdrant vector database

Stores chunks and both search representations

Hybrid retrieval

Find the best matching research passages

This is the ingestion and retrieval foundation of a RAG application. Your current plan builds the pipeline through indexing and includes a small search test. It does not yet implement the final LLM answer-generation stage.

## 2. First, understand five important terms

[Global Change Biology Template - Wiley](https://images.openai.com/static-rsc-4/jGalcRo3FoR9_ZnfsRVngbXiGqfcFMh9wxo4jUB4rEZda--mCevNO1uv2JdvNBKzIcOudusBAjbMv7Odnhz1Tf4lRcjUMCKJCBJUOg9j15OjMQc16m5-U7AuFbGQfyIRGcLoYXO7e4iEDeiNmJTMqMZIr2mlH3evPCVo_gbjcvc?purpose=inline)

1\. Document

One complete research paper. You currently have 1,050 documents, each potentially containing many sections, tables, figures, and equations.

[floats - How can I replicate the format of this research paper? - TeX - LaTeX Stack Exchange](https://images.openai.com/static-rsc-4/kIRs_RChM_tJOuD5c4VxtOO09KA-Bf1pMoBwTXTqsSEDP-V2fDgHmJbic-TooKJV-oBwr0KZ7FU2G-bBgTFoZdrZXxrWgWUsI-FTrUJV8v0BTsd826WxEaEa69w4I5kKimEiSfujxMYC5Q9t5Ts04DcZ8clTigRXZwZUzHT6iI0?purpose=inline)

2\. Chunk

A smaller passage extracted from a paper. Instead of searching an entire 22-page paper, your application can retrieve the specific passage that explains the topic.

[Putting Superb Curate to the Test on the MNIST Dataset: How Does It Work?](https://images.openai.com/static-rsc-4/uth8s-PKwximOVnIuiNtr_I_kYqCy2Uj3DI4TvPii2FdeaYU2jJARU_ribH6U6RJfoEE1Zeri12ZuIW2T4VdMNTJ_wLyG392AfomVQfzE-lpd_9wxam0W6JgxIGCewRqgs3slTz8i5ray1YD4jeOc9EA0Dla2mM359fTpKjf3CM?purpose=inline)

3\. Embedding

A list of numbers representing the meaning of a chunk. Your model, Qwen3-Embedding-0.6B, converts text into a 1,024-dimensional vector.

[Database Network](https://images.openai.com/static-rsc-4/OJqg-MWnG1PQZpDhrMZpeHz1HEqowW1txcspfDXjbtNwSiAgtMtCf5TVf15q8kIAtaltiy-gpuCHWPNBxWXqhIsUgjby8vOMduROdlFtAjyyrjUnGCtyHHB-fWKrqHO5iFt0M33VYrPrHHE0Sql3cWjbuARyeUjVher1gLmG9NVmw9l57_Q_XKW6Pxf5Mz-Y?purpose=inline)

4\. Vector database

Qdrant stores chunks, their metadata, and their search representations. It lets you find matching chunks without reading all 1,050 papers for every question.

[Answer Your Questions With these 6 AI Read PDF Tools](https://images.openai.com/static-rsc-4/4lP9XxutIkDGkpmAp4c0iBx-UFYr8COxvjJ4rcyzga9bxmRUMLZjujGxiiLSs6NWv3CaEaMwZe9ncDPiesF1h3_KSM3DY1DDh8qM6RwsBRg6OwM0DeT3BpCO_1dFqwlzEqbBQocuhGHn2P4DaIxQhq08MixgXXZB2WPCRNV1BjE?purpose=inline)

5\. Retrieval

Given a question, search for relevant chunks and return the best matches. RAG uses these retrieved passages as evidence for an LLM to generate an answer.

One important distinction: Qdrant is not your LLM, and embeddings are not answers. They help the application find evidence that a language model can use later.

## 3. Stage 1 — PDF to Markdown (`parse.py`)

Your input files are PDFs. Before searching their contents, you need to extract their text.

You chose `pymupdf4llm` for this job.

### Why convert a PDF to Markdown?

A PDF is designed primarily for visual presentation. It stores text at page positions, along with fonts, columns, headings, and other layout information. Markdown is a much easier format for your Python code to process.

For example, a PDF might visually show:

SIMPLIFIED PDF CONTENT

### 3.2 Hybrid Search

Hybrid search combines lexical retrieval and semantic retrieval to improve search quality.

The parser can produce Markdown such as:

```
### 3.2 Hybrid Search

Hybrid search combines lexical retrieval and
semantic retrieval to improve search quality.
```

Markdown makes it easier to identify headings, paragraphs, lists, tables, and code-like blocks.

### What does your parser do?

Your `parse.py` module will:

1. Open a PDF.
2. Extract the text page by page.
3. Preserve useful formatting as Markdown.
4. Add page markers so you can later trace a chunk to its original page.
5. Save the Markdown and a JSON metadata file.

For example, your folder will eventually contain:

```
data/
├── raw/
│   ├── pdf/
│   │   └── 2610.13563.pdf
│   └── md/
│       ├── 2610.13563.md
│       └── 2610.13563.json
```

The `.md` file contains the extracted text. The `.json` sidecar contains information about the paper and how it was parsed, such as its title, authors, PDF outline, and parser version.

### What are page markers for?

Suppose a passage starts on page 4 and continues onto page 5.

Your parser adds markers similar to:

```
<!-- page: 4 -->

The first part of the passage...

<!-- page: 5 -->

The continuation of the passage...
```

These markers are not intended to appear in the final answer. Your chunking stage uses them to record page numbers, allowing you to link retrieved evidence back to the original research paper.

### Why do you cache the Markdown?

PDF extraction takes time. If you rerun your application without changing the parser settings or the PDF, it should reuse the existing Markdown instead of parsing the document again.

Important: Markdown conversion does not guarantee perfect extraction. Equations, two-column reading order, tables, and unusual fonts can still cause problems. That is why your plan includes a four-paper parser trial before processing all 1,050 papers.

## 4. Stage 2 — Clean the extracted text (`clean.py`)

Your source PDFs contain extraction problems such as broken words, replacement characters, and unnecessary page text.

If you embed dirty text, the search system may retrieve less useful results. Cleaning improves the quality of the text you use downstream.

### Example 1: Broken words

The PDF extraction might produce:

```
The proposed ap-
proach improves retriev-
al quality.
```

Your cleaner tries to reconstruct:

```
The proposed approach improves retrieval quality.
```

But it must not blindly remove every hyphen. Consider:

```
state-of-the-art
```

That should remain hyphenated.

Your plan handles this by checking whether the joined word appears in the paper's vocabulary.

### Example 2: Strange characters

An extracted word might contain a replacement character:

```
ef�cient
```

Your cleaner attempts to repair this using likely ligature substitutions, such as `fi`, when supported by the vocabulary.

It also applies Unicode normalization (`NFKC`) to standardize compatible text characters.

Not every damaged character can be reconstructed confidently. Your implementation should avoid inventing text when there is insufficient evidence to repair it.

### Example 3: Unnecessary content

A PDF may contain:

- Repeated journal or conference footers.
- Page numbers extracted as standalone text.
- arXiv margin stamps.
- Repeated publisher notices.

These usually do not help answer research questions, so your cleaner removes known noise patterns.

However, figure captions and table captions should generally stay. They often contain important scientific details that do not appear elsewhere in the prose.

Your cleaning code should be conservative: remove clearly identified noise, not legitimate research content.

## 5. Stage 3 — Understand the document structure (`sections.py`)

This is one of the most important parts of your design.

You do not want to divide every paper into arbitrary blocks of text. Research papers already have a meaningful structure:

```
Paper
├── Abstract
├── 1. Introduction
├── 2. Related Work
├── 3. Methodology
│   ├── 3.1 Architecture
│   └── 3.2 Training Procedure
├── 4. Experiments
├── 5. Conclusion
└── References
```

Your code should preserve this structure so that each chunk knows which section it came from.

### What is a PDF outline?

Many PDFs contain a navigation outline, similar to a table of contents. Your corpus has outlines for 945 of its 1,050 papers.

An outline may tell you that `3.2 Training Procedure` is a subsection of `3. Methodology`.

But your Markdown parser might not reliably preserve those heading levels. Your plan therefore uses the PDF outline to correct the headings during chunking.

For papers without a usable outline, your code falls back to signals such as:

- Numbered headings (`3.2.1`).
- Common section titles (`Abstract`, `Introduction`, `Conclusion`).
- Markdown heading levels.
- Heading formatting and position.

For example, `3.2.1` generally suggests level 3, while `3.2` suggests level 2.

These are clues, not guarantees. Your parser trial and unit tests are important because incorrectly classified headings can damage the entire section hierarchy.

### Why remove the References section?

Suppose a paper ends with 50 bibliography entries.

If you index those entries like ordinary research content, a query containing an author's name or a technical term could retrieve a bibliography rather than an explanation of the concept.

Your plan drops References and Bibliography sections while preserving appendices. It also removes certain acknowledgments and funding sections.

This is a deliberate corpus-design decision: you want to search the scientific content, not primarily the administrative material and citations.

One subtle point: your corpus only has an explicit `References` outline entry for 286 papers. That does not mean the other papers lack references. Your code must also detect references through Markdown headings and other patterns, and check for bibliography-like leftovers when a heading was missed.

## 6. Stage 4 — Chunk the paper (`chunk.py`)

Now the extracted text is clean enough to split into passages.

Your strategy is structure-aware chunking with recursive fallback. Let's break that down.

### A. Structure-aware chunking

First, split the document according to its natural structure: sections, subsections, paragraphs, and special blocks such as tables.

For example:

```
3. Methodology
    ↓
3.1 Model Architecture
    ↓
3.2 Training Procedure
```

If `3.1 Model Architecture` is short enough, it may become one chunk. The next section should remain separate rather than being mixed into the first chunk unnecessarily.

This is better than splitting the whole PDF every 500 tokens, because the resulting chunks retain the topic boundaries established by the author.

### B. Recursive splitting

What if one section contains 2,000 tokens? It is too large for your 512-token limit.

Your recursive splitter attempts to divide it in this order:

1\. Paragraph boundaries

Keep complete paragraphs when possible.

2\. Line boundaries

Split on newlines if paragraphs are too large.

3\. Sentence boundaries

Split at sentence-ending punctuation.

4\. Smaller text boundaries

Split further when the text still exceeds the budget.

The aim is to preserve meaning for as long as possible before making smaller cuts.

### C. What is chunk overlap?

Your plan sets an overlap of 64 tokens when a section must be split.

Imagine a paragraph is split into two chunks:

Chunk A

...the model uses a contrastive objective to learn semantic representations...

64-token overlap

Chunk B

...learn semantic representations that improve retrieval across related research topics...

The repeated text helps preserve context near the boundary. Without overlap, a sentence split across two chunks may be harder to interpret when retrieved individually.

Overlap does have a cost: it creates repeated text, more embeddings, and potentially duplicate retrieval results. Your 64-token setting is a reasonable starting point to evaluate, not a universal optimum.

### D. Why is 512 tokens your limit?

A token is a unit of text processed by a language model. It is not exactly the same as a word.

Your maximum of 512 tokens limits the size of each chunk. This helps keep passages focused and makes the retrieval results easier to use.

Your plan also includes:

- `min_tokens = 128`: a threshold used when deciding whether small sibling sections should be merged.
- `drop_tokens = 24`: discard very short fragments that may have little value.

These are different settings. The 128-token threshold does not mean every final chunk must be at least 128 tokens, because a legitimate small section may remain smaller after merging decisions. The final 24-token filter removes the shortest fragments.

### E. Why include the heading in the chunk?

Suppose the original passage says:

```
It reduces computational overhead and improves convergence.
```

Without context, this passage is ambiguous. What does "it" refer to?

Your plan prepends a heading and breadcrumb:

```
3.2 Training Procedure
Methodology > Training Procedure

It reduces computational overhead and improves convergence.
```

The resulting `embed_text` gives the search system additional context about the passage. Your original `text` remains available for displaying the retrieved evidence.

This is a good design choice, but keep the header within the 512-token budget. If a heading and breadcrumb consume too many tokens, you must reduce the available text budget accordingly.

## 7. What is the difference between `text` and `embed_text`?

Your chunks contain two versions of the content.

| Field                    | Purpose                                                         |
| ------------------------ | --------------------------------------------------------------- |
| `text`                   | Original cleaned passage shown to the user                      |
| `embed_text`             | Heading + section breadcrumb + passage used for search indexing |
| `n_tokens`               | Token count used to enforce the chunk limit                     |
| `section_path`           | Hierarchical location within the paper                          |
| `page_start`, `page_end` | Original PDF pages containing the passage                       |

For example, a chunk might look like this:

```
{
  "chunk_id": "2610.13563:0004",
  "paper_id": "2610.13563",
  "chunk_index": 4,
  "section_path": [
    "3. Methodology",
    "3.2 Training Procedure"
  ],
  "text": "The model is trained using a contrastive objective...",
  "embed_text": "3.2 Training Procedure\nMethodology > Training Procedure\n\nThe model is trained using a contrastive objective...",
  "n_tokens": 186,
  "page_start": 5,
  "page_end": 6
}
```

This is an illustrative example of the schema, not a record extracted from your actual PDF.

Why preserve both? The search representation benefits from extra context, while the displayed passage should not contain unnecessary repeated headings.

## 8. Stage 5 — Dense search using Qwen3 embeddings (`embed.py`)

Now your chunks are ready to become searchable by meaning.

You selected `Qwen/Qwen3-Embedding-0.6B` as your local embedding model.

### What does an embedding model do?

An embedding model converts text into a list of numbers called a vector.

For illustration, a short passage might produce a vector like:

```
"Hybrid search combines BM25 and embeddings."

→ [0.13, -0.42, 0.08, 0.71, ...]
```

The real vector from your chosen model has 1,024 dimensions.

The key idea is that passages with related meanings often have vectors that are close together in vector space.

For example:

- Query: "How can I combine keyword and semantic retrieval?"
- Passage: "Hybrid retrieval integrates lexical matching with dense vector similarity."

The words are different, but their meanings are related. Dense retrieval can find that relationship.

### How does the query get matched?

Question

"How do I improve semantic retrieval?"

Qwen3 embedding model

Converts the question into 1,024 numbers

Compare against stored vectors

Find nearby vectors using cosine similarity

Relevant chunks

Ranked passages from your research papers

Your plan uses cosine similarity, a measure of the angle between two vectors. The closer the vectors point in a similar direction, the higher their cosine similarity.

### Why run this locally on your GPU?

You have an NVIDIA RTX 5060 Laptop GPU. Using it for embeddings lets you process batches of chunks locally instead of sending every passage to a paid embedding API.

Your planned settings include:

- GPU inference with CUDA.
- BF16 model weights where supported.
- Batches of 64 chunks initially.
- Smaller batches if GPU memory runs out.
- A cache of saved vectors so you do not need to recompute them unnecessarily.

The `.npy` files in your embedding cache contain numeric vectors. They are not the searchable database themselves; Qdrant will store the vectors used for retrieval.

### Why does Qwen3 need an instruction prompt for queries?

Some embedding models work better when the query describes what the model is trying to retrieve. Qwen3 supports instruction-aware query embeddings.

Conceptually, you may encode a question using an instruction like:

```
Instruct: Retrieve relevant research passages
that answer the user's question.
Query: How does hybrid search work?
```

Your actual code should follow the model's documented query format and use its `encode_query`/`encode_document` functionality correctly. Document embeddings should not automatically receive the same query instruction.

Important: Query and document embeddings must be compatible. Use the same embedding model and its prescribed encoding conventions on both sides. Changing the model later generally requires rebuilding the dense vectors and revalidating the index.

## 9. Stage 6 — Why you need both BM25 and dense search

This is the core of your hybrid retrieval design.

Dense retrieval is good at finding related meanings. BM25 is good at matching important words and terms.

Neither approach is best for every kind of question.

## BM25 — keyword search

BM25 is a lexical ranking algorithm. It rewards useful term matches, accounting for how frequently a term occurs and how common it is across the indexed corpus.

Example query: `Qwen3-Embedding-0.6B`

BM25 is useful for exact model names, technical identifiers, acronyms, error messages, and specific terminology.

## Dense retrieval — semantic search

Dense retrieval compares the query's embedding with the stored chunk embeddings.

Example query: `How can a system find passages with similar meaning?`

It is useful when the question and the relevant passage express the same idea using different wording.

### See why both are useful

Suppose your paper contains this sentence:

> "We use Reciprocal Rank Fusion to combine lexical and semantic search results."

You ask: "What technique combines BM25 and dense retrieval?"

- BM25 may match the words `BM25` and `dense retrieval` particularly well if they appear in the passage.
- Dense retrieval may find the passage even if the wording differs from your question.
- Hybrid retrieval combines the rankings to give you a stronger overall result.

That is why your design stores both search representations in the same Qdrant collection.

### What does Qdrant store?

Your collection will be called `arxiv_chunks`. Each point represents one chunk.

Point ID

A stable UUID derived from the paper ID and chunk index.

Dense vector

1,024-dimensional embedding generated by Qwen3.

Sparse BM25 vector

A sparse representation computed by Qdrant's server-side BM25 model.

Payload

Paper ID, title, section path, page numbers, chunk text, and other metadata.

The advantage of storing both representations in one collection is that you can search the same chunks using either method and combine their results without maintaining two separate databases.

### What is the Qdrant BM25 configuration doing?

Your plan uses:

```
models.Document(    text=embed_text,    model="Qdrant/bm25",    options=Bm25Config(avg_len=256),)
```

This tells Qdrant to create the BM25 representation from your chunk text. The sparse vector configuration uses `modifier=IDF`, which allows Qdrant to apply inverse document frequency weighting on the server side.

That is why you do not need to install a separate BM25 embedding model through FastEmbed for this design.

One detail to verify against the pinned Qdrant client and server versions is that the collection's sparse-vector configuration and the `models.Document` input are wired up exactly as expected. The average document length is also a tuning parameter; `avg_len=256` is a starting assumption to validate, not necessarily the measured average length of your chunks.

BM25 and dense vectors serve different purposes, but both must be derived from the same intended searchable content so that they index the same chunk consistently.

## 10. Stage 7 — Store everything in Qdrant (`index.py`)

Now your application has chunks, and it can generate embeddings. The next job is to store those chunks so retrieval can use them.

### What is Qdrant?

Qdrant is a database designed for vector search. Your application connects to a Qdrant server running in Docker.

The basic setup looks like this:

Python ingestion application

Chunks + embeddings + metadata

Docker Desktop

Runs the Qdrant server container

Qdrant collection: `arxiv_chunks`

Vectors, payloads and searchable indexes

Persistent Docker volume keeps the database data across container restarts.

Your Python code sends data to Qdrant through its client API. Qdrant performs database operations and search; it does not run your Qwen embedding model for you.

### Why use Docker?

Docker packages Qdrant as a service that you can start and stop independently of your Python application.

Your plan uses a named Docker volume rather than a Windows bind mount for the database's storage. This avoids relying on a Windows host folder for Qdrant's internal database files.

Once the service is running, the dashboard will be available at:

[http://localhost:6333/dashboard](http://localhost:6333/dashboard?utm_source=chatgpt.com)

This is a local development endpoint, not a public production deployment.

### Why create payload indexes?

Your payload contains metadata such as `paper_id`, `section_type`, and `section_top`.

Imagine asking:

> "Search for papers about hybrid retrieval, but only in appendices."

Or:

> "Return chunks only from paper 2610.13563."

Payload indexes help Qdrant filter by these fields efficiently. They complement vector search rather than replacing it.

### Why delete old chunks before inserting new ones?

Suppose a paper originally produces 30 chunks. After you improve the chunking algorithm, the same paper produces 25.

If you only insert the new 25 chunks, the old five may remain in the database. Those stale chunks could appear in search results.

Your plan deletes that paper's old points and then inserts its new chunks. It records the updated index state only after the paper has been processed successfully.

This is a sensible way to avoid stale records, although a crash between deletion and insertion can temporarily leave the paper absent or incomplete. For stronger production guarantees, you can later implement versioned points and an atomic alias switch.

### What is idempotency?

Idempotency means you can safely rerun an operation without creating unwanted duplicates or changing already-correct results.

Your ingestion pipeline tracks hashes of PDFs, parser settings, chunk settings, and chunk JSONL files. It can skip work when the relevant inputs have not changed.

For example:

- PDF unchanged and parser configuration unchanged → reuse Markdown.
- Chunk inputs and chunk configuration unchanged → reuse chunk files.
- Chunk JSONL unchanged and Qdrant point count matches → skip re-indexing.
- Chunking configuration changes → regenerate chunks and replace that paper's points.

This matters because processing 1,050 papers repeatedly would waste time and GPU resources.

## 11. Your actual folder structure, explained

Each Python file has a distinct responsibility. That separation makes the application easier to test, debug, and extend.

| File          | Responsibility                                                    |
| ------------- | ----------------------------------------------------------------- |
| `settings.py` | Central configuration for paths, models, chunk limits, and Qdrant |
| `cli.py`      | Command-line entry point for running stages                       |
| `parse.py`    | PDF extraction into Markdown                                      |
| `clean.py`    | Repair extraction errors and remove noise                         |
| `sections.py` | Detect headings, build the section hierarchy, classify sections   |
| `chunk.py`    | Merge small sections and split oversized sections                 |
| `embed.py`    | Generate and cache dense vectors                                  |
| `index.py`    | Insert or update chunks in Qdrant                                 |
| `stats.py`    | Calculate ingestion quality metrics                               |
| `smoke.py`    | Run basic retrieval tests                                         |
| `tests/`      | Test important edge cases automatically                           |

The data directories serve different purposes:

```
Z:\Study\AI\RAG\
├── data/
│   ├── raw/
│   │   ├── pdf/        # Original 1,050 PDFs
│   │   └── md/         # Parsed Markdown and JSON sidecars
│   └── processed/
│       ├── chunks/     # Chunk JSONL files
│       ├── emb/        # Cached dense embeddings
│       ├── stats.json
│       └── index_state.json
├── src/rag/
├── tests/
├── pyproject.toml
└── docker-compose.yml
```

The original PDFs remain your source of truth. The Markdown is an intermediate representation, the JSONL files are your chunk dataset, and Qdrant is the searchable index.

## 12. Stage 8 — Verify that ingestion actually worked

A pipeline can finish without errors and still produce bad search results. This is why your verification plan is valuable.

You have four major types of tests.

A. Unit tests

Test cleaning, heading detection, page spans, reference removal, abstract detection, recursive splitting, table handling, and deterministic IDs using small synthetic examples.

Goal: catch bugs in the code before running the whole corpus.

B. Ingestion statistics

Check parsing failures, empty papers, missing abstracts, chunk sizes, section paths, bibliography leftovers, and Qdrant point counts.

Goal: make sure the documents were processed correctly.

C. Smoke tests

Retrieve chunks using known passages as queries and verify that the expected chunks appear in the results.

Goal: check that the database and both search methods work.

D. Performance tests

Process 10 papers first, then 100, and finally the full corpus. Measure pages per second, tokens per second, GPU memory, and processing failures.

Goal: find bottlenecks before committing to a full run.

### What is Hit\@5?

Your smoke test uses 50 sentences sampled from your chunks as queries.

For each query, the expected original chunk is known. Hit\@5 measures whether that chunk appears anywhere in the top five retrieved results.

\\[ \operatorname{Hit\@5} = \frac{\text{Queries with expected chunk in top 5}} {\text{Total queries}} \\]

If the expected chunk is retrieved in the top five for 45 out of 50 queries, Hit\@5 is 0.90, or 90%.

Your target thresholds are:

| Search method | Target Hit\@5 |
| ------------- | ------------- |
| BM25          | ≥ 0.90        |
| Dense         | ≥ 0.85        |
| Hybrid RRF    | ≥ 0.90        |

These are useful smoke-test targets, but they do not prove that your system will answer arbitrary questions correctly. Because your queries are sampled from existing chunks, the test can favor exact or near-exact matches. Later, add manually written questions and expected relevant passages to measure real retrieval quality.

Also, RRF means Reciprocal Rank Fusion. It combines ranked result lists rather than comparing BM25 scores directly with cosine similarity scores.

## 13. Your implementation order

Do not try to build and run the entire system at once. Follow the order below.

## Implementation checklist

0 of 7

1\. Prepare the environment

Move caches to Z:, configure the Python environment, install dependencies, and start Docker.

2\. Validate the PDF parser

Test the four selected papers in layout and legacy modes; inspect headings, reading order, and extraction quality.

3\. Implement cleaning and section detection

Build the text fixes, outline alignment, heading hierarchy, and section classification.

4\. Implement chunking

Add tiny-section merging, recursive splitting, page spans, token limits, and JSONL output.

5\. Write and run unit tests

Validate edge cases before processing the full corpus.

6\. Implement embeddings and Qdrant indexing

Confirm CUDA, test vector generation, create the collection, and index ten papers.

7\. Test, benchmark, and scale up

Check idempotency, process 100 papers, tune throughput, and run the complete corpus.

Your command-line interface is intended to expose these operations:

```
uv run rag-ingest parse
uv run rag-ingest chunk
uv run rag-ingest index
uv run rag-ingest stats
uv run rag-ingest smoke
```

And, when you are ready to run the whole pipeline:

```
uv run rag-ingest all --limit 10
```

The `--limit 10` run should use a development collection, as in your plan. Start small, inspect the resulting chunks manually, and only then move to the full corpus.

## 14. A few things I would watch closely in your plan

Your architecture is reasonable for a local study project. I would pay particular attention to these implementation details:

1. Tokenizer consistency. Your 512-token limit must use the actual embedding model's tokenizer, including the final header and breadcrumb. The splitter's `chunk_size` is only as accurate as the supplied token-counting function. Assert the limit again after constructing `embed_text`.
2. PDF parsing quality. The PDF outline is useful, but heading matching can still be wrong. Inspect the four trial papers before finalizing heading heuristics, particularly the paper with a 28-level outline.
3. Tables and equations. Repeating table headers when splitting rows is sensible. But mathematical expressions that the parser cannot extract will be missing from both BM25 and dense retrieval. Track this as a corpus limitation rather than assuming that every paper's scientific content is fully represented.
4. Deletion and recovery. Delete-then-upsert is straightforward, but a failure between those steps can leave incomplete indexing. Ensure the state file is updated only after successful completion, and test recovery after deliberately interrupting a paper's indexing.
5. BM25 and RRF settings. Keep BM25 configuration consistent between indexing and query-time use. Explicitly configure and evaluate RRF's rank constant in the retrieval stage instead of blindly accepting the default.
6. GPU and cache space. Your 1,050 PDFs occupy 2.1 GB, but the Markdown, chunk files, embedding cache, and Qdrant database will consume additional space. Monitor the Z: drive and GPU memory during the 100-paper benchmark.
7. Quality metrics. A high Hit\@5 on sampled passages is useful, but it does not establish that the right evidence will be retrieved for natural-language questions. Build a small, manually reviewed evaluation set after ingestion.

## 15. What comes after this project stage?

Once the ingestion pipeline is complete, the next stage is to build the actual hybrid retrieval workflow:

User asks a question

Dense search

Semantic matches

BM25 search

Keyword matches

RRF fusion

Combine ranked results

Optional reranker

Improve ordering of candidate passages

LLM answer generation

Answer from retrieved evidence, with paper and page citations

That final LLM stage is where your application becomes a complete question-answering RAG system. You will then evaluate whether the retrieved evidence actually supports the answer, whether citations point to the correct pages, and whether the model admits when the corpus does not contain enough information.

The main thing to remember: your current task is not to train an AI model or generate answers yet. You are building a reliable, structured knowledge base from 1,050 research papers. The quality of that foundation will strongly influence the quality of every RAG feature you build afterward.
